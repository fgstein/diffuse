# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Modified: works without torchrun (plain `python train.py` = world size 1),
# picks the NCCL/Gloo backend based on CUDA availability, works on CPU, and
# supports graceful stop on SIGINT/SIGTERM.

import os
import signal
import torch
from . import training_stats

#----------------------------------------------------------------------------

_stop_requested = False

def _signal_handler(signum, _frame):
    global _stop_requested
    if _stop_requested:
        # Second signal: give up on graceful shutdown.
        raise KeyboardInterrupt
    _stop_requested = True
    print0(f'\nReceived signal {signum}; will dump training state and exit at the next iteration '
           f'(send again to abort immediately).', flush=True)

def install_signal_handlers():
    """Install SIGINT/SIGTERM handlers that request a graceful stop (see should_stop())."""
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

#----------------------------------------------------------------------------

def init(device_type=None):
    """Initialize (optional) multi-process training.

    - Launched via torchrun (WORLD_SIZE in the environment): initializes a process
      group with NCCL (CUDA) or Gloo (CPU).
    - Launched as a plain `python train.py`: no process group, world size 1.
    """
    if device_type is None:
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_cuda = (device_type == 'cuda') and torch.cuda.is_available()
    if 'WORLD_SIZE' in os.environ:
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', '29500')
        os.environ.setdefault('RANK', '0')
        os.environ.setdefault('LOCAL_RANK', '0')
        backend = 'nccl' if (use_cuda and os.name != 'nt') else 'gloo'
        torch.distributed.init_process_group(backend=backend, init_method='env://')
    if use_cuda:
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    sync_device = None
    if get_world_size() > 1:
        sync_device = torch.device('cuda') if use_cuda else torch.device('cpu')
    training_stats.init_multiprocessing(rank=get_rank(), sync_device=sync_device)

#----------------------------------------------------------------------------

def is_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def get_rank():
    return torch.distributed.get_rank() if is_initialized() else 0

def get_world_size():
    return torch.distributed.get_world_size() if is_initialized() else 1

def barrier():
    if is_initialized():
        torch.distributed.barrier()

#----------------------------------------------------------------------------

def should_stop():
    """True once a graceful stop was requested (SIGINT/SIGTERM), agreed across ranks."""
    flag = _stop_requested
    if is_initialized() and get_world_size() > 1:
        backend = torch.distributed.get_backend()
        dev = torch.device('cuda') if backend == 'nccl' else torch.device('cpu')
        t = torch.tensor([1.0 if flag else 0.0], device=dev)
        torch.distributed.all_reduce(t)
        flag = bool(t.item() > 0)
    return flag

#----------------------------------------------------------------------------

def update_progress(cur, total):
    _ = cur, total

#----------------------------------------------------------------------------

def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)

#----------------------------------------------------------------------------
