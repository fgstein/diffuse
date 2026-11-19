# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Modified for class-conditional pixel-space training: dict conditions,
# bf16 autocast, CPU support, multiple EMAs, LR decay, gradient clipping,
# fixed-sigma validation loss, per-sigma training loss, sample grids,
# exact resume, snapshot retention, graceful stop.

"""Main training loop."""

import os
import re
import glob
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from training.dataset import label_map_to_json
from training.loss import SIGMA_BIN_EDGES, sigma_bin_index
from training import sampling

#----------------------------------------------------------------------------
# Fixed validation noise levels (16 log-spaced sigmas in [0.002, 80]).

VAL_SIGMAS = np.exp(np.linspace(np.log(0.002), np.log(80), 16)).tolist()

@torch.no_grad()
def compute_val_loss(net, val_images, val_cond, sigma_data, device, batch_size=64, sigmas=VAL_SIGMAS, seed=12345):
    """EDM-weighted denoising loss at fixed sigmas with fixed noise (comparable across runs/ticks).
    Returns (mean over sigmas, list of per-sigma losses)."""
    net.eval()
    totals = np.zeros(len(sigmas), dtype=np.float64)
    count = 0
    for start in range(0, val_images.shape[0], batch_size):
        x = val_images[start:start + batch_size].to(device).to(torch.float32) / 127.5 - 1
        cond = {k: v[start:start + batch_size].to(device) for k, v in val_cond.items()}
        n = x.shape[0]
        for si, s in enumerate(sigmas):
            gen = torch.Generator().manual_seed(seed + 1000003 * start + si)
            noise = torch.randn(x.shape, generator=gen).to(device) * s
            sigma = torch.full([n], s, device=device)
            D = net(x + noise, sigma, cond)
            w = (s ** 2 + sigma_data ** 2) / (s * sigma_data) ** 2
            totals[si] += float((w * (D - x).square()).mean(dim=(1, 2, 3)).sum())
        count += n
    per_sigma = (totals / max(count, 1)).tolist()
    return float(np.mean(per_sigma)), per_sigma

#----------------------------------------------------------------------------

def _grid_classes(num_classes, max_classes):
    return list(range(min(num_classes, max_classes))) if num_classes > 0 else [None]

def save_reals_grid(dataset, path, cols=8, max_classes=16, seed=0):
    rnd = np.random.RandomState(seed)
    rows = []
    for c in _grid_classes(dataset.num_classes if dataset.has_labels else 0, max_classes):
        idx = dataset.indices_of_class(c) if c is not None else np.arange(len(dataset))
        if idx.size == 0:
            rows.append(np.full([cols] + dataset.image_shape, 255, np.uint8))
            continue
        pick = rnd.choice(idx, size=cols, replace=idx.size < cols)
        rows.append(np.stack([dataset[int(i)][0] for i in pick]))
    images = np.concatenate(rows)
    sampling.save_image(sampling.make_grid(images, len(rows), cols), path)

def save_sample_grid(net, path, device, num_classes, guidances, cols=8, max_classes=16, num_steps=32, sigma_max=80.0, batch_size=64):
    classes = _grid_classes(num_classes, max_classes)
    seeds = [s for _ in classes for s in range(cols)] # same seeds for every row
    cls = None if classes[0] is None else [c for c in classes for _ in range(cols)]
    grids = []
    for w in guidances:
        imgs = sampling.generate(net, seeds, cls, device=device, guidance=w, num_steps=num_steps,
                                 sigma_max=sigma_max, batch_size=batch_size)
        grids.append(sampling.make_grid(imgs, len(classes), cols))
    sampling.save_image(sampling.hconcat(grids), path)

#----------------------------------------------------------------------------

def _snapshot_kimg(path):
    m = re.search(r'-(\d+)\.(pkl|pt)$', os.path.basename(path))
    return int(m.group(1)) if m else None

def cleanup_snapshots(run_dir, keep_last, keep_every_kimg):
    """Keep only the last `keep_last` snapshots/states (+ those at multiples of keep_every_kimg)."""
    if keep_last is None or keep_last <= 0:
        return
    for pattern in ['network-snapshot-*.pkl', 'training-state-*.pt']:
        files = sorted(glob.glob(os.path.join(run_dir, pattern)), key=lambda p: (_snapshot_kimg(p) or 0))
        for p in files[:-keep_last]:
            k = _snapshot_kimg(p)
            if keep_every_kimg and k is not None and k % keep_every_kimg == 0:
                continue
            try:
                os.remove(p)
            except OSError:
                pass

#----------------------------------------------------------------------------

def _rng_state():
    st = dict(torch=torch.get_rng_state(), numpy=np.random.get_state())
    if torch.cuda.is_available():
        st['cuda'] = torch.cuda.get_rng_state_all()
    return st

def _set_rng_state(st):
    torch.set_rng_state(st['torch'])
    np.random.set_state(st['numpy'])
    if 'cuda' in st and torch.cuda.is_available() and len(st['cuda']) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(st['cuda'])

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    val_dataset_kwargs  = None,     # Options for validation set, None = no validation.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the primary EMA (kimg). Ignored if ema_list_kimg is given.
    ema_list_kimg       = None,     # List of EMA half-lives (kimg); the first one is the primary 'ema'.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    lr_decay            = 'none',   # 'none' | 'invsqrt' (EDM2-style: lr / sqrt(max(kimg / lr_ref_kimg, 1))).
    lr_ref_kimg         = 10000,    # Reference duration for invsqrt decay.
    grad_clip           = 0,        # Gradient norm clipping, 0 = off.
    loss_scaling        = 1,        # Loss scaling factor (only useful for fp16).
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    sample_ticks        = None,     # How often to save sample grids, None = disable.
    val_ticks           = None,     # How often to compute validation loss, None = disable.
    sample_guidance     = (1, 2, 4),# Guidance values for sample grids.
    sample_steps        = 32,       # Sampler steps for sample grids.
    sample_cols         = 8,        # Seeds per class row in sample grids.
    sample_max_classes  = 16,       # Max number of class rows in grids.
    val_max_images      = 1024,     # Max number of validation images used for val loss.
    keep_snapshots      = 10,       # Keep only the last N snapshots/states (0 = keep all).
    keep_every_kimg     = 0,        # Additionally keep snapshots at multiples of this kimg (0 = off).
    class_balanced      = False,    # Class-balanced sampling.
    balance_alpha       = 0.5,      # Weight exponent for class-balanced sampling.
    noise_config        = {},       # Stored in snapshots (P_mean_eff, sigma_max hint, resolution, patch size).
    resume_pkl          = None,     # Start from the given network snapshot (transfer), None = random init.
    resume_state_dump   = None,     # Resume from the given training state.
    resume_kimg         = None,     # Fallback progress if the state dump does not store it.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    tf32                = True,     # Allow TF32 matmuls/convs on Ampere+.
    compile             = False,    # torch.compile the training forward pass.
    tensorboard         = False,    # Log to TensorBoard (if installed).
    training_options    = None,     # Full resolved options (stored in snapshots).
    device              = torch.device('cuda'),
):
    # Initialize.
    start_time = time.time()
    device = torch.device(device)
    is_cuda = (device.type == 'cuda')
    rank0 = (dist.get_rank() == 0)
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    dist.install_signal_handlers()

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    if batch_size != batch_gpu * num_accumulation_rounds * dist.get_world_size():
        raise ValueError(f'--batch ({batch_size}) must be divisible by --batch-gpu ({batch_gpu}) x number of GPUs ({dist.get_world_size()}).')
    dist.print0(f'Batch: {batch_size} total, {batch_gpu} per GPU step, {num_accumulation_rounds} accumulation round(s).')

    # Load resume state early (needed for data order and progress).
    state = None
    cur_nimg = 0
    cur_tick = 0
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        state = torch.load(resume_state_dump, map_location=torch.device('cpu'), weights_only=False)
        cur_nimg = int(state.get('cur_nimg', (resume_kimg or 0) * 1000))
        cur_tick = int(state.get('cur_tick', 0))

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # training.dataset.CSVImageDataset
    label_map = dataset_obj.label_map if dataset_obj.has_labels else {}
    counts = dataset_obj.class_counts() if dataset_obj.has_labels else {}
    dist.print0(f'Training images: {len(dataset_obj)}  resolution: {dataset_obj.resolution}  channels: {dataset_obj.num_channels}  classes: {len(label_map)}')
    if counts:
        dist.print0('Class counts (train split): ' + ', '.join(f'{k}={v}' for k, v in counts.items()))
        nz = [v for v in counts.values() if v > 0]
        if len(nz) < len(counts):
            dist.print0(f'WARNING: {len(counts) - len(nz)} class(es) have no training images.')
        if nz and max(nz) / min(nz) > 10:
            dist.print0(f'WARNING: strong class imbalance (max/min = {max(nz) / min(nz):.1f}); consider --class-balanced=1.')
    if dataset_kwargs.get('cache'):
        workers = max(data_loader_kwargs.get('num_workers', 0), 1)
        dist.print0(f'Image cache enabled: estimated RAM {dataset_obj.estimated_cache_bytes() * workers / 2**30:.2f} GB '
                    f'({workers} worker(s), each keeps its own cache).')
    if rank0:
        with open(os.path.join(run_dir, 'labels.json'), 'wt') as f:
            json.dump(label_map_to_json(label_map, dataset_kwargs.get('label_col', 'label'), counts), f, indent=2)
    weights = dataset_obj.sample_weights(balance_alpha) if (class_balanced and dataset_obj.has_labels) else None
    dataset_sampler = misc.ResumableInfiniteSampler(len(dataset_obj), rank=dist.get_rank(), num_replicas=dist.get_world_size(),
                                                    seed=seed, weights=weights, start_idx=cur_nimg)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Validation data (preloaded into RAM, fixed subset).
    val_images = None
    if val_dataset_kwargs is not None and val_ticks:
        val_obj = dnnlib.util.construct_class_by_name(**val_dataset_kwargs)
        n_val = min(len(val_obj), val_max_images)
        if n_val == 0:
            dist.print0('WARNING: validation split is empty; validation loss disabled.')
        else:
            sel = np.linspace(0, len(val_obj) - 1, n_val).round().astype(np.int64)
            items = [val_obj[int(i)] for i in sel]
            val_images = torch.as_tensor(np.stack([it[0] for it in items]))
            val_cond = {k: torch.as_tensor(np.array([it[1][k] for it in items])) for k in items[0][1]}
            dist.print0(f'Validation images: {n_val} (of {len(val_obj)})')
        del val_obj

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    dist.print0(f'Parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M')
    if rank0:
        with torch.no_grad():
            n = min(batch_gpu, 2)
            images = torch.zeros([n, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            sigma = torch.ones([n], device=device)
            misc.print_module_summary(net, [images, sigma, net.null_cond(n, device)], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.EDMLoss
    dist.print0(f'Noise distribution: ln(sigma) ~ N({loss_fn.P_mean_eff:.3f}, {loss_fn.P_std:.3f}^2)')
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    ddp = net
    if dist.get_world_size() > 1:
        ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=([device] if is_cuda else None))
    ema_list_kimg = list(ema_list_kimg) if ema_list_kimg else [ema_halflife_kimg]
    emas = [copy.deepcopy(net).eval().requires_grad_(False) for _ in ema_list_kimg]
    train_net = torch.compile(ddp) if compile else ddp # pickles always use the uncompiled modules

    # Transfer learning from a network pickle.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        with dnnlib.util.open_url(resume_pkl, verbose=rank0) as f:
            data = pickle.load(f)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        for ema in emas:
            misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data

    # Resume full training state.
    if state is not None:
        if state.get('label_map') is not None and state['label_map'] != label_map:
            raise ValueError('Label map of the resumed state differs from the current dataset label map.')
        net.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer_state'])
        saved_emas = state.get('emas', {})
        for hl, ema in zip(ema_list_kimg, emas):
            key = str(float(hl))
            if key in saved_emas:
                ema.load_state_dict(saved_emas[key])
            else:
                dist.print0(f'WARNING: no saved EMA with half-life {hl} kimg; initializing it from the network.')
                ema.load_state_dict(state['net'])
        if 'rng' in state and dist.get_world_size() == 1:
            _set_rng_state(state['rng'])
        dist.print0(f'Resumed at kimg {cur_nimg / 1000:.3f}, tick {cur_tick}.')
        del state

    # Reference grids.
    num_classes = dataset_obj.num_classes if dataset_obj.has_labels else 0
    sigma_max_hint = float(noise_config.get('sigma_max', 80.0))
    if rank0 and not os.path.isfile(os.path.join(run_dir, 'reals.png')):
        save_reals_grid(dataset_obj, os.path.join(run_dir, 'reals.png'), cols=sample_cols, max_classes=sample_max_classes)

    # TensorBoard.
    tb_writer = None
    if tensorboard and rank0:
        try:
            import torch.utils.tensorboard as tensorboard_mod
            tb_writer = tensorboard_mod.SummaryWriter(run_dir)
        except ImportError as err:
            dist.print0(f'TensorBoard not available ({err}); logging to stats.jsonl only.')

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    collector = training_stats.Collector(regex='.*', keep_previous=False)
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    first_tick = (cur_nimg == 0) # print a status line right after the first step of a fresh run
    stopping = False
    tick_loss_sum, tick_loss_num = 0.0, 0
    num_bins = len(SIGMA_BIN_EDGES) - 1
    bin_sum = torch.zeros([num_bins], dtype=torch.float64, device=device)
    bin_cnt = torch.zeros([num_bins], dtype=torch.float64, device=device)
    while True:

        # Accumulate gradients.
        net.train()
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                images, cond = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                cond = {k: v.to(device) for k, v in cond.items()}
                loss, sigma = loss_fn(net=train_net, images=images, labels=cond, augment_pipe=augment_pipe, return_sigma=True)
                per_sample = loss.mean(dim=(1, 2, 3)).detach()
                training_stats.report('Loss/loss', per_sample)
                tick_loss_sum = tick_loss_sum + per_sample.sum() # stays on device (no sync)
                tick_loss_num += per_sample.numel()
                bins = sigma_bin_index(sigma) # accumulated on device, reported once per tick (no host sync)
                bin_sum.index_add_(0, bins, per_sample.to(torch.float64))
                bin_cnt.index_add_(0, bins, torch.ones_like(per_sample, dtype=torch.float64))
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Update weights.
        lr = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        if lr_decay == 'invsqrt' and lr_ref_kimg > 0:
            lr = lr / np.sqrt(max(cur_nimg / (lr_ref_kimg * 1000), 1))
        for g in optimizer.param_groups:
            g['lr'] = lr
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        if loss_scaling != 1:
            for param in net.parameters():
                if param.grad is not None:
                    param.grad.div_(loss_scaling)
        if grad_clip and grad_clip > 0:
            gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            training_stats.report('Loss/grad_norm', gnorm)
        optimizer.step()

        # Update EMAs.
        cur_nimg += batch_size
        for hl_kimg, ema in zip(ema_list_kimg, emas):
            ema_halflife_nimg = hl_kimg * 1000
            if ema_rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
            with torch.no_grad():
                for p_ema, p_net in zip(ema.parameters(), net.parameters()):
                    p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and dist.should_stop():
            done = stopping = True
        if (not done) and (not first_tick) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue
        first_tick = False

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / max(cur_nimg - tick_start_nimg, 1) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"loss {float(tick_loss_sum) / max(tick_loss_num, 1):<7.4f}"]
        for k, (bs, bc) in enumerate(zip(bin_sum.tolist(), bin_cnt.tolist())):
            if bc > 0:
                training_stats.report0(f'Loss/sigma_bin_{k}', bs / bc)
        bin_sum.zero_()
        bin_cnt.zero_()
        fields += [f"lr {training_stats.report0('Progress/lr', lr):.2e}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        if is_cuda:
            fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
            fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
            torch.cuda.reset_peak_memory_stats()

        # Validation loss (EMA network, fixed sigmas and noise).
        if val_images is not None and (done or cur_tick % val_ticks == 0):
            t0 = time.time()
            val_mean, val_per_sigma = compute_val_loss(emas[0], val_images, val_cond, loss_fn.sigma_data, device)
            training_stats.report0('Val/loss', val_mean)
            for i, v in enumerate(val_per_sigma):
                training_stats.report0(f'Val/loss_sigma_{i:02d}', v)
            fields += [f"val {val_mean:.4f} ({time.time() - t0:.1f}s)"]
        dist.print0(' '.join(fields))

        if stopping:
            dist.print0()
            dist.print0('Stop requested: saving snapshot and training state...')

        # Sample grids (rank 0, primary EMA, fixed seeds).
        if rank0 and sample_ticks and (done or cur_tick % sample_ticks == 0) and not stopping:
            save_sample_grid(emas[0], os.path.join(run_dir, f'samples-{cur_nimg//1000:06d}.png'), device, num_classes,
                             guidances=sample_guidance, cols=sample_cols, max_classes=sample_max_classes,
                             num_steps=sample_steps, sigma_max=sigma_max_hint, batch_size=max(batch_gpu, 8))

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(
                ema=emas[0],
                emas={str(float(hl)): e for hl, e in zip(ema_list_kimg, emas)},
                loss_fn=loss_fn, augment_pipe=augment_pipe,
                dataset_kwargs=dict(dataset_kwargs, label_map=None),
                label_map=label_map_to_json(label_map, dataset_kwargs.get('label_col', 'label'), counts),
                cond_config=copy.deepcopy(net.cond_config),
                noise_config=dict(noise_config),
                training_options=training_options,
                kimg=cur_nimg / 1000,
            )
            data['emas'] = {k: copy.deepcopy(v).eval().requires_grad_(False).cpu() for k, v in data['emas'].items()}
            for k, v in data['emas'].items():
                misc.check_ddp_consistency(v)
            data['ema'] = data['emas'][str(float(ema_list_kimg[0]))]
            if rank0:
                path = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
                with open(path + '.tmp', 'wb') as f:
                    pickle.dump(data, f)
                os.replace(path + '.tmp', path)
            del data # conserve memory

        # Save full dump of the training state.
        if (state_dump_ticks is not None) and (done or (cur_tick % state_dump_ticks == 0 and cur_tick != 0)) and rank0:
            path = os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt')
            torch.save(dict(
                net=net.state_dict(),
                optimizer_state=optimizer.state_dict(),
                emas={str(float(hl)): e.state_dict() for hl, e in zip(ema_list_kimg, emas)},
                cur_nimg=cur_nimg, cur_tick=cur_tick + 1, seed=seed,
                rng=_rng_state(), label_map=label_map, cond_config=net.cond_config,
            ), path + '.tmp')
            os.replace(path + '.tmp', path)

        if rank0:
            cleanup_snapshots(run_dir, keep_snapshots, keep_every_kimg)

        # Update logs.
        collector.update()
        if rank0:
            stats = {k: v for k, v in collector.as_dict().items() if v.num > 0}
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(stats, timestamp=time.time())) + '\n')
            stats_jsonl.flush()
            if tb_writer is not None:
                for k, v in stats.items():
                    tb_writer.add_scalar(k, v.mean, global_step=int(cur_nimg / 1e3))
                tb_writer.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        tick_loss_sum, tick_loss_num = 0.0, 0
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if stats_jsonl is not None:
        stats_jsonl.close()
    if tb_writer is not None:
        tb_writer.close()
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
