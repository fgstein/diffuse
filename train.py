# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Modified: class-conditional training from a CSV (or a prepared dataset),
# YAML presets, configurable topology, bf16, CPU support, CFG label dropout.

"""Train a class-conditional pixel-space diffusion model (EDM formulation)."""

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop
from training.loss import compute_P_mean_eff, compute_sigma_max_hint

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch.

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if s is None or isinstance(s, list): return s
    if isinstance(s, int): return [s]
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in str(s).split(','):
        p = p.strip()
        if not p:
            continue
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

def parse_float_list(s):
    if s is None: return None
    if isinstance(s, (list, tuple)): return [float(x) for x in s]
    if isinstance(s, (int, float)): return [float(s)]
    return [float(x) for x in str(s).split(',') if x.strip()]

class IntList(click.ParamType):
    name = 'LIST'
    def convert(self, value, param, ctx):
        try:
            return parse_int_list(value)
        except ValueError:
            self.fail(f'{value!r} is not a comma-separated list of integers', param, ctx)

class FloatList(click.ParamType):
    name = 'LIST'
    def convert(self, value, param, ctx):
        try:
            return parse_float_list(value)
        except ValueError:
            self.fail(f'{value!r} is not a comma-separated list of numbers', param, ctx)

#----------------------------------------------------------------------------
# YAML presets: keys fill defaults, explicit CLI flags override them.

CONFIG_KEY_ALIASES = {
    'lr_rampup_kimg': 'lr_rampup',
    'p_mean': 'p_mean', 'p_std': 'p_std',
    'attn_resolutions': 'attn_res',
    'cbase': 'model_channels', 'cres': 'channel_mult',
    'sample_every_ticks': 'sample_every', 'val_every_ticks': 'val_every',
}

def load_config_callback(ctx, _param, value):
    if not value:
        return value
    import yaml
    with open(value, 'rt') as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise click.BadParameter(f'{value}: expected a mapping at top level')
    known = {p.name for p in ctx.command.params}
    defaults = {}
    for k, v in cfg.items():
        key = str(k).replace('-', '_').lower()
        key = CONFIG_KEY_ALIASES.get(key, key)
        if key not in known or key == 'config':
            raise click.BadParameter(f'{value}: unknown option "{k}"')
        if isinstance(v, (list, tuple)):
            v = ','.join(str(x) for x in v)
        defaults[key] = v
    ctx.default_map = dict(ctx.default_map or {}, **defaults)
    return value

#----------------------------------------------------------------------------

ARCH_PRESETS = {
    'ddpmpp': dict(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard',
                   channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2], num_blocks=4, attn_resolutions=[16]),
    'ncsnpp': dict(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard',
                   channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2], num_blocks=4, attn_resolutions=[16]),
    'adm':    dict(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4], num_blocks=3, attn_resolutions=[32,16,8]),
}

#----------------------------------------------------------------------------

@click.command(context_settings=dict(max_content_width=120))

# Config preset.
@click.option('--config',        help='YAML preset; explicit flags override it', metavar='YAML', type=click.Path(exists=True, dir_okay=False), callback=load_config_callback, is_eager=True, expose_value=True)

# Data.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Prepared dataset (zip or dir from tools/prepare_dataset.py)', metavar='ZIP|DIR', type=str)
@click.option('--csv',           help='CSV with relative image paths and labels', metavar='PATH',   type=str)
@click.option('--image-root',    help='Root directory of the relative paths in --csv', metavar='DIR', type=str)
@click.option('--path-col',      help='CSV column with relative paths', metavar='STR',              type=str, default='path', show_default=True)
@click.option('--label-col',     help='CSV column with string labels', metavar='STR',               type=str, default='label', show_default=True)
@click.option('--split-col',     help='Optional CSV column with train/val values', metavar='STR',   type=str)
@click.option('--val-fraction',  help='Validation fraction if no --split-col', metavar='FLOAT',     type=click.FloatRange(min=0, max=0.5), default=0.02, show_default=True)
@click.option('--resolution',    help='Training resolution [default: native]', metavar='INT',       type=click.IntRange(min=8))
@click.option('--channels',      help='Image channels (3 = RGB, 1 = grayscale)', metavar='INT',     type=click.Choice(['1', '3']), default='3', show_default=True)
@click.option('--uncond',        help='Train unconditional (debugging)', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--label-dropout', help='Label dropout probability for CFG', metavar='FLOAT',         type=click.FloatRange(min=0, max=1), default=0.1, show_default=True)
@click.option('--class-balanced', help='Class-balanced sampling', metavar='BOOL',                   type=bool, default=False, show_default=True)
@click.option('--balance-alpha', help='Sampling weight = 1/count^alpha', metavar='FLOAT',           type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips (may not be label-preserving!)', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--augment',       help='Augment probability (EDM geometric aug.)', metavar='FLOAT',  type=click.FloatRange(min=0, max=1), default=0.0, show_default=True)

# Architecture.
@click.option('--arch',          help='Architecture preset', metavar='ddpmpp|ncsnpp|adm',           type=click.Choice(['ddpmpp', 'ncsnpp', 'adm']), default='ddpmpp', show_default=True)
@click.option('--model-channels', help='Base channel count [default: preset]', metavar='INT',       type=click.IntRange(min=8))
@click.option('--channel-mult',  help='Channel multipliers per resolution [default: preset]', metavar='LIST', type=IntList())
@click.option('--num-blocks',    help='Residual blocks per resolution [default: preset]', metavar='INT', type=click.IntRange(min=1))
@click.option('--attn-res',      help='Feature-map resolutions with self-attention [default: preset]', metavar='LIST', type=IntList())
@click.option('--num-heads',     help='Attention heads [default: preset]', metavar='INT',           type=click.IntRange(min=1))
@click.option('--head-channels', help='Channels per attention head (overrides --num-heads)', metavar='INT', type=click.IntRange(min=1))
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.1, show_default=True)
@click.option('--patch-size',    help='Input patching factor (pixel_unshuffle)', metavar='INT',     type=click.IntRange(min=1), default=1, show_default=True)

# Noise distribution.
@click.option('--P-mean', 'p_mean', help='Mean of ln(sigma) during training', metavar='FLOAT',     type=float, default=-1.2, show_default=True)
@click.option('--P-std', 'p_std',   help='Std of ln(sigma) during training', metavar='FLOAT',      type=click.FloatRange(min=0, min_open=True), default=1.2, show_default=True)
@click.option('--noise-shift-base', help='Noise shift base resolution B (0 = off): P_mean += ln(res/B)', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--sigma-max-cap', help='Cap for the stored sampler sigma_max hint', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True), default=1000.0, show_default=True)

# Optimization.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU step (grad. accumulation)', metavar='INT', type=click.IntRange(min=1))
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=2e-4, show_default=True)
@click.option('--lr-rampup',     help='Learning rate ramp-up', metavar='KIMG',                      type=click.FloatRange(min=0), default=2000, show_default=True)
@click.option('--lr-decay',      help='Learning rate decay', metavar='none|invsqrt',                type=click.Choice(['none', 'invsqrt']), default='none', show_default=True)
@click.option('--lr-ref-kimg',   help='Reference duration for invsqrt decay', metavar='KIMG',       type=click.FloatRange(min=0, min_open=True), default=10000, show_default=True)
@click.option('--grad-clip',     help='Gradient norm clipping (0 = off)', metavar='FLOAT',          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--weight-decay',  help='Weight decay (AdamW if > 0)', metavar='FLOAT',               type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--ema-list',      help='Several EMA half-lives (first = primary), overrides --ema', metavar='MIMG,...', type=FloatList())

# Performance-related.
@click.option('--precision',     help='Mixed precision of the UNet', metavar='fp32|fp16|bf16',      type=click.Choice(['fp32', 'fp16', 'bf16']), default='bf16', show_default=True)
@click.option('--tf32',          help='Allow TF32 matmuls/convs', metavar='BOOL',                   type=bool, default=True, show_default=True)
@click.option('--grad-ckpt',     help='Gradient checkpointing', metavar='BOOL',                     type=bool, default=False, show_default=True)
@click.option('--compile',       help='torch.compile the training forward pass', metavar='BOOL',    type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling (fp16 only)', metavar='FLOAT',                  type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache decoded images in CPU memory', metavar='BOOL',         type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=0), default=4, show_default=True)
@click.option('--device',        help='Device [default: cuda if available]', metavar='cuda|cpu',    type=click.Choice(['cuda', 'cpu']))

# Monitoring / I/O.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.FloatRange(min=0, min_open=True), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=20, show_default=True)
@click.option('--dump',          help='How often to dump training state', metavar='TICKS',          type=click.IntRange(min=1), default=100, show_default=True)
@click.option('--sample-every',  help='How often to save sample grids (0 = off)', metavar='TICKS',  type=click.IntRange(min=0), default=20, show_default=True)
@click.option('--sample-guidance', help='Guidance values for sample grids', metavar='LIST',         type=FloatList(), default='1,2,4', show_default=True)
@click.option('--sample-steps',  help='Sampler steps for sample grids', metavar='INT',              type=click.IntRange(min=2), default=32, show_default=True)
@click.option('--sample-cols',   help='Seeds per class in sample grids', metavar='INT',             type=click.IntRange(min=1), default=8, show_default=True)
@click.option('--sample-max-classes', help='Max class rows in grids', metavar='INT',                type=click.IntRange(min=1), default=16, show_default=True)
@click.option('--val-every',     help='How often to compute validation loss (0 = off)', metavar='TICKS', type=click.IntRange(min=0), default=20, show_default=True)
@click.option('--val-max',       help='Max validation images used', metavar='INT',                  type=click.IntRange(min=1), default=1024, show_default=True)
@click.option('--keep-snapshots', help='Keep only the last N snapshots/states (0 = all)', metavar='INT', type=click.IntRange(min=0), default=10, show_default=True)
@click.option('--keep-every',    help='Also keep snapshots at multiples of this (0 = off)', metavar='KIMG', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--tensorboard',   help='Log to TensorBoard if installed', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--seed',          help='Random seed', metavar='INT',                                 type=int, default=0, show_default=True)
@click.option('--transfer',      help='Initialize weights from network pickle', metavar='PKL',      type=str)
@click.option('--resume',        help='Resume from training-state-*.pt (continues in the same run dir)', metavar='PT', type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)

def main(**kwargs):
    """Train a class-conditional pixel-space diffusion model (EDM formulation).

    Examples:

    \b
    # 64x64 on a single GPU from a prepared zip
    python train.py --config configs/64px_3090.yaml --data /data/prepared-64.zip --outdir runs

    \b
    # Same from the raw CSV (on-the-fly resize from 224 to 64)
    python train.py --config configs/64px_3090.yaml --csv /data/train.csv \\
        --image-root /data/images --resolution 64 --outdir runs
    """
    opts = dnnlib.EasyDict(kwargs)
    device_type = opts.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    if device_type == 'cuda' and not torch.cuda.is_available():
        raise click.ClickException('--device cuda requested but CUDA is not available.')
    torch.multiprocessing.set_start_method('spawn', force=True)
    dist.init(device_type)
    device = torch.device('cuda', torch.cuda.current_device()) if device_type == 'cuda' else torch.device('cpu')

    if opts.data is None and (opts.csv is None or opts.image_root is None):
        raise click.ClickException('Give either --data (prepared dataset) or both --csv and --image-root.')
    if opts.data is not None and opts.csv is not None:
        raise click.ClickException('--data and --csv are mutually exclusive.')

    # Resume: read the saved label map first (never re-index silently).
    label_map = None
    resume_kimg = None
    if opts.resume is not None:
        if not os.path.isfile(opts.resume):
            raise click.ClickException(f'--resume: file not found: {opts.resume}')
        match = re.fullmatch(r'training-state-(\d+)\.pt', os.path.basename(opts.resume))
        resume_kimg = int(match.group(1)) if match else None
        st = torch.load(opts.resume, map_location='cpu', weights_only=False)
        label_map = st.get('label_map')
        del st

    # Dataset.
    c = dnnlib.EasyDict()
    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.CSVImageDataset',
        csv=opts.csv, image_root=opts.image_root, data=opts.data,
        path_col=opts.path_col, label_col=opts.label_col, split_col=opts.split_col,
        split='train', val_fraction=opts.val_fraction, split_seed=opts.seed,
        resolution=opts.resolution, channels=int(opts.channels),
        label_map=label_map, use_labels=not opts.uncond, xflip=opts.xflip, cache=opts.cache)
    try:
        dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
    except (IOError, ValueError) as err:
        raise click.ClickException(f'Dataset: {err}')
    dataset_name = dataset_obj.name
    resolution = dataset_obj.resolution
    c.dataset_kwargs.resolution = resolution
    c.dataset_kwargs.label_map = dataset_obj.label_map if not opts.uncond else None
    cond_config = dataset_obj.cond_config(label_dropout=opts.label_dropout)
    num_classes = dataset_obj.num_classes
    del dataset_obj
    c.val_dataset_kwargs = dnnlib.EasyDict(c.dataset_kwargs, split='val', xflip=False, cache=False)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=(device_type == 'cuda'), num_workers=opts.workers)
    if opts.workers > 0:
        c.data_loader_kwargs.update(prefetch_factor=2, persistent_workers=True)

    # Network architecture: preset, then explicit overrides.
    net = dict(ARCH_PRESETS[opts.arch])
    if opts.model_channels is not None: net['model_channels'] = opts.model_channels
    if opts.channel_mult is not None:   net['channel_mult'] = opts.channel_mult
    if opts.num_blocks is not None:     net['num_blocks'] = opts.num_blocks
    if opts.attn_res is not None:       net['attn_resolutions'] = opts.attn_res
    if opts.num_heads is not None:      net['num_heads'] = opts.num_heads
    if opts.head_channels is not None:  net['head_channels'] = opts.head_channels
    net.update(dropout=opts.dropout, patch_size=opts.patch_size, grad_ckpt=opts.grad_ckpt, cond_config=cond_config)
    try:
        from training.networks import validate_topology
        validate_topology(resolution, opts.patch_size, net['channel_mult'], net['attn_resolutions'])
    except ValueError as err:
        raise click.ClickException(f'Invalid network topology: {err}')
    c.network_kwargs = dnnlib.EasyDict(class_name='training.networks.EDMPrecond', precision=opts.precision, **net)

    # Loss and noise schedule.
    shift_base = opts.noise_shift_base or None
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.EDMLoss', P_mean=opts.p_mean, P_std=opts.p_std,
                                    resolution=resolution, noise_shift_base=shift_base)
    c.noise_config = dict(
        P_mean=opts.p_mean, P_std=opts.p_std, P_mean_eff=compute_P_mean_eff(opts.p_mean, resolution, shift_base),
        noise_shift_base=shift_base, sigma_max=compute_sigma_max_hint(resolution, shift_base, 80.0, opts.sigma_max_cap),
        resolution=resolution, patch_size=opts.patch_size)

    # Optimizer.
    opt_class = 'torch.optim.AdamW' if opts.weight_decay > 0 else 'torch.optim.Adam'
    c.optimizer_kwargs = dnnlib.EasyDict(class_name=opt_class, lr=opts.lr, betas=[0.9, 0.999], eps=1e-8, weight_decay=opts.weight_decay)

    # Augmentation.
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        # EDM's non-leaking augmentations: the network is conditioned on the augmentation parameters
        # and sampling uses 'no augmentation', so flips/rotations here do not leak into samples.
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9

    # Training options.
    c.total_kimg = max(opts.duration * 1000, 0.001)
    c.ema_halflife_kimg = opts.ema * 1000
    c.ema_list_kimg = [x * 1000 for x in opts.ema_list] if opts.ema_list else None
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu, lr_rampup_kimg=opts.lr_rampup,
             lr_decay=opts.lr_decay, lr_ref_kimg=opts.lr_ref_kimg, grad_clip=opts.grad_clip)
    c.update(loss_scaling=opts.ls if opts.precision == 'fp16' else 1, cudnn_benchmark=opts.bench, tf32=opts.tf32, compile=opts.compile)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump,
             sample_ticks=(opts.sample_every or None), val_ticks=(opts.val_every or None),
             sample_guidance=opts.sample_guidance, sample_steps=opts.sample_steps, sample_cols=opts.sample_cols,
             sample_max_classes=opts.sample_max_classes, val_max_images=opts.val_max,
             keep_snapshots=opts.keep_snapshots, keep_every_kimg=opts.keep_every,
             class_balanced=opts.class_balanced, balance_alpha=opts.balance_alpha, tensorboard=opts.tensorboard)
    c.seed = opts.seed
    c.device = str(device)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        c.resume_state_dump = opts.resume
        c.resume_kimg = resume_kimg

    # Description string.
    cond_str = 'uncond' if opts.uncond else f'cond{num_classes}'
    desc = f'{dataset_name:s}-{resolution}px-{cond_str:s}-{opts.arch:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{opts.precision}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Pick output directory.
    if dist.get_rank() != 0:
        c.run_dir = None
    elif opts.resume is not None:
        c.run_dir = os.path.dirname(os.path.abspath(opts.resume)) # continue in the same run directory
    elif opts.nosubdir:
        c.run_dir = opts.outdir
    else:
        prev_run_dirs = []
        if os.path.isdir(opts.outdir):
            prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)

    # Options stored in snapshots (without the potentially long label map).
    c.training_options = json.loads(json.dumps({k: v for k, v in c.items() if k != 'training_options'}, default=str))
    c.training_options['dataset_kwargs'].pop('label_map', None)
    c.training_options['val_dataset_kwargs'].pop('label_map', None)
    c.training_options['cli'] = {k: v for k, v in opts.items()}

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c.training_options, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset:                 {opts.data or opts.csv} ({resolution}x{resolution}, {num_classes} classes)')
    dist.print0(f'Class-conditional:       {not opts.uncond} (label dropout {opts.label_dropout})')
    dist.print0(f'Network:                 {net["model_type"]} ch={net["model_channels"]} mult={net["channel_mult"]} blocks={net["num_blocks"]} attn={net["attn_resolutions"]} patch={opts.patch_size}')
    dist.print0(f'Noise:                   P_mean_eff={c.noise_config["P_mean_eff"]:.3f} P_std={opts.p_std} sigma_max hint={c.noise_config["sigma_max"]:.1f}')
    dist.print0(f'Device / GPUs:           {device} / {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Precision:               {opts.precision} (tf32={opts.tf32})')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c.training_options, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
