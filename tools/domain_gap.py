# This file is part of a derivative of NVIDIA's EDM codebase and is
# licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (see LICENSE.txt).

"""Measure the low-level domain gap between image sets (no pretrained features).

Typical use: compare REAL images, SIMULATED images and their translations
(tools/translate.py) to choose sigma_start / ILVR settings.

Per set:
  * radially averaged power spectrum of the luminance (log10), and the
    high-frequency energy (normalised frequency > 0.25) relative to the reference;
  * grain: noise std estimate in 0-255 units (Immerkaer 1996, Laplacian-difference filter);
  * sharpness: mean gradient magnitude (Sobel) in 0-255 units;
  * with --network: the model's denoising-loss profile at 16 fixed sigmas
    (same as the training validation loss, unconditional). Images the real-data model
    finds "in distribution" have a profile close to the real images'.

Sets are given as NAME=PATH, where PATH is a directory of images (searched
recursively), a prepared dataset (.zip, or a directory with dataset.json), or
CSV@ROOT (a CSV with a 'path' column, see --path-col, and its image root).

Example:
    python tools/domain_gap.py --set real=/data/prepared-224.zip --set sim=/data/sim \\
        --set s0.1=sweep/sigma-0.1 --set s0.2=sweep/sigma-0.2 --network snap.pkl --outdir gap
"""

import os
import sys
import json
import click
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.translation import luminance, radial_spectrum, grain_std, sharpness, load_image_set # pylint: disable=wrong-import-position


#----------------------------------------------------------------------------

#----------------------------------------------------------------------------

@click.command(context_settings=dict(max_content_width=120))
@click.option('--set', 'sets', help='NAME=PATH (repeatable); PATH = image dir, prepared zip/dir, or CSV@ROOT', multiple=True, required=True)
@click.option('--reference', help='Name of the reference (real) set [default: first set]', default=None)
@click.option('--network', 'network_pkl', help='Real-data snapshot for the denoising-loss profile (optional)', default=None)
@click.option('--resolution', help='Analysis resolution [default: network resolution, else 224]', type=int, default=None)
@click.option('--channels', help='Image channels', type=click.Choice(['1', '3']), default='3', show_default=True)
@click.option('--max-images', help='Max images per set', type=click.IntRange(min=1), default=256, show_default=True)
@click.option('--path-col', help='Path column for CSV@ROOT sets', default='path', show_default=True)
@click.option('--batch', 'batch_size', help='Batch size for the denoising profile', type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--device', help='Device [default: cuda if available]', type=click.Choice(['cuda', 'cpu']), default=None)
@click.option('--outdir', help='Output directory (report.json, domain_gap.png)', required=True)
def main(sets, reference, network_pkl, resolution, channels, max_images, path_col, batch_size, device, outdir):
    """Compare spectra, grain, sharpness and the model's denoising profile across image sets."""
    named = []
    for s in sets:
        if '=' not in s:
            raise click.ClickException(f'--set must be NAME=PATH, got {s!r}')
        name, path = s.split('=', 1)
        named.append((name, path))
    reference = reference or named[0][0]
    if reference not in dict(named):
        raise click.ClickException(f'--reference {reference!r} is not one of the set names')
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    channels = int(channels)

    net = None
    if network_pkl:
        from torch_utils import distributed as dist
        from generate import load_network
        dist.init(device.type)
        net, _, _ = load_network(network_pkl, device)
        resolution = resolution or net.img_resolution
        if resolution != net.img_resolution:
            raise click.ClickException(f'--resolution must equal the network resolution ({net.img_resolution})')
    resolution = resolution or 224

    report = dict(resolution=resolution, reference=reference, sets={})
    spectra = {}
    for name, path in named:
        try:
            x = load_image_set(path, resolution, channels, max_images, path_col)
        except IOError as e:
            raise click.ClickException(str(e))
        lum = luminance(x)
        f, p = radial_spectrum(lum)
        spectra[name] = (f, p)
        entry = dict(path=path, num_images=int(x.shape[0]), grain_std=grain_std(lum), sharpness=sharpness(lum),
                     mean=float(x.float().mean()), std=float(x.float().std()))
        if net is not None:
            from training.training_loop import compute_val_loss, VAL_SIGMAS
            mean_loss, per_sigma = compute_val_loss(net, x, {}, net.sigma_data, device, batch_size=batch_size)
            entry.update(denoise_loss=mean_loss, denoise_loss_per_sigma=per_sigma)
            report['sigmas'] = VAL_SIGMAS
        report['sets'][name] = entry
        print(f'{name:>12s}: {entry["num_images"]:5d} images  grain {entry["grain_std"]:6.2f}  sharpness {entry["sharpness"]:6.2f}'
              + (f'  denoise-loss {entry["denoise_loss"]:.4f}' if net is not None else ''))

    # Gaps relative to the reference.
    f_ref, p_ref = spectra[reference]
    hf = f_ref > 0.25
    ref = report['sets'][reference]
    for name, entry in report['sets'].items():
        f, p = spectra[name]
        entry['spectrum_freq'] = f.tolist()
        entry['spectrum_log10_power'] = np.log10(p).tolist()
        entry['hf_energy_ratio_vs_ref'] = float(np.nanmean(p[hf]) / np.nanmean(p_ref[hf]))
        entry['spectrum_log_gap_vs_ref'] = float(np.nanmean(np.abs(np.log10(p) - np.log10(p_ref))))
        entry['grain_ratio_vs_ref'] = entry['grain_std'] / max(ref['grain_std'], 1e-8)
        entry['sharpness_ratio_vs_ref'] = entry['sharpness'] / max(ref['sharpness'], 1e-8)
        if net is not None:
            a = np.log(np.array(entry['denoise_loss_per_sigma']))
            b = np.log(np.array(ref['denoise_loss_per_sigma']))
            entry['denoise_profile_log_gap_vs_ref'] = float(np.mean(np.abs(a - b)))

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, 'report.json'), 'wt') as fh:
        json.dump(report, fh, indent=2)

    print(f'\nRelative to "{reference}" (1.0 = same; spectrum/profile gap: 0 = same):')
    print(f'{"set":>12s} {"HF energy":>10s} {"grain":>8s} {"sharpness":>10s} {"spectrum gap":>13s}' + (f' {"profile gap":>12s}' if net else ''))
    for name, e in report['sets'].items():
        print(f'{name:>12s} {e["hf_energy_ratio_vs_ref"]:10.3f} {e["grain_ratio_vs_ref"]:8.3f} {e["sharpness_ratio_vs_ref"]:10.3f} '
              f'{e["spectrum_log_gap_vs_ref"]:13.4f}' + (f' {e["denoise_profile_log_gap_vs_ref"]:12.4f}' if net else ''))

    # Plot.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ncols = 3 if net is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 4.5))
    for name, (f, p) in spectra.items():
        axes[0].plot(f, np.log10(p), label=name, lw=2.5 if name == reference else 1.2)
        axes[1].plot(f, np.log10(p) - np.log10(p_ref), label=name, lw=2.5 if name == reference else 1.2)
    axes[0].set_xlabel('normalised frequency'); axes[0].set_ylabel('log10 power'); axes[0].set_title('Radial power spectrum')
    axes[1].set_xlabel('normalised frequency'); axes[1].set_ylabel(f'log10 power - {reference}'); axes[1].set_title(f'Spectrum relative to {reference}')
    axes[1].axhline(0, color='k', lw=0.5)
    if net is not None:
        for name, e in report['sets'].items():
            axes[2].plot(report['sigmas'], e['denoise_loss_per_sigma'], 'o-', ms=3, label=name, lw=2.5 if name == reference else 1.2)
        axes[2].set_xscale('log'); axes[2].set_yscale('log'); axes[2].set_xlabel('sigma'); axes[2].set_ylabel('denoising loss')
        axes[2].set_title('Real-model denoising profile')
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'domain_gap.png'), dpi=110)
    print(f'Wrote {os.path.join(outdir, "report.json")} and domain_gap.png')

if __name__ == '__main__':
    main()
