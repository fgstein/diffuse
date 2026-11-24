# This file is part of a derivative of NVIDIA's EDM codebase and is
# licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (see LICENSE.txt).

"""Shared helpers for sim-to-real translation (tools/translate.py, tools/style_mosaic.py).

Seeding convention: image i, variant k always uses seed `base + i * K + k`,
whatever sigma_start / ILVR settings are used. Sweeps therefore use the same
noise for every setting (common random numbers), and differences between
settings come from the settings alone.
"""

import os
import glob
import numpy as np
import torch
from training.dataset import load_image, CSVImageDataset
from training.sampling import StackedRandomGenerator, edm_sampler, to_uint8

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')

#----------------------------------------------------------------------------

def list_inputs(csv_path=None, image_root=None, input_dir=None, path_col='path', label_col=None):
    """Return (root, [(relative path, label or '')]) from a CSV + root, or a directory."""
    if input_dir is not None:
        files = sorted(f for f in glob.glob(os.path.join(input_dir, '**', '*'), recursive=True)
                       if f.lower().endswith(IMAGE_EXTS))
        if not files:
            raise IOError(f'No images found under {input_dir}')
        return input_dir, [(os.path.relpath(f, input_dir), '') for f in files]
    import pandas as pd
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    for c in [path_col, label_col]:
        if c is not None and c not in df.columns:
            raise IOError(f'Column "{c}" not in {csv_path}; available: {list(df.columns)}')
    labels = df[label_col].tolist() if label_col else [''] * len(df)
    return image_root, list(zip(df[path_col].tolist(), labels))

def load_inputs(root, rels, resolution, channels):
    """Load images as uint8 [N, C, H, W], resized to the network resolution."""
    imgs = []
    for rel in rels:
        try:
            with open(os.path.join(root, rel), 'rb') as f:
                imgs.append(load_image(f, resolution=resolution, channels=channels, name=rel))
        except Exception as e:
            raise IOError(f'Cannot load "{rel}": {e}') from e
    return torch.as_tensor(np.stack(imgs))

def check_labels(net, label_json, labels):
    """Validate labels for conditional translation; returns label_to_idx."""
    if 'class' not in net.cond_stack.names:
        raise ValueError('This network is unconditional; translate without labels.')
    label_to_idx = dict(label_json['label_to_idx']) if label_json else {}
    unknown = sorted({l for l in labels if l and l not in label_to_idx})
    if unknown:
        raise ValueError(f'Labels not in the snapshot label map: {unknown[:10]}')
    return label_to_idx

def make_cond(net, labels, label_to_idx, device):
    """Condition dict: class index per label, NULL for '' or when label_to_idx is None (unconditional)."""
    if 'class' not in net.cond_stack.names:
        return None
    null_idx = net.cond_stack.encoders['class'].null_index
    idx = [label_to_idx[l] if (label_to_idx is not None and l) else null_idx for l in labels]
    return {'class': torch.as_tensor(idx, dtype=torch.int64, device=device)}

def load_image_set(path, resolution, channels, max_images, path_col, seed=0):
    """Load an image set as uint8 [N, C, H, W] at `resolution`.

    path: a directory of images (recursive; mosaic/compare outputs are skipped), a prepared
    dataset (.zip, or a directory with dataset.json), or CSV@ROOT. At most max_images
    (random subset, seeded)."""
    if '@' in path:
        csv_path, root = path.split('@', 1)
        ds = CSVImageDataset(csv=csv_path, image_root=root, path_col=path_col, label_col=None, split='all',
                             resolution=resolution, channels=channels, use_labels=False, max_size=max_images)
        return torch.stack([torch.as_tensor(ds[i][0]) for i in range(len(ds))])
    if path.lower().endswith('.zip') or os.path.isfile(os.path.join(path, 'dataset.json')):
        ds = CSVImageDataset(data=path, split='all', resolution=resolution, channels=channels,
                             use_labels=False, max_size=max_images)
        return torch.stack([torch.as_tensor(ds[i][0]) for i in range(len(ds))])
    files = sorted(f for f in glob.glob(os.path.join(path, '**', '*'), recursive=True)
                   if f.lower().endswith(IMAGE_EXTS) and os.path.basename(f) not in ('compare.png', 'grid.png') and not os.path.basename(f).startswith(('full_', 'zoom_', 'diff_', 'grid2d_', 'metrics')))
    if not files:
        raise IOError(f'No images under {path}')
    if max_images and len(files) > max_images:
        sel = np.sort(np.random.RandomState(seed).permutation(len(files))[:max_images])
        files = [files[i] for i in sel]
    imgs = []
    for f in files:
        with open(f, 'rb') as fh:
            imgs.append(torch.as_tensor(load_image(fh, resolution=resolution, channels=channels, name=f)))
    return torch.stack(imgs)

@torch.no_grad()
def translate(net, x_uint8, seeds, sigma_start, device, cond=None, num_steps=16, guidance=1.0,
              ilvr_factor=0, ilvr_sigma_min=0.0, batch_size=32):
    """SDEdit (+ optional ILVR) translation of uint8 images [N, C, H, W]. Returns uint8 [N, C, H, W] (CPU)."""
    out = []
    for start in range(0, x_uint8.shape[0], batch_size):
        x = x_uint8[start:start + batch_size].to(device).to(torch.float32) / 127.5 - 1
        bseeds = seeds[start:start + batch_size]
        rnd = StackedRandomGenerator(device, bseeds)
        eps = rnd.randn(list(x.shape), device=device)
        bcond = None if cond is None else {k: v[start:start + batch_size] for k, v in cond.items()}
        y = edm_sampler(net, eps, bcond, randn_like=rnd.randn_like, num_steps=num_steps, guidance=guidance,
                        x_init=x, sigma_start=sigma_start, ilvr_ref=(x if ilvr_factor and ilvr_factor > 1 else None),
                        ilvr_factor=ilvr_factor, ilvr_sigma_min=ilvr_sigma_min)
        out.append(to_uint8(y).cpu())
    return torch.cat(out)

#----------------------------------------------------------------------------
# Low-level image statistics (no pretrained features). Inputs: uint8 [N, C, H, W].

def luminance(x):
    x = torch.as_tensor(x).to(torch.float64)
    if x.shape[1] == 3:
        return 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]
    return x[:, 0]

def radial_spectrum(lum, num_bands=32):
    """Mean power per radial frequency band (normalised frequency 0..0.5), Hann-windowed."""
    N, H, W = lum.shape
    win = torch.outer(torch.hann_window(H, periodic=False, dtype=torch.float64), torch.hann_window(W, periodic=False, dtype=torch.float64))
    x = (lum - lum.mean(dim=(1, 2), keepdim=True)) * win
    P = torch.fft.fftshift(torch.fft.fft2(x).abs().square(), dim=(-2, -1)).mean(0) / (H * W)
    fy = torch.fft.fftshift(torch.fft.fftfreq(H, dtype=torch.float64))
    fx = torch.fft.fftshift(torch.fft.fftfreq(W, dtype=torch.float64))
    r = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    edges = torch.linspace(0, 0.5, num_bands + 1, dtype=torch.float64)
    centers, power = [], []
    for b in range(num_bands):
        m = (r >= edges[b]) & (r < edges[b + 1])
        if b == 0:
            m &= r > 0 # skip DC
        centers.append(float((edges[b] + edges[b + 1]) / 2))
        power.append(float(P[m].mean()) if m.any() else float('nan'))
    return np.array(centers), np.array(power)

def hf_energy(lum, cutoff=0.25):
    f, p = radial_spectrum(lum)
    return float(np.nanmean(p[f > cutoff]))

def grain_std(lum):
    """Immerkaer's fast noise std estimate (0-255 units), averaged over images."""
    M = torch.tensor([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=torch.float64)[None, None]
    y = torch.nn.functional.conv2d(lum[:, None], M)
    H, W = lum.shape[1:]
    return float((np.sqrt(np.pi / 2) / (6 * (W - 2) * (H - 2)) * y.abs().sum(dim=(1, 2, 3))).mean())

def sharpness(lum):
    """Mean Sobel gradient magnitude (0-255 units)."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float64)[None, None] / 8
    gx = torch.nn.functional.conv2d(lum[:, None], kx)
    gy = torch.nn.functional.conv2d(lum[:, None], kx.transpose(-1, -2))
    return float(torch.sqrt(gx ** 2 + gy ** 2).mean())

def image_stats(x_uint8):
    lum = luminance(x_uint8)
    return dict(grain=grain_std(lum), sharpness=sharpness(lum), hf_energy=hf_energy(lum))

#----------------------------------------------------------------------------
