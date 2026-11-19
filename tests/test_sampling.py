"""Sampler / CFG / snapshot tests."""

import os
import sys
import json
import pickle
import subprocess
import numpy as np
import torch
from conftest import tiny_net
from training.sampling import edm_sampler, guided_denoise, StackedRandomGenerator

def _reference_sampler(net, latents, cond, num_steps=6, sigma_min=0.002, sigma_max=80, rho=7):
    """Plain EDM Heun sampler without any CFG code path."""
    step_indices = torch.arange(num_steps, dtype=torch.float64)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_hat, t_hat = x_next, t_cur
        d_cur = (x_hat - net(x_hat, t_hat, cond).to(torch.float64)) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if i < num_steps - 1:
            d_prime = (x_next - net(x_next, t_next, cond).to(torch.float64)) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
    return x_next

def test_cfg_guidance_one_and_zero():
    net = tiny_net(32).eval()
    torch.manual_seed(1)
    latents = torch.randn(3, 3, 32, 32)
    cond = {'class': torch.tensor([0, 1, 2])}
    with torch.no_grad():
        ref_c = _reference_sampler(net, latents, cond)
        ref_u = _reference_sampler(net, latents, net.null_cond(3, latents.device))
        g1 = edm_sampler(net, latents, cond, num_steps=6, guidance=1.0)
        g0 = edm_sampler(net, latents, cond, num_steps=6, guidance=0.0)
        g2 = edm_sampler(net, latents, cond, num_steps=6, guidance=2.0)
    assert (g1 - ref_c).abs().max() <= 1e-6
    assert (g0 - ref_u).abs().max() <= 1e-6
    assert not torch.allclose(g2, g1)

def test_cfg_formula_and_interval():
    net = tiny_net(32).eval()
    x = torch.randn(2, 3, 32, 32)
    s = torch.tensor(1.5, dtype=torch.float64)
    cond = {'class': torch.tensor([0, 2])}
    with torch.no_grad():
        dc = net(x, s, cond)
        du = net(x, s, net.null_cond(2, x.device))
        dg = guided_denoise(net, x, s, cond, guidance=3.0)
        outside = guided_denoise(net, x, s, cond, guidance=3.0, guidance_interval=(2.0, 10.0))
    assert torch.allclose(dg, du + 3.0 * (dc - du), atol=1e-5)
    assert torch.equal(outside, dc)

def test_sampler_deterministic():
    net = tiny_net(32).eval()
    cond = {'class': torch.tensor([1, 1])}
    outs = []
    for _ in range(2):
        rnd = StackedRandomGenerator('cpu', [5, 6])
        lat = rnd.randn([2, 3, 32, 32])
        with torch.no_grad():
            outs.append(edm_sampler(net, lat, cond, randn_like=rnd.randn_like, num_steps=4, guidance=2.0, S_churn=5))
    assert torch.equal(outs[0], outs[1])

def test_snapshot_generate_fresh_process(tmp_path, repo):
    net = tiny_net(32).eval()
    label_map = dict(label_to_idx={'alpha': 0, 'beta/x': 1, 'gamma': 2}, idx_to_label=['alpha', 'beta/x', 'gamma'], label_col='label', counts={})
    snap = tmp_path / 'network-snapshot-000000.pkl'
    with open(snap, 'wb') as f:
        pickle.dump(dict(ema=net, emas={'500.0': net}, label_map=label_map, cond_config=net.cond_config,
                         noise_config=dict(sigma_max=80.0, resolution=32, patch_size=1)), f)
    out = tmp_path / 'gen'
    r = subprocess.run([sys.executable, 'generate.py', '--device', 'cpu', '--network', str(snap), '--outdir', str(out),
                        '--per-class', '2', '--guidance', '2', '--steps', '3', '--grid', '1', '--batch', '4'],
                       cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import pandas as pd
    meta = pd.read_csv(out / 'metadata.csv', keep_default_na=False)
    assert len(meta) == 6 and set(meta['label']) == {'alpha', 'beta/x', 'gamma'}
    assert (out / 'alpha' / '000000.png').is_file() and (out / 'beta_x' / '000001.png').is_file()
    assert (out / 'grid.png').is_file()
    # By class name, and determinism across processes.
    out2 = tmp_path / 'gen2'
    r = subprocess.run([sys.executable, 'generate.py', '--device', 'cpu', '--network', str(snap), '--outdir', str(out2),
                        '--class', 'alpha', '--seeds', '0-1', '--guidance', '2', '--steps', '3'],
                       cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import PIL.Image
    a = np.asarray(PIL.Image.open(out / 'alpha' / '000001.png')).astype(int)
    b = np.asarray(PIL.Image.open(out2 / 'alpha' / '000001.png')).astype(int)
    # Different batch composition (4 vs 2 images) changes float round-off by ~1e-6,
    # which can flip a uint8 rounding by at most 1. Same batching is bitwise identical.
    assert np.abs(a - b).max() <= 1
    out3 = tmp_path / 'gen3'
    r = subprocess.run([sys.executable, 'generate.py', '--device', 'cpu', '--network', str(snap), '--outdir', str(out3),
                        '--class', 'alpha', '--seeds', '0-1', '--guidance', '2', '--steps', '3'],
                       cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    c = np.asarray(PIL.Image.open(out3 / 'alpha' / '000001.png')).astype(int)
    assert np.array_equal(b, c)
