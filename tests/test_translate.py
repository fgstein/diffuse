"""Image-to-image (SDEdit / ILVR) sampler and the translate / domain_gap tools."""

import os
import sys
import json
import pickle
import subprocess
import pytest
import torch
from conftest import tiny_net
from training.sampling import edm_sampler, lowpass

def _x(n=2, res=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 3, res, res, generator=g) * 2 - 1

def test_sdedit_small_sigma_keeps_image():
    net = tiny_net(32).eval()
    x0 = _x()
    with torch.no_grad():
        out = edm_sampler(net, torch.randn_like(x0), None, num_steps=4, x_init=x0, sigma_start=0.01)
    assert (out - x0).abs().mean() < 0.02

def test_sdedit_larger_sigma_changes_more():
    net = tiny_net(32).eval()
    x0 = _x()
    eps = torch.randn_like(x0)
    with torch.no_grad():
        d = [(edm_sampler(net, eps, None, num_steps=6, x_init=x0, sigma_start=s) - x0).abs().mean() for s in (0.05, 2.0)]
    assert d[0] < d[1]

def test_sdedit_requires_sigma_start_above_sigma_min():
    net = tiny_net(32).eval()
    x0 = _x()
    with pytest.raises(ValueError):
        edm_sampler(net, torch.randn_like(x0), None, x_init=x0, sigma_start=0.001)
    with pytest.raises(ValueError):
        edm_sampler(net, torch.randn_like(x0), None, x_init=x0)

def test_ilvr_keeps_low_frequencies():
    net = tiny_net(32).eval()
    ref = _x(seed=1)
    eps = torch.randn_like(ref)
    with torch.no_grad():
        free = edm_sampler(net, eps, None, num_steps=8, x_init=ref, sigma_start=5.0)
        ilvr = edm_sampler(net, eps, None, num_steps=8, x_init=ref, sigma_start=5.0, ilvr_ref=ref, ilvr_factor=4)
    lp = lambda t: lowpass(t.to(torch.float64), 4)
    assert (lp(ilvr) - lp(ref)).abs().mean() < 0.5 * (lp(free) - lp(ref)).abs().mean()

def test_lowpass_factor_must_divide():
    with pytest.raises(ValueError):
        lowpass(torch.zeros(1, 3, 32, 32), 5)
    x = torch.randn(1, 3, 32, 32)
    assert lowpass(x, 1) is x

def _snapshot(tmp_path):
    net = tiny_net(32).eval()
    label_map = dict(label_to_idx={'disk': 0, 'square': 1, 'stripes': 2}, idx_to_label=['disk', 'square', 'stripes'], label_col='category', counts={})
    snap = tmp_path / 'net.pkl'
    with open(snap, 'wb') as f:
        pickle.dump(dict(ema=net, emas={}, label_map=label_map, cond_config=net.cond_config, noise_config=dict(sigma_max=80.0)), f)
    return snap

def test_translate_tool(tmp_path, repo, synth):
    snap = _snapshot(tmp_path)
    base = [sys.executable, 'tools/translate.py', '--device', 'cpu', '--network', str(snap), '--csv', synth['csv'],
            '--image-root', synth['root'], '--path-col', synth['path_col'], '--max-images', '3', '--steps', '3']
    # Unconditional sweep with 2 variants and ILVR.
    out = tmp_path / 'tr'
    r = subprocess.run(base + ['--outdir', str(out), '--sigma-start', '0.1,0.5', '--samples-per-image', '2', '--ilvr-factor', '4'],
                       cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    import pandas as pd
    meta = pd.read_csv(out / 'metadata.csv', keep_default_na=False)
    assert len(meta) == 3 * 2 * 2 and set(meta['sigma_start']) == {0.1, 0.5}
    assert all((out / p).is_file() for p in meta['output'])
    assert meta['output'][0].startswith('sigma-0.1/') and meta['output'][0].endswith('_k0.png')
    assert (out / 'compare.png').is_file()
    # Conditional on the CSV labels.
    out2 = tmp_path / 'tr2'
    r = subprocess.run(base + ['--outdir', str(out2), '--label-col', synth['label_col'], '--guidance', '2'],
                       cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    meta2 = pd.read_csv(out2 / 'metadata.csv', keep_default_na=False)
    assert set(meta2['label']) <= {'disk', 'square', 'stripes'} and len(meta2) == 3

def test_domain_gap_tool(tmp_path, repo, synth):
    snap = _snapshot(tmp_path)
    r = subprocess.run([sys.executable, 'tools/domain_gap.py', '--device', 'cpu', '--network', str(snap),
                        '--set', f'real={synth["csv"]}@{synth["root"]}', '--set', f'dir={synth["root"]}/images/disk',
                        '--path-col', synth['path_col'], '--max-images', '6', '--outdir', str(tmp_path / 'gap')],
                       cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    rep = json.load(open(tmp_path / 'gap' / 'report.json'))
    real = rep['sets']['real']
    assert real['hf_energy_ratio_vs_ref'] == pytest.approx(1.0) and real['denoise_profile_log_gap_vs_ref'] == 0
    assert len(real['denoise_loss_per_sigma']) == 16 and rep['resolution'] == 32
    assert (tmp_path / 'gap' / 'domain_gap.png').is_file()
