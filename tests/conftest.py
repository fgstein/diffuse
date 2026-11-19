"""Shared pytest fixtures (CPU only)."""

import os
import sys
import subprocess
import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
torch.set_num_threads(min(4, torch.get_num_threads()))

@pytest.fixture(scope='session')
def repo():
    return REPO

@pytest.fixture(scope='session')
def synth(tmp_path_factory):
    """Tiny synthetic dataset with CUSTOM column names: 3 classes x 6 images, 224x224."""
    out = tmp_path_factory.mktemp('synth')
    subprocess.run([sys.executable, os.path.join(REPO, 'tools', 'make_synthetic_dataset.py'), '--out', str(out),
                    '--n-per-class', '6', '--classes', '3', '--path-col', 'relpath', '--label-col', 'category'],
                   check=True, capture_output=True)
    return dict(root=str(out), csv=str(out / 'data.csv'), path_col='relpath', label_col='category')

def tiny_net(resolution=32, num_classes=3, patch_size=1, channel_mult=(1, 2), attn=(16,), dropout=0.1, label_dropout=0.1, **kw):
    from training.networks import EDMPrecond
    from training.conditioning import make_class_cond_config
    cond = make_class_cond_config(num_classes, label_dropout) if num_classes else []
    torch.manual_seed(0)
    perturb = kw.pop('perturb', True)
    net = EDMPrecond(img_resolution=resolution, img_channels=3, model_type=kw.pop('model_type', 'SongUNet'),
                      model_channels=kw.pop('model_channels', 16), channel_mult=list(channel_mult), num_blocks=1,
                      attn_resolutions=list(attn), dropout=dropout, patch_size=patch_size, cond_config=cond, **kw)
    if perturb:
        # Freshly initialized EDM UNets have (near-)zero output layers, which makes the
        # output almost independent of the condition. Perturb weights so tests are meaningful.
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn_like(p) * 0.05)
    return net
