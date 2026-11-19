"""CLI tests: YAML presets with CLI overrides, topology errors."""

import os
import sys
import json
import subprocess

def _train(repo, synth, tmp_path, *extra):
    return subprocess.run([sys.executable, 'train.py', '--device', 'cpu', '--outdir', str(tmp_path / 'runs'),
                           '--csv', synth['csv'], '--image-root', synth['root'], '--path-col', synth['path_col'],
                           '--label-col', synth['label_col'], '--dry-run', *extra], cwd=repo, capture_output=True, text=True)

def test_config_and_override(repo, synth, tmp_path):
    r = _train(repo, synth, tmp_path, '--config', 'configs/64px_3090.yaml', '--lr', '3e-4')
    assert r.returncode == 0, r.stderr + r.stdout
    out = r.stdout
    start = out.index('Training options:') + len('Training options:')
    opts = json.loads(out[start:out.index('\nOutput directory')])
    assert opts['optimizer_kwargs']['lr'] == 3e-4                  # CLI override
    assert opts['network_kwargs']['channel_mult'] == [1, 2, 2, 2]  # from YAML
    assert opts['network_kwargs']['attn_resolutions'] == [16, 8]
    assert opts['dataset_kwargs']['resolution'] == 64
    assert opts['network_kwargs']['cond_config'][0]['num_classes'] == 3

def test_224_preset_noise_shift(repo, synth, tmp_path):
    r = _train(repo, synth, tmp_path, '--config', 'configs/224px_3090.yaml')
    assert r.returncode == 0, r.stderr + r.stdout
    assert 'P_mean_eff=0.053' in r.stdout

def test_invalid_topology_cli(repo, synth, tmp_path):
    r = _train(repo, synth, tmp_path, '--resolution', '64', '--channel-mult', '1,2,2', '--attn-res', '12')
    assert r.returncode != 0 and 'do not occur' in (r.stderr + r.stdout)
