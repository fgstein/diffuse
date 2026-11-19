"""Dataset tests: CSV parsing, label mapping, splits, resizing, prepared datasets."""

import os
import sys
import json
import shutil
import subprocess
import numpy as np
import pandas as pd
import PIL.Image
import pytest
import torch
from training.dataset import CSVImageDataset, label_map_to_json, load_label_map_json, stratified_val_split
from torch_utils.misc import ResumableInfiniteSampler

def _ds(synth, **kw):
    args = dict(csv=synth['csv'], image_root=synth['root'], path_col=synth['path_col'], label_col=synth['label_col'])
    args.update(kw)
    return CSVImageDataset(**args)

def test_custom_columns_and_label_map(synth):
    ds = _ds(synth, split='all', resolution=64)
    assert len(ds) == 18
    assert ds.label_map == {'disk': 0, 'square': 1, 'stripes': 2} # sorted, deterministic
    assert ds.idx_to_label == ['disk', 'square', 'stripes']
    df = pd.read_csv(synth['csv'])
    for i in range(len(ds)):
        assert ds.labels[i] == ds.label_map[df[synth['label_col']][i]]

def test_resize_224_to_64(synth):
    ds = _ds(synth, split='all', resolution=64)
    img, cond = ds[0]
    assert img.dtype == np.uint8 and img.shape == (3, 64, 64)
    assert set(cond) == {'class'} and cond['class'].dtype == np.int64
    # Native resolution (no resize).
    img224, _ = _ds(synth, split='all')[0]
    assert img224.shape == (3, 224, 224)

def test_default_collate_gives_dict(synth):
    ds = _ds(synth, split='all', resolution=32)
    images, cond = next(iter(torch.utils.data.DataLoader(ds, batch_size=4)))
    assert images.shape == (4, 3, 32, 32) and images.dtype == torch.uint8
    assert cond['class'].dtype == torch.int64 and cond['class'].shape == (4,)

def test_labels_json_roundtrip(synth, tmp_path):
    ds = _ds(synth, split='all')
    p = tmp_path / 'labels.json'
    p.write_text(json.dumps(label_map_to_json(ds.label_map, 'category', ds.class_counts())))
    lm = load_label_map_json(str(p))
    assert lm == ds.label_map
    ds2 = _ds(synth, split='all', label_map=lm)
    assert np.array_equal(ds2.labels, ds.labels)

def test_unknown_label_on_resume_fails(synth):
    with pytest.raises(ValueError, match='not in the saved label map'):
        _ds(synth, split='all', label_map={'disk': 0, 'square': 1})

def test_missing_file_clear_error(synth, tmp_path):
    df = pd.read_csv(synth['csv'])
    df.loc[len(df)] = {synth['path_col']: 'does/not/exist.png', synth['label_col']: 'disk'}
    bad_csv = tmp_path / 'bad.csv'
    df.to_csv(bad_csv, index=False)
    with pytest.raises(IOError, match='not found'):
        CSVImageDataset(csv=str(bad_csv), image_root=synth['root'], path_col=synth['path_col'], label_col=synth['label_col'])

def test_missing_column_clear_error(synth):
    with pytest.raises(IOError, match='missing column'):
        CSVImageDataset(csv=synth['csv'], image_root=synth['root'], path_col='path', label_col='label')

def test_stratified_split_deterministic(synth):
    tr = _ds(synth, split='train', val_fraction=0.2)
    va = _ds(synth, split='val', val_fraction=0.2)
    assert len(tr) + len(va) == 18
    assert sorted(set(va.labels.tolist())) == [0, 1, 2] # stratified: every class in val
    assert set(tr.paths).isdisjoint(va.paths)
    va2 = _ds(synth, split='val', val_fraction=0.2)
    assert va.paths == va2.paths
    # The label map is built from ALL rows, so train/val agree.
    assert tr.label_map == va.label_map

def test_split_column(synth, tmp_path):
    df = pd.read_csv(synth['csv'])
    df['split'] = ['val' if i % 6 == 0 else 'train' for i in range(len(df))]
    p = tmp_path / 'split.csv'
    df.to_csv(p, index=False)
    va = CSVImageDataset(csv=str(p), image_root=synth['root'], path_col=synth['path_col'], label_col=synth['label_col'], split_col='split', split='val')
    assert len(va) == 3

def test_grayscale_and_rgba_converted(tmp_path):
    root = tmp_path / 'imgs'
    root.mkdir()
    PIL.Image.fromarray(np.full((224, 224), 77, np.uint8), 'L').save(root / 'g.png')
    PIL.Image.fromarray(np.full((224, 224, 4), 200, np.uint8), 'RGBA').save(root / 'a.png')
    pd.DataFrame({'path': ['g.png', 'a.png'], 'label': ['x', 'y']}).to_csv(tmp_path / 'd.csv', index=False)
    ds = CSVImageDataset(csv=str(tmp_path / 'd.csv'), image_root=str(root), split='all', resolution=32)
    for i in range(2):
        img, _ = ds[i]
        assert img.shape == (3, 32, 32)
    assert ds[0][0].min() == 77
    ds1 = CSVImageDataset(csv=str(tmp_path / 'd.csv'), image_root=str(root), split='all', resolution=32, channels=1)
    assert ds1[1][0].shape == (1, 32, 32)

@pytest.mark.parametrize('fmt', ['zip', 'dir'])
def test_prepared_equals_on_the_fly(synth, tmp_path, repo, fmt):
    out = str(tmp_path / ('prep.zip' if fmt == 'zip' else 'prep'))
    subprocess.run([sys.executable, os.path.join(repo, 'tools', 'prepare_dataset.py'), '--csv', synth['csv'],
                    '--image-root', synth['root'], '--path-col', synth['path_col'], '--label-col', synth['label_col'],
                    '--resolution', '64', '--out', out, '--workers', '2'], check=True, capture_output=True)
    raw = _ds(synth, split='all', resolution=64)
    prep = CSVImageDataset(data=out, split='all')
    assert prep.resolution == 64 and len(prep) == len(raw)
    assert prep.label_map == raw.label_map
    for i in range(len(raw)):
        a, ca = raw[i]
        b, cb = prep[i]
        assert np.array_equal(a, b), i
        assert ca['class'] == cb['class']
    # Same val split from raw and prepared data.
    assert [os.path.splitext(p)[0] for p in _ds(synth, split='val').paths] == \
           [os.path.splitext(p)[0] for p in CSVImageDataset(data=out, split='val').paths]

def test_prepare_reports_bad_rows(synth, tmp_path, repo):
    root = tmp_path / 'root'
    shutil.copytree(synth['root'], root)
    df = pd.read_csv(synth['csv'])
    PIL.Image.new('RGB', (100, 100)).save(root / 'small.png')
    (root / 'broken.png').write_bytes(b'not an image')
    extra = pd.DataFrame({synth['path_col']: ['missing.png', 'small.png', 'broken.png'], synth['label_col']: ['disk'] * 3})
    pd.concat([df, extra]).to_csv(root / 'bad.csv', index=False)
    out = tmp_path / 'p.zip'
    subprocess.run([sys.executable, os.path.join(repo, 'tools', 'prepare_dataset.py'), '--csv', str(root / 'bad.csv'),
                    '--image-root', str(root), '--path-col', synth['path_col'], '--label-col', synth['label_col'],
                    '--resolution', '32', '--out', str(out)], check=True, capture_output=True)
    bad = pd.read_csv(tmp_path / 'p-bad_rows.csv')
    assert len(bad) == 3 and 'error' in bad.columns
    assert len(CSVImageDataset(data=str(out), split='all')) == 18

def test_resumable_sampler():
    full = ResumableInfiniteSampler(10, seed=3)
    it = iter(full)
    stream = [next(it) for _ in range(35)]
    for start in [0, 7, 10, 23]:
        it2 = iter(ResumableInfiniteSampler(10, seed=3, start_idx=start))
        assert [next(it2) for _ in range(35 - start)] == stream[start:]
    # Two ranks interleave the global stream.
    r0 = iter(ResumableInfiniteSampler(10, rank=0, num_replicas=2, seed=3))
    r1 = iter(ResumableInfiniteSampler(10, rank=1, num_replicas=2, seed=3))
    assert [next(r0) for _ in range(5)] == stream[0:10:2]
    assert [next(r1) for _ in range(5)] == stream[1:10:2]
    # Weighted sampling favours heavy items.
    w = np.ones(10); w[0] = 100
    it3 = iter(ResumableInfiniteSampler(10, seed=0, weights=w))
    draws = [next(it3) for _ in range(1000)]
    assert draws.count(0) > 500
