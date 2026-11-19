# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Datasets.

`CSVImageDataset` (new) reads images listed in a CSV (relative paths + string
labels) from an image root, or a dataset prepared by tools/prepare_dataset.py
(zip or directory). It returns `(uint8 image [C,H,W], cond_dict)` where
`cond_dict = {"class": int64}`; the default collate function turns that into
`{"class": LongTensor[N]}`.

The original EDM `Dataset` / `ImageFolderDataset` classes are kept below for
reference/compatibility but are not used by the training loop."""

import os
import numpy as np
import zipfile
import PIL.Image
import json
import io
import warnings
import torch
import dnnlib

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------
# Dataset subclass that loads images recursively from the specified directory
# or ZIP file.

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        use_pyspng      = True, # Use pyspng if available?
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._use_pyspng = use_pyspng
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if self._use_pyspng and pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Shared image loading / resizing (used by the dataset AND tools/prepare_dataset.py
# so that on-the-fly and prepared data are bit-identical).

PREPARED_FORMAT = 'edm-cond-prepared-v1'
VAL_SPLIT_VALUES = ('val', 'valid', 'validation')
TRAIN_SPLIT_VALUES = ('train',)

def pil_mode_for_channels(channels):
    if channels == 3: return 'RGB'
    if channels == 1: return 'L'
    raise ValueError(f'Unsupported number of channels: {channels} (expected 1 or 3)')

def resize_filter(src_size, dst_size):
    """Anti-aliased filter: BOX for integer downscale factors, LANCZOS otherwise."""
    if dst_size < src_size and src_size % dst_size == 0:
        return PIL.Image.BOX
    return PIL.Image.LANCZOS

def load_image(fobj, resolution=None, channels=3, name=''):
    """Decode an image file object, convert to RGB/L, check it is square, resize to
    `resolution` (None = keep). Returns uint8 numpy array [C, H, W]."""
    img = PIL.Image.open(fobj)
    img.load()
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        # Composite transparency on black (deterministic) before conversion.
        img = img.convert('RGBA')
        bg = PIL.Image.new('RGBA', img.size, (0, 0, 0, 255))
        img = PIL.Image.alpha_composite(bg, img)
    img = img.convert(pil_mode_for_channels(channels))
    w, h = img.size
    if w != h:
        raise IOError(f'Image "{name}" is not square ({w}x{h}); only square images are supported.')
    if resolution is not None and w != resolution:
        img = img.resize((resolution, resolution), resample=resize_filter(w, resolution))
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    return np.ascontiguousarray(arr.transpose(2, 0, 1))

#----------------------------------------------------------------------------
# Label map helpers.

def build_label_map(labels):
    uniq = sorted(set(str(x) for x in labels))
    return {name: i for i, name in enumerate(uniq)}

def label_map_to_json(label_map, label_col='label', counts=None):
    idx_to_label = [None] * len(label_map)
    for name, i in label_map.items():
        idx_to_label[i] = name
    return dict(label_to_idx=dict(label_map), idx_to_label=idx_to_label, label_col=label_col,
                counts=dict(counts) if counts is not None else {})

def load_label_map_json(path_or_dict):
    d = path_or_dict
    if not isinstance(d, dict):
        with open(path_or_dict, 'rt') as f:
            d = json.load(f)
    if 'label_to_idx' in d:
        return {str(k): int(v) for k, v in d['label_to_idx'].items()}
    return {str(k): int(v) for k, v in d.items()}

#----------------------------------------------------------------------------
# Record reading.

def _read_csv_records(csv_path, path_col, label_col, split_col):
    import pandas as pd
    if not os.path.isfile(csv_path):
        raise IOError(f'CSV file not found: {csv_path}')
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [c for c in [path_col, label_col, split_col] if c is not None and c not in df.columns]
    if missing:
        raise IOError(f'CSV {csv_path} is missing column(s) {missing}; available columns: {list(df.columns)}. '
                      f'Use --path-col / --label-col / --split-col to select columns.')
    paths = df[path_col].tolist()
    labels = df[label_col].tolist() if label_col is not None else [''] * len(df)
    splits = df[split_col].tolist() if split_col is not None else None
    empty = [i for i, p in enumerate(paths) if not p]
    if empty:
        raise IOError(f'CSV {csv_path}: empty path in row(s) {empty[:10]} (0-based, excluding header).')
    return paths, labels, splits

def stratified_val_split(labels, val_fraction, seed):
    """Deterministic stratified split. Returns a boolean array: True = validation."""
    labels = np.asarray(labels, dtype=object)
    is_val = np.zeros(len(labels), dtype=bool)
    if val_fraction <= 0:
        return is_val
    rnd = np.random.RandomState(seed % (1 << 31))
    for name in sorted(set(labels.tolist())):
        idx = np.nonzero(labels == name)[0]
        n = idx.size
        if n < 2:
            continue
        n_val = int(min(max(1, round(val_fraction * n)), n - 1))
        is_val[rnd.permutation(idx)[:n_val]] = True
    return is_val

#----------------------------------------------------------------------------
# CSV / prepared dataset.

class CSVImageDataset(torch.utils.data.Dataset):
    def __init__(self,
        csv             = None,     # Path to CSV with relative image paths + labels (raw mode).
        image_root      = None,     # Root directory for the relative paths (raw mode).
        data            = None,     # Prepared dataset (zip or directory with dataset.json), alternative to csv/image_root.
        path_col        = 'path',   # CSV column with relative image paths.
        label_col       = 'label',  # CSV column with string labels.
        split_col       = None,     # Optional CSV column with 'train'/'val' values.
        split           = 'train',  # Which split to expose: 'train' | 'val' | 'all'.
        val_fraction    = 0.02,     # Validation fraction if split_col is None (stratified, seeded).
        split_seed      = 0,        # Seed for the stratified split.
        resolution      = None,     # Output resolution, None = native resolution of the files.
        channels        = 3,        # 3 = RGB, 1 = grayscale.
        label_map       = None,     # dict label -> index (e.g. from a previous run), None = build from data.
        use_labels      = True,     # False = unconditional (cond dict is empty).
        xflip           = False,    # Double the dataset with horizontal flips (NOT label-safe in general).
        cache           = False,    # Cache decoded uint8 images in RAM (per DataLoader worker).
        max_size        = None,     # Limit dataset size (applied after the split), None = no limit.
        check_files     = True,     # Verify that every listed file exists at construction.
        name            = None,     # Dataset name (for run directory names).
    ):
        self._channels = int(channels)
        self._cache = cache
        self._cached_images = dict()
        self._zipfile = None
        self._use_labels = use_labels
        self.path_col, self.label_col, self.split_col = path_col, label_col, split_col
        meta = None

        # Read records.
        if data is not None:
            self._root = data
            if os.path.isdir(data):
                self._type = 'dir'
                if os.path.isfile(os.path.join(data, 'dataset.json')):
                    with open(os.path.join(data, 'dataset.json'), 'rt') as f:
                        meta = json.load(f)
                elif os.path.isfile(os.path.join(data, 'data.csv')):
                    csv = os.path.join(data, 'data.csv')
                else:
                    raise IOError(f'--data directory {data} contains neither dataset.json nor data.csv')
            elif os.path.splitext(data)[1].lower() == '.zip':
                if not os.path.isfile(data):
                    raise IOError(f'Prepared dataset not found: {data}')
                self._type = 'zip'
                zf = self._get_zipfile()
                if 'dataset.json' not in zf.namelist():
                    raise IOError(f'{data}: dataset.json not found in zip (create it with tools/prepare_dataset.py)')
                meta = json.loads(zf.read('dataset.json'))
            else:
                raise IOError(f'--data must be a .zip or a directory, got {data}')
            if meta is not None:
                if meta.get('format') != PREPARED_FORMAT:
                    raise IOError(f'{data}: unsupported dataset.json format (expected "{PREPARED_FORMAT}")')
                rows = meta['rows']
                paths = [r['path'] for r in rows]
                labels = [r.get('label', '') for r in rows]
                splits = [r['split'] for r in rows] if (rows and 'split' in rows[0]) else None
                self.path_col = meta.get('path_col', path_col)
                self.label_col = meta.get('label_col', label_col)
                self.split_col = meta.get('split_col', None) if splits is not None else None
            else:
                paths, labels, splits = _read_csv_records(csv, path_col, label_col, split_col)
        else:
            if csv is None or image_root is None:
                raise IOError('Either --data, or both --csv and --image-root must be given.')
            if not os.path.isdir(image_root):
                raise IOError(f'Image root directory not found: {image_root}')
            self._type = 'dir'
            self._root = image_root
            paths, labels, splits = _read_csv_records(csv, path_col, label_col, split_col)
        if len(paths) == 0:
            raise IOError('Dataset is empty.')
        paths = [p.replace('\\', '/') for p in paths]
        labels = [str(x) for x in labels]

        # Existence check (all rows, independent of the split).
        if check_files and len(paths) > 0:
            if self._type == 'zip':
                names = set(self._get_zipfile().namelist())
                missing = [p for p in paths if p not in names]
            else:
                missing = [p for p in paths if not os.path.isfile(os.path.join(self._root, p))]
            if missing:
                raise IOError(f'{len(missing)} image file(s) listed in the CSV were not found under "{self._root}", '
                              f'e.g. {missing[:5]}. Run tools/prepare_dataset.py to validate and drop bad rows.')

        # Label map (built from ALL rows so train/val agree).
        if label_map is None:
            label_map = build_label_map(labels)
        else:
            label_map = {str(k): int(v) for k, v in label_map.items()}
            unknown = sorted(set(labels) - set(label_map))
            if unknown:
                raise ValueError(f'The data contains {len(unknown)} label(s) that are not in the saved label map '
                    f'(e.g. {unknown[:5]}). Refusing to silently re-index; start a new run or fix the CSV.')
        self._label_map = label_map
        all_label_idx = np.array([label_map[x] for x in labels], dtype=np.int64)

        # Split.
        if splits is not None:
            sv = np.array([str(x).strip().lower() for x in splits], dtype=object)
            is_val = np.isin(sv, VAL_SPLIT_VALUES)
            is_train = np.isin(sv, TRAIN_SPLIT_VALUES)
            other = ~(is_val | is_train)
            if other.any():
                warnings.warn(f'{int(other.sum())} rows have a split value other than train/val; they are ignored.')
        else:
            is_val = stratified_val_split(labels, val_fraction, split_seed)
            is_train = ~is_val
        if split == 'train':
            sel = np.nonzero(is_train)[0]
        elif split == 'val':
            sel = np.nonzero(is_val)[0]
        elif split == 'all':
            sel = np.arange(len(paths))
        else:
            raise ValueError(f'Unknown split "{split}"')
        if max_size is not None and sel.size > max_size:
            sel = np.sort(np.random.RandomState(split_seed % (1 << 31)).permutation(sel)[:max_size])
        self._split = split
        self._paths = [paths[i] for i in sel]
        self._labels = all_label_idx[sel]
        if not self._use_labels:
            self._labels = np.zeros_like(self._labels)

        # Resolution.
        if resolution is None:
            if meta is not None and 'resolution' in meta:
                resolution = int(meta['resolution'])
            elif len(self._paths) > 0:
                resolution = self._load_raw_image_native(0).shape[1]
        self._resolution = resolution
        self._name = name or (os.path.splitext(os.path.basename(csv or data))[0] if (csv or data) else 'dataset')

        # xflip.
        self._raw_idx = np.arange(len(self._paths), dtype=np.int64)
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    # --- file access
    def _get_zipfile(self):
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._root)
        return self._zipfile

    def _open_file(self, relpath):
        if self._type == 'zip':
            return io.BytesIO(self._get_zipfile().read(relpath))
        return open(os.path.join(self._root, relpath), 'rb')

    def _load_raw_image_native(self, raw_idx):
        p = self._paths[raw_idx]
        with self._open_file(p) as f:
            return load_image(f, resolution=None, channels=self._channels, name=p)

    def _load_raw_image(self, raw_idx):
        p = self._paths[raw_idx]
        try:
            with self._open_file(p) as f:
                return load_image(f, resolution=self._resolution, channels=self._channels, name=p)
        except Exception as e:
            raise IOError(f'Failed to load image "{p}" (root "{self._root}"): {e}') from e

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(self.__dict__, _zipfile=None)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # --- torch Dataset API
    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = int(self._raw_idx[idx])
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        if self._xflip[idx]:
            image = image[:, :, ::-1]
        return image.copy(), self.get_cond(idx)

    def get_cond(self, idx):
        if not self._use_labels:
            return {}
        return {'class': np.int64(self._labels[self._raw_idx[idx]])}

    # --- properties / helpers
    @property
    def name(self): return self._name
    @property
    def resolution(self): return self._resolution
    @property
    def num_channels(self): return self._channels
    @property
    def image_shape(self): return [self._channels, self._resolution, self._resolution]
    @property
    def label_map(self): return dict(self._label_map)
    @property
    def num_classes(self): return len(self._label_map)
    @property
    def has_labels(self): return self._use_labels
    @property
    def idx_to_label(self):
        out = [None] * len(self._label_map)
        for k, v in self._label_map.items():
            out[v] = k
        return out
    @property
    def labels(self):
        """int64 label index for every item (including xflip duplicates)."""
        return self._labels[self._raw_idx]
    @property
    def paths(self): return list(self._paths)

    def class_counts(self):
        counts = np.bincount(self._labels, minlength=self.num_classes)
        return {name: int(counts[i]) for name, i in self._label_map.items()}

    def sample_weights(self, alpha=0.5):
        """Per-item weights proportional to 1 / count(class)^alpha (for class-balanced sampling)."""
        labels = self.labels
        counts = np.bincount(labels, minlength=self.num_classes).astype(np.float64)
        return 1.0 / np.maximum(counts[labels], 1) ** alpha

    def cond_config(self, label_dropout=0.1):
        from training.conditioning import make_class_cond_config
        return make_class_cond_config(self.num_classes, dropout=label_dropout) if self._use_labels else []

    def estimated_cache_bytes(self):
        return len(self._paths) * self._channels * (self._resolution or 0) ** 2

    def indices_of_class(self, class_idx):
        return np.nonzero(self._labels[self._raw_idx] == class_idx)[0]

#----------------------------------------------------------------------------
