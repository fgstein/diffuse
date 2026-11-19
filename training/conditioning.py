# This file is part of a derivative of NVIDIA's EDM codebase and is
# licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (see LICENSE.txt).

"""Modular conditioning encoders.

A *condition* is one raw per-sample input (e.g. a class index). Each condition
is handled by a `ConditionEncoder` that maps it to a vector of size `emb_dim`.
The `ConditionStack` sums the outputs of all encoders; the UNet adds the sum to
its noise embedding, so every condition modulates every residual block through
the existing adaptive scale/shift.

Conditions travel through the code base as a dict:

    cond = {"class": LongTensor[N], ...}

from the Dataset (via the default collate function) to the loss, the
preconditioner and the UNet.

Classifier-free guidance (CFG)
------------------------------
During training, conditions are dropped at random (per-condition probability
`dropout`, plus a probability `joint_dropout` of dropping *all* conditions of a
sample at once). A dropped condition is replaced by its encoder's *learned*
null embedding. At sampling time, `null_cond()` builds the raw values that
mean "unconditional"; a missing key in the cond dict also means "dropped".

Adding a new condition type
---------------------------
1. Write a subclass of `ConditionEncoder` (see `ScalarFourierEmbedding` below
   for a sketch of a scalar condition), decorate it with
   `@persistence.persistent_class`, and register it in `ENCODER_TYPES`.
2. Make the dataset emit the raw value under the condition's name (see
   `CSVImageDataset` in training/dataset.py: it already returns a dict).
3. Add an entry to the condition config (list of dicts, stored in snapshots):
       dict(name='age', type='scalar', dropout=0.1, min=0, max=100)
"""

import numpy as np
import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Base class.

@persistence.persistent_class
class ConditionEncoder(torch.nn.Module):
    """Maps one raw condition to a vector of size emb_dim (added to the noise embedding)."""

    def __init__(self, name, emb_dim, dropout=0.0):
        super().__init__()
        self.name = name
        self.emb_dim = emb_dim
        self.dropout = float(dropout)

    def forward(self, cond_value, drop_mask):
        """cond_value: raw values for N samples. drop_mask: BoolTensor[N] (True = use null).
        Returns: Tensor[N, emb_dim]."""
        raise NotImplementedError

    def null_value(self, n, device):
        """Raw value(s) meaning 'unconditional' for n samples."""
        raise NotImplementedError

    def is_null(self, cond_value):
        """BoolTensor[N]: which raw values already are the null value."""
        raise NotImplementedError

    def coerce(self, cond_value, n, device):
        """Convert user-provided raw values to the canonical tensor format."""
        return torch.as_tensor(cond_value, device=device)

#----------------------------------------------------------------------------
# Categorical condition: learned embedding table with a learned NULL token.

@persistence.persistent_class
class ClassEmbedding(ConditionEncoder):
    def __init__(self, name, emb_dim, num_classes, dropout=0.0, init_std=1.0):
        super().__init__(name=name, emb_dim=emb_dim, dropout=dropout)
        assert num_classes >= 1
        self.num_classes = int(num_classes)
        self.null_index = self.num_classes # index num_classes == NULL token
        self.embed = torch.nn.Embedding(self.num_classes + 1, emb_dim)
        torch.nn.init.normal_(self.embed.weight, std=init_std)

    def coerce(self, cond_value, n, device):
        v = torch.as_tensor(cond_value, device=device)
        if v.is_floating_point() and v.ndim == 2:
            # Backward compatibility: EDM-style one-hot vectors.
            all_zero = (v.abs().sum(dim=1) == 0)
            v = torch.where(all_zero, torch.full_like(v[:, 0], self.null_index), v.argmax(dim=1).to(v.dtype))
        v = v.to(torch.int64).reshape(-1)
        if v.numel() == 1 and n > 1:
            v = v.expand(n)
        return v

    def forward(self, cond_value, drop_mask):
        idx = cond_value.to(torch.int64)
        if (idx < 0).any() or (idx > self.null_index).any():
            raise ValueError(f'Condition "{self.name}": class index out of range [0, {self.num_classes}]')
        if drop_mask is not None:
            idx = torch.where(drop_mask, torch.full_like(idx, self.null_index), idx)
        return self.embed(idx)

    def null_value(self, n, device):
        return torch.full([n], self.null_index, dtype=torch.int64, device=device)

    def is_null(self, cond_value):
        return cond_value.to(torch.int64) == self.null_index

#----------------------------------------------------------------------------
# Scalar condition (extension point; not wired to the CLI yet).
# Values are normalized to [0, 1] with (min, max) and embedded with random
# Fourier features followed by a linear layer. NaN means NULL.

@persistence.persistent_class
class ScalarFourierEmbedding(ConditionEncoder):
    def __init__(self, name, emb_dim, min=0.0, max=1.0, num_freqs=64, scale=16.0, dropout=0.0):
        super().__init__(name=name, emb_dim=emb_dim, dropout=dropout)
        self.vmin = float(min)
        self.vmax = float(max)
        self.register_buffer('freqs', torch.randn(num_freqs) * scale)
        self.proj = torch.nn.Linear(2 * num_freqs, emb_dim)
        self.null_emb = torch.nn.Parameter(torch.randn(emb_dim))

    def coerce(self, cond_value, n, device):
        v = torch.as_tensor(cond_value, device=device, dtype=torch.float32).reshape(-1)
        return v.expand(n) if (v.numel() == 1 and n > 1) else v

    def forward(self, cond_value, drop_mask):
        v = cond_value.to(torch.float32)
        null = torch.isnan(v)
        if drop_mask is not None:
            null = null | drop_mask
        u = ((v - self.vmin) / max(self.vmax - self.vmin, 1e-8)).nan_to_num(0.0)
        f = u[:, None] * (2 * np.pi * self.freqs[None, :])
        e = self.proj(torch.cat([f.cos(), f.sin()], dim=1))
        return torch.where(null[:, None], self.null_emb[None, :].to(e.dtype), e)

    def null_value(self, n, device):
        return torch.full([n], float('nan'), device=device)

    def is_null(self, cond_value):
        return torch.isnan(cond_value.to(torch.float32))

#----------------------------------------------------------------------------

ENCODER_TYPES = {
    'class':  ClassEmbedding,
    'scalar': ScalarFourierEmbedding,
}

def make_class_cond_config(num_classes, dropout=0.1):
    """Default condition config: one categorical condition named 'class'."""
    return [dict(name='class', type='class', num_classes=int(num_classes), dropout=float(dropout))]

#----------------------------------------------------------------------------
# Stack of condition encoders.

@persistence.persistent_class
class ConditionStack(torch.nn.Module):
    def __init__(self,
        cond_config,            # List of dicts: dict(name=..., type=..., dropout=..., **encoder_kwargs).
        emb_dim,                # Output dimensionality (= noise embedding dim at the injection point).
        joint_dropout = 0.0,    # Probability of dropping ALL conditions of a sample jointly.
    ):
        super().__init__()
        self.cond_config = [dict(c) for c in cond_config]
        self.emb_dim = emb_dim
        self.joint_dropout = float(joint_dropout)
        self.encoders = torch.nn.ModuleDict()
        for c in self.cond_config:
            c = dict(c)
            name, typ = c.pop('name'), c.pop('type')
            if typ not in ENCODER_TYPES:
                raise ValueError(f'Unknown condition type "{typ}" for condition "{name}"; known: {sorted(ENCODER_TYPES)}')
            if name in self.encoders:
                raise ValueError(f'Duplicate condition name "{name}"')
            self.encoders[name] = ENCODER_TYPES[typ](name=name, emb_dim=emb_dim, **c)

    @property
    def names(self):
        return list(self.encoders.keys())

    def normalize(self, cond, n, device):
        """Accept a dict, a bare tensor (legacy class labels), or None; return a dict of
        canonical tensors. Missing conditions are filled with their null value."""
        if cond is None:
            cond = {}
        elif torch.is_tensor(cond) or isinstance(cond, np.ndarray):
            if len(self.encoders) == 0 or (cond.ndim == 2 and cond.shape[1] == 0):
                cond = {}
            else:
                cond = {self.names[0]: cond}
        out = {}
        for name, enc in self.encoders.items():
            out[name] = enc.coerce(cond[name], n, device) if name in cond else enc.null_value(n, device)
        return out

    def null_cond(self, n, device):
        return {name: enc.null_value(n, device) for name, enc in self.encoders.items()}

    def forward(self, cond, n, device):
        """Return the summed condition embedding [n, emb_dim], or None if there are no conditions."""
        if len(self.encoders) == 0:
            return None
        cond = self.normalize(cond, n, device)
        joint = None
        if self.training and self.joint_dropout > 0:
            joint = torch.rand([n], device=device) < self.joint_dropout
        emb = None
        for name, enc in self.encoders.items():
            drop = None
            if self.training and enc.dropout > 0:
                drop = torch.rand([n], device=device) < enc.dropout
            if joint is not None:
                drop = joint if drop is None else (drop | joint)
            e = enc(cond[name], drop)
            emb = e if emb is None else emb + e
        return emb

#----------------------------------------------------------------------------
