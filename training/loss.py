# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from torch_utils import persistence

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Fixed log-sigma bins used for per-sigma loss diagnostics (comparable across runs).

SIGMA_BIN_EDGES = np.exp(np.linspace(np.log(0.002), np.log(80), 9)).tolist() # 8 bins

def sigma_bin_index(sigma):
    """Map sigma values (any shape) to bin indices 0..len(SIGMA_BIN_EDGES)-2 (clamped)."""
    edges = torch.as_tensor(SIGMA_BIN_EDGES[1:-1], dtype=torch.float32, device=sigma.device)
    return torch.bucketize(sigma.to(torch.float32).flatten(), edges)

def compute_P_mean_eff(P_mean, resolution=None, noise_shift_base=None):
    """Resolution-dependent noise shift (Simple Diffusion, Hoogeboom et al. 2023):
    P_mean_eff = P_mean + ln(resolution / base). Shifts log-SNR by -2 ln(res/base)."""
    if noise_shift_base is None or not noise_shift_base or resolution is None:
        return float(P_mean)
    return float(P_mean + np.log(resolution / noise_shift_base))

def compute_sigma_max_hint(resolution=None, noise_shift_base=None, sigma_max=80.0, cap=None):
    """Sampler hint matching the noise shift: sigma_max * resolution / base (optionally capped)."""
    s = float(sigma_max)
    if noise_shift_base and resolution is not None:
        s = s * resolution / noise_shift_base
    if cap is not None:
        s = min(s, float(cap))
    return s

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM), with an optional
# resolution-dependent shift of the training noise distribution.

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, resolution=None, noise_shift_base=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.resolution = resolution
        self.noise_shift_base = noise_shift_base
        self.P_mean_eff = compute_P_mean_eff(P_mean, resolution, noise_shift_base)

    def __call__(self, net, images, labels=None, augment_pipe=None, return_sigma=False):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean_eff).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        if return_sigma:
            return loss, sigma.flatten()
        return loss

    def weight(self, sigma):
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

#----------------------------------------------------------------------------
