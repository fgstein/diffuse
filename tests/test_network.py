"""Network tests: shapes, topology validation, checkpointing, precision, conditioning, noise shift."""

import numpy as np
import pytest
import torch
from conftest import tiny_net
from training.networks import validate_topology
from training.loss import EDMLoss, compute_P_mean_eff, compute_sigma_max_hint, sigma_bin_index, SIGMA_BIN_EDGES

def _run(net, res, n=2, cond=None):
    x = torch.randn(n, 3, res, res)
    sigma = torch.full([n], 0.7)
    cond = {'class': torch.arange(n) % 3} if cond is None else cond
    return net(x, sigma, cond)

@pytest.mark.parametrize('res,patch,mult,attn,model_type', [
    (64, 1, (1, 2, 2, 2), (16, 8), 'SongUNet'),
    (224, 2, (1, 2, 3, 4), (28, 14), 'SongUNet'),
    (32, 1, (1, 2), (16,), 'SongUNet'),
    (32, 2, (1, 2), (8,), 'DhariwalUNet'),
])
def test_forward_shapes(res, patch, mult, attn, model_type):
    net = tiny_net(res, patch_size=patch, channel_mult=mult, attn=attn, model_type=model_type, model_channels=8 if res == 224 else 16).eval()
    with torch.no_grad():
        y = _run(net, res)
    assert y.shape == (2, 3, res, res) and y.dtype == torch.float32 and torch.isfinite(y).all()

def test_grad_ckpt_same_output_and_grads():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 32, 32)
    outs = []
    for ckpt in [False, True]:
        net = tiny_net(32, dropout=0, label_dropout=0, grad_ckpt=ckpt).train()
        y = net(x, torch.full([2], 0.5), {'class': torch.tensor([0, 1])})
        y.square().mean().backward()
        outs.append((y.detach(), [p.grad.clone() for p in net.parameters()]))
    assert torch.allclose(outs[0][0], outs[1][0], atol=1e-6)
    for g0, g1 in zip(outs[0][1], outs[1][1]):
        assert torch.allclose(g0, g1, atol=1e-5)

def test_bf16_autocast_cpu():
    net = tiny_net(32, precision='bf16').eval()
    with torch.no_grad():
        y = _run(net, 32)
        y32 = _run(net, 32) if False else net(torch.zeros(2, 3, 32, 32), torch.full([2], 0.7), {'class': torch.tensor([0, 1])}, force_fp32=True)
    assert y.dtype == torch.float32 and torch.isfinite(y).all() and torch.isfinite(y32).all()

def test_bf16_close_to_fp32():
    net = tiny_net(32, dropout=0).eval()
    x = torch.randn(2, 3, 32, 32)
    s = torch.full([2], 1.0)
    c = {'class': torch.tensor([0, 2])}
    with torch.no_grad():
        y32 = net(x, s, c)
        net.precision = 'bf16'
        y16 = net(x, s, c)
    assert (y32 - y16).abs().max() < 0.1

@pytest.mark.parametrize('res,patch,mult,attn,match', [
    (64, 1, (1, 2, 2), (12,), 'do not occur'),
    (40, 1, (1, 2, 2, 2, 2), (), 'divisible'),
    (224, 1, (1, 2, 2, 2, 2, 2, 2), (), 'divisible'),
    (64, 3, (1,), (), 'not divisible by patch size'),
])
def test_invalid_topology(res, patch, mult, attn, match):
    with pytest.raises(ValueError, match=match):
        validate_topology(res, patch, mult, attn)
    with pytest.raises(ValueError, match=match):
        tiny_net(res, patch_size=patch, channel_mult=mult, attn=attn)

def test_label_dropout_one_equals_null():
    net = tiny_net(32, dropout=0, label_dropout=1.0).train()
    x = torch.randn(3, 3, 32, 32)
    s = torch.full([3], 0.8)
    with torch.no_grad():
        y_drop = net(x, s, {'class': torch.tensor([0, 1, 2])})
        y_null = net(x, s, net.null_cond(3, x.device))
    assert torch.equal(y_drop, y_null)
    # Eval mode: no dropout, the class matters.
    net.eval()
    with torch.no_grad():
        y_c = net(x, s, {'class': torch.tensor([0, 1, 2])})
        y_n = net(x, s, net.null_cond(3, x.device))
        y_c2 = net(x, s, {'class': torch.tensor([0, 1, 2])})
    assert not torch.allclose(y_c, y_n)
    assert torch.equal(y_c, y_c2)

def test_null_embedding_is_learned_not_zero():
    net = tiny_net(32)
    enc = net.cond_stack.encoders['class']
    assert enc.embed.weight.shape[0] == 4 # 3 classes + NULL
    assert enc.embed.weight[enc.null_index].abs().sum() > 0 and enc.embed.weight.requires_grad

def test_legacy_label_inputs():
    net = tiny_net(32, dropout=0).eval()
    x = torch.randn(2, 3, 32, 32)
    s = torch.full([2], 0.5)
    with torch.no_grad():
        a = net(x, s, {'class': torch.tensor([1, 2])})
        b = net(x, s, torch.tensor([1, 2]))                     # bare LongTensor
        c = net(x, s, torch.eye(3)[[1, 2]])                     # EDM-style one-hot
        d = net(x, s, class_labels=torch.tensor([1, 2]))
        e = net(x, s, None)
        f = net(x, s, net.null_cond(2, x.device))
    assert torch.equal(a, b) and torch.equal(a, c) and torch.equal(a, d)
    assert torch.equal(e, f)

def test_unconditional_network():
    net = tiny_net(32, num_classes=0).eval()
    with torch.no_grad():
        y = net(torch.randn(2, 3, 32, 32), torch.full([2], 0.5), {})
    assert y.shape == (2, 3, 32, 32)

def test_noise_shift():
    assert compute_P_mean_eff(-1.2) == -1.2
    assert compute_P_mean_eff(-1.2, 224, None) == -1.2
    assert np.isclose(compute_P_mean_eff(-1.2, 224, 64), -1.2 + np.log(224 / 64))
    assert np.isclose(np.log(224 / 64), 1.2528, atol=1e-3)
    assert np.isclose(compute_P_mean_eff(-1.2, 64, 64), -1.2)
    loss = EDMLoss(P_mean=-1.2, P_std=1.2, resolution=224, noise_shift_base=64)
    assert np.isclose(loss.P_mean_eff, -1.2 + np.log(3.5))
    assert np.isclose(compute_sigma_max_hint(224, 64), 280.0)
    assert compute_sigma_max_hint(224, 64, cap=200) == 200
    assert compute_sigma_max_hint(64, None) == 80.0

def test_loss_and_sigma_bins():
    net = tiny_net(32).train()
    loss_fn = EDMLoss()
    images = torch.rand(4, 3, 32, 32) * 2 - 1
    loss, sigma = loss_fn(net, images, {'class': torch.tensor([0, 1, 2, 0])}, return_sigma=True)
    assert loss.shape == images.shape and sigma.shape == (4,)
    loss.mean().backward()
    b = sigma_bin_index(torch.tensor([1e-4, 0.002, 0.5, 80, 1e4]))
    assert b.tolist()[0] == 0 and b.tolist()[-1] == len(SIGMA_BIN_EDGES) - 2
