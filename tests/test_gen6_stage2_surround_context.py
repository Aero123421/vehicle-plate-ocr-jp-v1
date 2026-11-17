"""Concentric chart alignment and zero-initialized spatial context fusion."""
import numpy as np
import torch
import torch.nn.functional as F
from vpocr_det.quad_cropper import PlateQuadTrackNet, extract_dual_concentric_plate_rois


def test_canonical_sampling_fills_missing_fovea_from_same_surround_plane():
    model = PlateQuadTrackNet().eval()
    model.canonical_encoder = torch.nn.Identity()
    corners = torch.tensor([[[-.25, -.25], [1.25, -.25], [1.25, 1.25], [-.25, 1.25]]])
    fovea = torch.ones(1, model.config.base_ch, 96, 160)
    surround = torch.full((1, model.config.base_ch, 64, 96), 3.)
    patch, support, _ = model._canonical_features(fovea, corners, surround)
    # Constant radiance on each chart gives an independent partition-of-unity
    # reference, including fractional support at the fovea sensor boundary.
    expected = (3. - 2. * support).expand_as(patch[:, :-1])
    torch.testing.assert_close(patch[:, :-1], expected, atol=1e-6, rtol=1e-6)
    assert (support == 0).any() and (support == 1).any()


def test_coarse_center_score_localizes_beyond_fovea_without_clamping_geometry():
    from vpocr_det.quad_cropper import PlateQuadTrackLoss
    model = PlateQuadTrackNet().eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 96, 160), torch.zeros(1, 3, 64, 96))
    truth = out['mode_corners'][:, 0] + torch.tensor([.7, 0.])
    assert truth.mean(1)[0, 0] > 1.
    center = .5 + (truth.mean(1) - .5) / out['coarse_chart_scale']
    hh, ww = out['grid_hw']
    yy, xx = torch.meshgrid((torch.arange(hh)+.5)/hh, (torch.arange(ww)+.5)/ww, indexing='ij')
    logits = -((xx-center[0, 0]).square()+(yy-center[0, 1]).square()) * (hh*ww)
    logits = logits.reshape(1, 1, -1).expand(1, 2, -1).clone().requires_grad_()
    out['mode_cell_log_probs'] = logits.log_softmax(-1)
    criterion = PlateQuadTrackLoss()
    args = (torch.ones(1), truth, torch.zeros(1, dtype=torch.long), torch.ones(1))
    correct = criterion(out, *args)['loss_center']
    # Preserve the exact same geometric supervision while moving only spatial
    # evidence to the opposite side of the observed surround chart.
    out['mode_cell_log_probs'] = logits.reshape(1, 2, hh, ww).flip(-1).flatten(2).log_softmax(-1)
    wrong = criterion(out, *args)['loss_center']
    assert correct < wrong


def test_surround_context_uses_exact_concentric_chart_map():
    model = PlateQuadTrackNet().eval()
    h, w = 16, 24
    x = (torch.arange(w) + .5) * (2. / w) - 1.
    y = (torch.arange(h) + .5) * (2. / h) - 1.
    field = (x[None, :] + 2. * y[:, None])[None, None]
    actual = model._align_surround_context(field, (24, 40))
    xf = (torch.arange(40) + .5) * (2. / 40) - 1.
    yf = (torch.arange(24) + .5) * (2. / 24) - 1.
    # Use actual crop windows as the independent geometric reference.
    _, _, fov_window, sur_window = extract_dual_concentric_plate_rois(
        np.zeros((480, 640, 3), dtype=np.uint8), (100., 80., 220., 140.),
        fovea_scale=model.config.fovea_scale, surround_scale=model.config.surround_scale)
    expected = (xf[None, :] * (fov_window[2] / sur_window[2])
                + 2. * yf[:, None] * (fov_window[3] / sur_window[3]))[None, None]
    torch.testing.assert_close(actual, expected, atol=3e-7, rtol=1e-6)


def test_surround_context_fusion_preserves_init_and_receives_gradients():
    torch.manual_seed(8)
    model = PlateQuadTrackNet().eval()
    f = torch.rand(1, model.config.base_ch, 96, 160)
    s = torch.rand(1, model.config.surround_ch, 64, 96)
    u2 = F.avg_pool2d(model.ctx_down2(f), 2)
    u4 = F.avg_pool2d(model.ctx_down4(u2), 2)
    u8 = F.avg_pool2d(model.ctx_down8(u4), 2)
    original = .5 * (u4 + F.interpolate(u8, size=u4.shape[-2:], mode='nearest'))
    actual = model._context_features(f, s)
    torch.testing.assert_close(actual, model._align_fovea_context(original, original.shape[-2:]), atol=0, rtol=0)
    actual.square().mean().backward()
    grad = model.surround_ctx_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_full_surround_readout_preserves_physical_init_quad():
    model = PlateQuadTrackNet().eval()
    with torch.no_grad():
        model.cell_head.weight.zero_()
        model.cell_head.bias.zero_()
        out = model(torch.zeros(1, 3, 96, 160), torch.zeros(1, 3, 64, 96))
    expected = torch.tensor([[.25, .35], [.75, .35], [.75, .65], [.25, .65]])
    torch.testing.assert_close(out['mode_init_corners'], expected[None, None].expand(1, 2, 4, 2), atol=2e-6, rtol=2e-6)


def test_fovea_context_transport_matches_actual_wide_crop():
    model = PlateQuadTrackNet().eval()
    h, w = 24, 40
    x = (torch.arange(w) + .5) * (2. / w) - 1.
    y = (torch.arange(h) + .5) * (2. / h) - 1.
    field = (x[None, :] + 2. * y[:, None])[None, None]
    actual = model._align_fovea_context(field, (h, w))
    _, _, fov_window, sur_window = extract_dual_concentric_plate_rois(
        np.zeros((480, 640, 3), dtype=np.uint8), (100., 80., 220., 140.),
        fovea_scale=model.config.fovea_scale, surround_scale=model.config.surround_scale)
    gx = x * (sur_window[2] / fov_window[2])
    gy = y * (sur_window[3] / fov_window[3])
    expected = (gx[None, :] + 2. * gy[:, None])[None, None]
    inner = ((gx.abs() < 1. - 1. / w)[None, :] & (gy.abs() < 1. - 1. / h)[:, None])[None, None]
    outer = ((gx.abs() > 1. + 1. / w)[None, :] | (gy.abs() > 1. + 1. / h)[:, None])[None, None]
    torch.testing.assert_close(actual[inner], expected[inner], atol=3e-7, rtol=1e-6)
    assert torch.count_nonzero(actual[outer]) == 0


def test_coarse_offsets_keep_fovea_units_across_context_scales():
    from vpocr_det.quad_cropper import QuadTrackModelConfig
    for surround_scale in (3.4, 4.5, 6.8):
        model = PlateQuadTrackNet(QuadTrackModelConfig(surround_scale=surround_scale)).eval()
        with torch.no_grad():
            model.cell_head.weight.zero_()
            model.cell_head.bias.zero_()
        out = model(torch.zeros(1, 3, 96, 160), torch.zeros(1, 3, 64, 96))
        grad, = torch.autograd.grad(out['mode_init_corners'].sum(), model.displacement_head.bias)
        torch.testing.assert_close(grad, torch.ones_like(grad), atol=64 * torch.finfo(torch.float32).eps, rtol=0)


def test_integral_geometry_retains_fp32_under_bfloat16_autocast():
    from vpocr_det.quad_cropper import anchor_free_mode_init
    torch.manual_seed(42)
    logits = torch.randn(1, 2, 24, 40).bfloat16()
    corners = torch.randn(1, 2, 4, 2, 24, 40).bfloat16()
    expected = anchor_free_mode_init(logits.float(), corners.float())
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = anchor_free_mode_init(logits, corners)
    for a, b in zip(actual, expected):
        assert a.dtype == torch.float32
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_quality_training_logits_remain_learnable_for_invalid_mode():
    model = PlateQuadTrackNet().eval()
    with torch.no_grad():
        model.cell_head.weight.zero_()
        model.cell_head.bias.copy_(torch.tensor([0., 1.]))
        model.displacement_head.bias[:8].copy_(torch.tensor([-.2, -.1, .2, .1, .2, -.1, -.2, .1]))
    out = model(torch.rand(1, 3, 96, 160), torch.rand(1, 3, 64, 96))
    assert out['mode_valid'].tolist() == [[False, True]]
    assert out['selected_mode'].item() == 1
    torch.testing.assert_close(out['mode_weights'], torch.tensor([[0., 1.]]))
    torch.testing.assert_close(out['mode_logits'], torch.tensor([[0., 1.]]), atol=2e-6, rtol=2e-6)
    loss = F.cross_entropy(out['mode_logits'], torch.tensor([0]))
    assert torch.isfinite(loss) and loss < 2.
    grad, = torch.autograd.grad(loss, model.cell_head.bias)
    assert grad[0] < 0 and grad[1] > 0
