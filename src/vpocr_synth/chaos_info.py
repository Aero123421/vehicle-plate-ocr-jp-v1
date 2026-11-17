# SPDX-License-Identifier: Apache-2.0
"""Chaotic dynamical systems & information-theoretic synthesis/curriculum primitives (`v0.7.0`).

Provides deterministic, seed-reproducible mathematical building blocks for Sim-to-Real robustness:
1. **Chaos Theory**:
   - `lorenz_rk4_offset`: 4th-order Runge-Kutta integration on the Lorenz strange attractor
     (`sigma=10, rho=28, beta=8/3`, largest Lyapunov exponent `lambda_1 ~= +0.906 > 0`) for
     broadband, non-periodic camera shake, pan/tilt drift, and Stage-2 crop corner intermittency.
   - `coupled_map_lattice_2d`: Kaneko 2D spatiotemporal Coupled Map Lattice (CML) with chaotic
     logistic local dynamics (`r = 3.97`) and diffusive spatial coupling for dappled foliage/urban
     shadow fields and structured non-plate background clutter that breaks sinusoidal shortcuts.
   - `chebyshev_ergodic_grid`: Ulam-von Neumann / Chebyshev `T_2` + irrational Kronecker Weyl
     rotation for space-filling continuous multi-scale & anisotropic resolution sampling.
2. **Information Theory**:
   - `shannon_slot_capacity_weights`: Slot importance derived from prior Shannon entropy
     `H(S_k) = log2 |V_k|` normalized by physical glyph spatial channel capacity.
   - `focal_self_information_ce`: Calibration-preserving proper cross-entropy weighted by
     detached self-information surprise `(1 - p_y)^gamma` to prevent easy 65mm serial digits
     from starving high-entropy 32mm Hiragana and 40mm multi-stroke Kanji gradients.
   - `boltzmann_kl_token_weights`: Jaynes Maximum-Entropy / Gibbs distribution
     `q*(c) proportional to exp(beta * E[D_KL(delta_y || p_hat)])` for active hard-token curriculum.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

_GOLDEN_CONJ = 0.6180339887498949
_SILVER_CONJ = 0.4142135623730951


def _lorenz_rhs(x: float, y: float, z: float, sigma: float, rho: float, beta: float) -> tuple[float, float, float]:
    return (sigma * (y - x), x * (rho - z) - y, x * y - beta * z)


def lorenz_rk4_offset(
    t: float,
    seed_phase: float,
    *,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8.0 / 3.0,
    time_scale: float = 2.6,
) -> tuple[float, float, float]:
    """Evaluate a smooth, zero-at-t=0 trajectory segment on the Lorenz strange attractor.

    Deterministically burns in 56 RK4 steps from `seed_phase` onto the attractor basin, then
    integrates continuously to `t * time_scale` so `eval(0.0) == (0.0, 0.0, 0.0)` identically.
    Returns bounded `(ux, uy, uz)` in `[-1, 1]^3`.
    """
    tf = float(t)
    if abs(tf) < 1e-12:
        return (0.0, 0.0, 0.0)

    # Deterministic initial point inside the attractor basin parameterized by seed_phase
    ph = float(seed_phase)
    x = 8.5 * math.cos(ph * 3.17 + 0.4)
    y = 8.5 * math.sin(ph * 5.23 + 1.1)
    z = 23.0 + 6.0 * math.sin(ph * 2.09)

    dt_burn = 0.015
    for _ in range(56):
        k1 = _lorenz_rhs(x, y, z, sigma, rho, beta)
        k2 = _lorenz_rhs(x + 0.5 * dt_burn * k1[0], y + 0.5 * dt_burn * k1[1], z + 0.5 * dt_burn * k1[2], sigma, rho, beta)
        k3 = _lorenz_rhs(x + 0.5 * dt_burn * k2[0], y + 0.5 * dt_burn * k2[1], z + 0.5 * dt_burn * k2[2], sigma, rho, beta)
        k4 = _lorenz_rhs(x + dt_burn * k3[0], y + dt_burn * k3[1], z + dt_burn * k3[2], sigma, rho, beta)
        x += (dt_burn / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        y += (dt_burn / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        z += (dt_burn / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])

    x0, y0, z0 = x, y, z
    span = max(-2.4, min(2.4, tf * float(time_scale)))
    n_steps = max(4, min(96, int(math.ceil(abs(span) / 0.012))))
    h = span / float(n_steps)
    for _ in range(n_steps):
        k1 = _lorenz_rhs(x, y, z, sigma, rho, beta)
        k2 = _lorenz_rhs(x + 0.5 * h * k1[0], y + 0.5 * h * k1[1], z + 0.5 * h * k1[2], sigma, rho, beta)
        k3 = _lorenz_rhs(x + 0.5 * h * k2[0], y + 0.5 * h * k2[1], z + 0.5 * h * k2[2], sigma, rho, beta)
        k4 = _lorenz_rhs(x + h * k3[0], y + h * k3[1], z + h * k3[2], sigma, rho, beta)
        x += (h / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        y += (h / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        z += (h / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])

    return (
        math.tanh((x - x0) / 14.0),
        math.tanh((y - y0) / 16.0),
        math.tanh((z - z0) / 16.0),
    )


def coupled_map_lattice_2d(
    batch_size: int,
    h: int,
    w: int,
    *,
    steps: int = 5,
    r: float = 3.97,
    coupling: float = 0.28,
    device: torch.device | str = "cpu",
    init_u: Tensor | None = None,
) -> Tensor:
    """Evolve a Kaneko 2D Coupled Map Lattice `(B, 1, H, W)` in `[0, 1]` on GPU/CPU.

    Uses chaotic logistic local dynamics `f(x) = r * x * (1 - x)` (`r = 3.97` in positive-entropy
    chaotic regime) with 4-neighbor Laplacian diffusion to synthesize non-periodic dappled shadows
    and broadband structured urban clutter.
    """
    dev = torch.device(device)
    if init_u is None:
        x = 0.08 + 0.84 * torch.rand((int(batch_size), 1, int(h), int(w)), device=dev, dtype=torch.float32)
    else:
        x = (0.08 + 0.84 * init_u.to(device=dev, dtype=torch.float32).clamp(0.0, 1.0)).reshape(
            int(batch_size), 1, int(h), int(w)
        )
    eps = float(max(0.0, min(0.85, coupling)))
    rf = float(r)
    for _ in range(max(1, int(steps))):
        fx = (rf * x * (1.0 - x)).clamp(1e-4, 1.0 - 1e-4)
        pad = F.pad(fx, (1, 1, 1, 1), mode="circular")
        nbr = 0.25 * (pad[:, :, 1:-1, :-2] + pad[:, :, 1:-1, 2:] + pad[:, :, :-2, 1:-1] + pad[:, :, 2:, 1:-1])
        x = (1.0 - eps) * fx + eps * nbr
    return x.clamp(0.0, 1.0)


def chaotic_intermittent_corner_burst(
    u_seed: Tensor,
    bsz: int,
    t_len: int,
    *,
    r: float = 3.985,
) -> Tensor:
    """Generate zero-mean heavy-tailed chaotic intermittency `(B, T, 4, 2)` via logistic-Chebyshev orbit.

    Simulates realistic Stage-2 tracking micro-slips during sudden vehicle pitch/handheld camera jerk
    that stationary Gaussian AR(1) noise fails to span.
    """
    x = (0.11 + 0.78 * u_seed.float().clamp(0.0, 1.0)).reshape(bsz, 8)
    seq: list[Tensor] = []
    for t in range(int(t_len)):
        # Burn 3 chaotic iterations per frame step to decorrelate higher moments while keeping state coupling
        for _ in range(3):
            x = (r * x * (1.0 - x)).clamp(1e-4, 1.0 - 1e-4)
        # Map arcsine-invariant [0, 1] orbit to zero-mean intermittent burst
        centered = torch.cos(math.pi * x)  # exact uniform-angle Chebyshev projection in [-1, 1]
        burst_gate = (centered.abs() > 0.78).float()
        seq.append((centered * (0.35 + 1.15 * burst_gate)).reshape(bsz, 4, 2))
    return torch.stack(seq, dim=1)


def chebyshev_ergodic_scales(
    u_base: Tensor,
    *,
    lo: float = 0.23,
    hi: float = 0.96,
) -> tuple[Tensor, Tensor]:
    """Map uniform/seed tensor `u_base` `(N,)` to space-filling `(scale_h, scale_w)` resolution pairs.

    Combines a chaotic Chebyshev `T_3` map with irrational Weyl rotation so plate resolutions span
    the entire sub-Nyquist-to-crisp continuum (`18px..154px`) plus mild horizontal foreshortening
    anisotropy (`w_scale / h_scale in [0.74, 1.12]`) observed on oblique traffic cameras.
    """
    u = u_base.float().clamp(1e-4, 1.0 - 1e-4)
    idx = torch.arange(u.numel(), device=u.device, dtype=torch.float32).reshape_as(u)
    q1 = torch.frac(u + idx * _GOLDEN_CONJ)
    q2 = torch.frac(0.5 * (1.0 - torch.cos(3.0 * torch.acos(2.0 * u - 1.0))) + idx * _SILVER_CONJ)
    iso = float(lo) + (float(hi) - float(lo)) * (q1 ** 0.85)
    aniso = 0.74 + 0.38 * q2  # horizontal yaw foreshortening factor
    scale_h = iso.clamp(0.20, 1.0)
    scale_w = (iso * aniso).clamp(0.18, 1.0)
    return scale_h, scale_w


def shannon_slot_capacity_weights(device: torch.device | str) -> Tensor:
    """Principled 12-slot weights proportional to `H(S_k) / SpatialCapacity(S_k)`.

    Balances prior Shannon information (`7.11 bits` across 138 Kanji regions, `5.49 bits` on 45-way
    32mm Hiragana/Latin slot 7, variable 1/2/3-char classification termination on slots 4..6, and
    `3.32 bits` per 65mm serial digit) against physical stroke area so upper-row Kanji/Classification
    and Hiragana slots are never gradient-starved by large serial digits.
    """
    return torch.tensor(
        [3.05, 3.05, 1.80, 1.40, 1.65, 1.85, 2.05, 2.30, 1.20, 1.20, 1.20, 1.20],
        device=device,
        dtype=torch.float32,
    )


def focal_self_information_ce(
    logits: Tensor,
    targets: Tensor,
    *,
    gamma: float = 0.45,
    floor_mix: float = 0.72,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Proper cross-entropy modulated by detached self-information surprise `(1 - p_y)^gamma`.

    Detaching the surprise weight preserves the exact Bayes-optimal argmax and probability calibration
    stationary point (`grad = w * (p - y)` vanishes iff `p == y_smooth`) while amplifying hard-tail
    gradients on low-SNR Kanji/Hiragana strokes.
    """
    flat_l = logits.float().reshape(-1, logits.shape[-1])
    flat_t = targets.clamp(0, logits.shape[-1] - 1).reshape(-1)
    ce = F.cross_entropy(flat_l, flat_t, reduction="none", label_smoothing=float(label_smoothing))
    with torch.no_grad():
        probs = F.softmax(flat_l, dim=-1)
        p_t = probs.gather(-1, flat_t.unsqueeze(-1)).squeeze(-1).clamp(1e-5, 1.0)
        mod = float(floor_mix) + (1.0 - float(floor_mix)) * ((1.0 - p_t).clamp_min(0.0) ** float(gamma)) * 1.65
    return (ce * mod).reshape(targets.shape)


def boltzmann_kl_token_weights(
    kl_surprise_or_err: np.ndarray | Tensor | Mapping[str, float],
    vocab_size: int,
    *,
    beta: float = 1.35,
    uniform_mix: float = 0.14,
) -> np.ndarray:
    """Compute Jaynes Maximum-Entropy / Boltzmann active curriculum weights over token IDs `1..V-1`."""
    scores = np.zeros((int(vocab_size),), dtype=np.float64)
    if isinstance(kl_surprise_or_err, torch.Tensor):
        arr = kl_surprise_or_err.detach().float().cpu().numpy().reshape(-1)
        n = min(len(scores), len(arr))
        scores[:n] = np.maximum(0.0, arr[:n])
    elif isinstance(kl_surprise_or_err, np.ndarray):
        arr = kl_surprise_or_err.astype(np.float64).reshape(-1)
        n = min(len(scores), len(arr))
        scores[:n] = np.maximum(0.0, arr[:n])
    scores[0] = 0.0  # Exclude <PAD>
    pos = scores[1:]
    if pos.size == 0 or float(pos.max()) <= 1e-9:
        return np.zeros((int(vocab_size),), dtype=np.float32)

    # Standardize positive-support surprise so `beta` is scale-invariant across training stages
    active = pos[pos > 0]
    scale = max(1e-4, float(np.percentile(active, 90)))
    norm_s = np.clip(pos / scale, 0.0, 3.0)
    gibbs = np.where(pos > 0, np.exp(float(beta) * norm_s) - 1.0, 0.0)
    if float(gibbs.sum()) <= 1e-9:
        return np.zeros((int(vocab_size),), dtype=np.float32)
    gibbs = gibbs / float(gibbs.sum())
    n_active = float(max(1, int((pos > 0).sum())))
    unif = np.where(pos > 0, 1.0 / n_active, 0.0)
    mixed = (1.0 - float(uniform_mix)) * gibbs + float(uniform_mix) * unif
    out = np.zeros((int(vocab_size),), dtype=np.float32)
    out[1:] = mixed.astype(np.float32)
    return out
