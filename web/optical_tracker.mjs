// SPDX-License-Identifier: Apache-2.0
// Sub-pixel Coarse-to-Fine ZNCC + 4-DOF Inverse-Compositional Lucas-Kanade Optical Tracker.
// Shared between worker.mjs (Stage-2 pre-alignment & jitter suppression) and app.mjs (60 FPS zero-lag lock-on).

export const OPT_TW = 40;
export const OPT_TH = 20;
export const OPT_PAD = 0.08;
export const OPT_SPAN = 1 + 2 * OPT_PAD; // 1.16

const CW = 20;
const CH = 10;
const N_FINE = OPT_TW * OPT_TH;
const N_COARSE = CW * CH;

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

// Reusable scratch buffers (single-threaded per realm: main thread & worker each get their own module instance).
const sampleBuf = new Float32Array(N_FINE);
const coarseBuf = new Float32Array(N_COARSE);

export function quadBasis(corners) {
  const [c0, c1, c2, c3] = corners;
  const mx = 0.25 * (c0[0] + c1[0] + c2[0] + c3[0]);
  const my = 0.25 * (c0[1] + c1[1] + c2[1] + c3[1]);
  const aux = 0.5 * (c1[0] - c0[0] + c2[0] - c3[0]);
  const auy = 0.5 * (c1[1] - c0[1] + c2[1] - c3[1]);
  const avx = 0.5 * (c3[0] - c0[0] + c2[0] - c1[0]);
  const avy = 0.5 * (c3[1] - c0[1] + c2[1] - c1[1]);
  const auvx = c0[0] - c1[0] + c2[0] - c3[0];
  const auvy = c0[1] - c1[1] + c2[1] - c3[1];
  const width = Math.max(2, Math.hypot(aux, auy));
  const height = Math.max(2, Math.hypot(avx, avy));
  return { mx, my, aux, auy, avx, avy, auvx, auvy, width, height };
}

export function warpToCorners(basis, tu, tv, su, sv) {
  const { mx, my, aux, auy, avx, avy, auvx, auvy } = basis;
  const u0 = OPT_SPAN * tu - 0.5 * su;
  const u1 = OPT_SPAN * tu + 0.5 * su;
  const v0 = OPT_SPAN * tv - 0.5 * sv;
  const v1 = OPT_SPAN * tv + 0.5 * sv;
  return [
    [mx + u0 * aux + v0 * avx + u0 * v0 * auvx, my + u0 * auy + v0 * avy + u0 * v0 * auvy],
    [mx + u1 * aux + v0 * avx + u1 * v0 * auvx, my + u1 * auy + v0 * avy + u1 * v0 * auvy],
    [mx + u1 * aux + v1 * avx + u1 * v1 * auvx, my + u1 * auy + v1 * avy + u1 * v1 * auvy],
    [mx + u0 * aux + v1 * avx + u0 * v1 * auvx, my + u0 * auy + v1 * avy + u0 * v1 * auvy],
  ];
}

export function cornersToWarp(basis, corners) {
  const initB = quadBasis(corners);
  const dx = initB.mx - basis.mx;
  const dy = initB.my - basis.my;
  const det = basis.aux * basis.avy - basis.auy * basis.avx;
  let tu = 0, tv = 0;
  if (Math.abs(det) > 1e-6) {
    tu = (dx * basis.avy - dy * basis.avx) / (det * OPT_SPAN);
    tv = (basis.aux * dy - basis.auy * dx) / (det * OPT_SPAN);
  }
  const su = clamp(initB.width / basis.width, 0.65, 1.55);
  const sv = clamp(initB.height / basis.height, 0.65, 1.55);
  return { tu, tv, su, sv };
}

export function sampleGrayBilinear(gray, gw, gh, x, y) {
  const cx = clamp(x, 0, gw - 1.001);
  const cy = clamp(y, 0, gh - 1.001);
  const x0 = cx | 0;
  const y0 = cy | 0;
  const fx = cx - x0;
  const fy = cy - y0;
  const r0 = y0 * gw + x0;
  const r1 = r0 + gw;
  return (
    gray[r0] * (1 - fx) * (1 - fy) +
    gray[r0 + 1] * fx * (1 - fy) +
    gray[r1] * (1 - fx) * fy +
    gray[r1 + 1] * fx * fy
  );
}

export function sampleRgbaLumaBilinear(rgba, w, h, x, y) {
  const cx = clamp(x, 0, w - 1.001);
  const cy = clamp(y, 0, h - 1.001);
  const x0 = cx | 0;
  const y0 = cy | 0;
  const fx = cx - x0;
  const fy = cy - y0;
  const a = (y0 * w + x0) * 4;
  const b = a + 4;
  const d = a + w * 4;
  const e = d + 4;
  const la = 0.299 * rgba[a] + 0.587 * rgba[a + 1] + 0.114 * rgba[a + 2];
  const lb = 0.299 * rgba[b] + 0.587 * rgba[b + 1] + 0.114 * rgba[b + 2];
  const ld = 0.299 * rgba[d] + 0.587 * rgba[d + 1] + 0.114 * rgba[d + 2];
  const le = 0.299 * rgba[e] + 0.587 * rgba[e + 1] + 0.114 * rgba[e + 2];
  return (
    la * (1 - fx) * (1 - fy) +
    lb * fx * (1 - fy) +
    ld * (1 - fx) * fy +
    le * fx * fy
  );
}

export function extractPatchFromRgba(rgba, w, h, corners) {
  const basis = quadBasis(corners);
  const patch = new Uint8Array(N_FINE);
  let idx = 0;
  for (let j = 0; j < OPT_TH; j++) {
    const v = OPT_SPAN * ((j + 0.5) / OPT_TH - 0.5);
    for (let i = 0; i < OPT_TW; i++) {
      const u = OPT_SPAN * ((i + 0.5) / OPT_TW - 0.5);
      const px = basis.mx + u * basis.aux + v * basis.avx + u * v * basis.auvx;
      const py = basis.my + u * basis.auy + v * basis.avy + u * v * basis.auvy;
      patch[idx++] = clamp(Math.round(sampleRgbaLumaBilinear(rgba, w, h, px, py)), 0, 255);
    }
  }
  return patch;
}

function invert4x4(m) {
  const a = Float64Array.from(m);
  const inv = new Float64Array(16);
  inv[0] = 1; inv[5] = 1; inv[10] = 1; inv[15] = 1;
  for (let k = 0; k < 4; k++) {
    let p = k;
    for (let i = k + 1; i < 4; i++) {
      if (Math.abs(a[i * 4 + k]) > Math.abs(a[p * 4 + k])) p = i;
    }
    if (Math.abs(a[p * 4 + k]) < 1e-9) return null;
    if (p !== k) {
      for (let j = 0; j < 4; j++) {
        const tmpA = a[k * 4 + j]; a[k * 4 + j] = a[p * 4 + j]; a[p * 4 + j] = tmpA;
        const tmpI = inv[k * 4 + j]; inv[k * 4 + j] = inv[p * 4 + j]; inv[p * 4 + j] = tmpI;
      }
    }
    const diag = a[k * 4 + k];
    for (let j = 0; j < 4; j++) {
      a[k * 4 + j] /= diag;
      inv[k * 4 + j] /= diag;
    }
    for (let i = 0; i < 4; i++) {
      if (i === k) continue;
      const f = a[i * 4 + k];
      for (let j = 0; j < 4; j++) {
        a[i * 4 + j] -= f * a[k * 4 + j];
        inv[i * 4 + j] -= f * inv[k * 4 + j];
      }
    }
  }
  return inv;
}

export function createOpticalRef(patchU8, corners) {
  if (!patchU8 || patchU8.length !== N_FINE) return null;
  const basis = quadBasis(corners);
  if (basis.width < 10.0 || basis.height < 4.8) return null;
  let sum = 0;
  for (let i = 0; i < N_FINE; i++) sum += patchU8[i];
  const mean = sum / N_FINE;
  let sq = 0;
  for (let i = 0; i < N_FINE; i++) {
    const d = patchU8[i] - mean;
    sq += d * d;
  }
  const std = Math.sqrt(sq / N_FINE);
  if (std < 2.8) return null; // Insufficient contrast for optical lock.

  const tpl = new Float32Array(N_FINE);
  const invStd = 1 / std;
  for (let i = 0; i < N_FINE; i++) tpl[i] = (patchU8[i] - mean) * invStd;

  // Build 20x10 coarse zero-mean unit-std template.
  const coarse = new Float32Array(N_COARSE);
  let cSum = 0;
  for (let cj = 0; cj < CH; cj++) {
    const r0 = (cj * 2) * OPT_TW;
    const r1 = r0 + OPT_TW;
    for (let ci = 0; ci < CW; ci++) {
      const c0 = ci * 2;
      const val = 0.25 * (tpl[r0 + c0] + tpl[r0 + c0 + 1] + tpl[r1 + c0] + tpl[r1 + c0 + 1]);
      coarse[cj * CW + ci] = val;
      cSum += val;
    }
  }
  const cMean = cSum / N_COARSE;
  let cSq = 0;
  for (let i = 0; i < N_COARSE; i++) {
    coarse[i] -= cMean;
    cSq += coarse[i] * coarse[i];
  }
  const cInvNorm = 1 / Math.max(1e-6, Math.sqrt(cSq));
  for (let i = 0; i < N_COARSE; i++) coarse[i] *= cInvNorm;

  // Precompute 4-DOF Inverse-Compositional Steepest Descent images & 4x4 regularized inverse Hessian.
  const jac = new Float32Array(N_FINE * 4);
  const H = new Float64Array(16);
  for (let j = 0; j < OPT_TH; j++) {
    const ym = Math.max(0, j - 1) * OPT_TW;
    const yp = Math.min(OPT_TH - 1, j + 1) * OPT_TW;
    const yrow = j * OPT_TW;
    const yc = j + 0.5 - 0.5 * OPT_TH;
    const wy = j === 0 || j === OPT_TH - 1 ? 0.35 : 1.0;
    for (let i = 0; i < OPT_TW; i++) {
      const xm = Math.max(0, i - 1);
      const xp = Math.min(OPT_TW - 1, i + 1);
      const xc = i + 0.5 - 0.5 * OPT_TW;
      const w = (i === 0 || i === OPT_TW - 1 ? 0.35 : 1.0) * wy;
      const gx = 0.5 * (tpl[yrow + xp] - tpl[yrow + xm]) * w;
      const gy = 0.5 * (tpl[yp + i] - tpl[ym + i]) * w;
      const j0 = gx;
      const j1 = gy;
      const j2 = gx * xc;
      const j3 = gy * yc;
      const off = (yrow + i) * 4;
      jac[off] = j0;
      jac[off + 1] = j1;
      jac[off + 2] = j2;
      jac[off + 3] = j3;

      H[0] += j0 * j0; H[1] += j0 * j1; H[2] += j0 * j2; H[3] += j0 * j3;
      H[5] += j1 * j1; H[6] += j1 * j2; H[7] += j1 * j3;
      H[10] += j2 * j2; H[11] += j2 * j3;
      H[15] += j3 * j3;
    }
  }
  // Reject pure horizontal stripes (grilles/hoods lacking vertical character strokes).
  const minH0 = basis.width < 18.0 ? 5.5 : 14.0;
  if (H[0] < minH0 || H[0] < 0.09 * H[5]) return null;

  // Symmetrize & add Tikhonov damping on scale DOFs so pure 1D stripes never cause ill-conditioning.
  H[4] = H[1]; H[8] = H[2]; H[12] = H[3];
  H[9] = H[6]; H[13] = H[7];
  H[14] = H[11];
  H[0] += 1.5;
  H[5] += 1.5;
  H[10] += 120.0;
  H[15] += 80.0;

  const invH = invert4x4(H);
  if (!invH) return null;

  return {
    patchU8,
    corners: corners.map(([x, y]) => [x, y]),
    basis,
    tpl,
    coarse,
    jac,
    invH,
  };
}

function evalCoarseZncc(ref, sampleFn, w, h, tu, tv, su, sv) {
  const { mx, my, aux, auy, avx, avy, auvx, auvy } = ref.basis;
  const cx = mx + OPT_SPAN * tu * aux + OPT_SPAN * tv * avx;
  const cy = my + OPT_SPAN * tu * auy + OPT_SPAN * tv * avy;
  if (cx < 2 || cx > w - 3 || cy < 2 || cy > h - 3) return -1;

  let sum = 0;
  let idx = 0;
  for (let cj = 0; cj < CH; cj++) {
    const eta = (cj + 0.5) / CH - 0.5;
    const v = OPT_SPAN * (tv + sv * eta);
    for (let ci = 0; ci < CW; ci++) {
      const xi = (ci + 0.5) / CW - 0.5;
      const u = OPT_SPAN * (tu + su * xi);
      const px = mx + u * aux + v * avx + u * v * auvx;
      const py = my + u * auy + v * avy + u * v * auvy;
      const val = sampleFn(px, py);
      coarseBuf[idx++] = val;
      sum += val;
    }
  }
  const mean = sum / N_COARSE;
  let dot = 0, sq = 0;
  const tplC = ref.coarse;
  for (let i = 0; i < N_COARSE; i++) {
    const d = coarseBuf[i] - mean;
    sq += d * d;
    dot += d * tplC[i];
  }
  if (sq < 6.25 * N_COARSE) return -1;
  return dot / Math.sqrt(sq);
}

export function trackPlateOpticalCore(
  ref,
  sampleFn,
  w,
  h,
  initCorners,
  { searchRadiusPx = 28, searchRadiusY = null, allowScaleSearch = false, minZncc = 0.58, maxIter = 6 } = {}
) {
  if (!ref) return { ok: false, zncc: -1 };
  const basis = ref.basis;
  const initW = cornersToWarp(basis, initCorners ?? ref.corners);
  let { tu, tv, su, sv } = initW;
  const effRadX = Math.min(searchRadiusPx, Math.max(12, 1.35 * basis.width * su));
  const effRadY = Math.min(searchRadiusY ?? searchRadiusPx * 0.62, Math.max(8, 1.05 * basis.height * sv));

  // Stage 1: Coarse-to-fine ZNCC grid search with spatial proximity prior (prevents grille/hood jumps).
  if (effRadX > 1.5) {
    const radU = clamp(effRadX / Math.max(8, basis.width * OPT_SPAN), 0.02, 1.6);
    const radV = clamp(effRadY / Math.max(6, basis.height * OPT_SPAN), 0.02, 1.3);
    const stepsU = Math.min(10, Math.max(2, Math.round(effRadX / Math.max(1.1, 0.14 * basis.width))));
    const stepsV = Math.min(8, Math.max(2, Math.round(effRadY / Math.max(0.9, 0.18 * basis.height))));
    const stepU = radU / stepsU;
    const stepV = radV / stepsV;

    let bestTu = tu, bestTv = tv, bestSu = su, bestSv = sv;
    let bestRaw = evalCoarseZncc(ref, sampleFn, w, h, tu, tv, su, sv);
    let bestPenalized = bestRaw;

    for (let iv = -stepsV; iv <= stepsV; iv++) {
      const candTv = tv + iv * stepV;
      const vNormSq = (iv / stepsV) ** 2;
      for (let iu = -stepsU; iu <= stepsU; iu++) {
        if (iu === 0 && iv === 0) continue;
        const candTu = tu + iu * stepU;
        const rawSc = evalCoarseZncc(ref, sampleFn, w, h, candTu, candTv, su, sv);
        const penSc = rawSc - 0.055 * ((iu / stepsU) ** 2 + vNormSq);
        if (penSc > bestPenalized) {
          bestPenalized = penSc;
          bestRaw = rawSc;
          bestTu = candTu;
          bestTv = candTv;
        }
      }
    }

    // Sub-grid 3x3 refinement + optional isotropic scale probe for approaching vehicles.
    const fineU = 0.45 * stepU;
    const fineV = 0.45 * stepV;
    const baseTu = bestTu, baseTv = bestTv;
    for (let iv = -1; iv <= 1; iv++) {
      for (let iu = -1; iu <= 1; iu++) {
        if (iu === 0 && iv === 0) continue;
        const candTu = baseTu + iu * fineU;
        const candTv = baseTv + iv * fineV;
        const rawSc = evalCoarseZncc(ref, sampleFn, w, h, candTu, candTv, bestSu, bestSv);
        const penSc = rawSc - 0.055 * (((candTu - initW.tu) / radU) ** 2 + ((candTv - initW.tv) / radV) ** 2);
        if (penSc > bestPenalized) {
          bestPenalized = penSc;
          bestRaw = rawSc;
          bestTu = candTu;
          bestTv = candTv;
        }
      }
    }
    if (allowScaleSearch && bestRaw > 0.30) {
      for (const factor of [0.93, 1.07, 1.14]) {
        const cSu = clamp(su * factor, 0.68, 1.52);
        const cSv = clamp(sv * factor, 0.68, 1.52);
        const rawSc = evalCoarseZncc(ref, sampleFn, w, h, bestTu, bestTv, cSu, cSv);
        if (rawSc > bestRaw + 0.012) {
          bestRaw = rawSc;
          bestSu = cSu;
          bestSv = cSv;
        }
      }
    }
    if (bestRaw < 0.28) return { ok: false, zncc: bestRaw };
    tu = bestTu;
    tv = bestTv;
    su = bestSu;
    sv = bestSv;
  }

  // Stage 2: Sub-pixel 4-DOF Inverse-Compositional Gauss-Newton refinement on 40x20 grid.
  const { mx, my, aux, auy, avx, avy, auvx, auvy } = basis;
  const { tpl, jac, invH } = ref;
  let finalZncc = -1;

  for (let iter = 0; iter < maxIter; iter++) {
    let sum = 0;
    let idx = 0;
    for (let j = 0; j < OPT_TH; j++) {
      const eta = (j + 0.5) / OPT_TH - 0.5;
      const v = OPT_SPAN * (tv + sv * eta);
      for (let i = 0; i < OPT_TW; i++) {
        const xi = (i + 0.5) / OPT_TW - 0.5;
        const u = OPT_SPAN * (tu + su * xi);
        const px = mx + u * aux + v * avx + u * v * auvx;
        const py = my + u * auy + v * avy + u * v * auvy;
        const val = sampleFn(px, py);
        sampleBuf[idx++] = val;
        sum += val;
      }
    }
    const mean = sum / N_FINE;
    let sq = 0;
    for (let i = 0; i < N_FINE; i++) {
      const d = sampleBuf[i] - mean;
      sampleBuf[i] = d;
      sq += d * d;
    }
    const std = Math.sqrt(sq / N_FINE);
    if (std < 2.4) return { ok: false, zncc: -1 };
    const invStd = 1 / std;

    let b0 = 0, b1 = 0, b2 = 0, b3 = 0;
    let dot = 0;
    for (let i = 0; i < N_FINE; i++) {
      const normI = sampleBuf[i] * invStd;
      dot += normI * tpl[i];
      const err = normI - tpl[i];
      const off = i * 4;
      b0 += jac[off] * err;
      b1 += jac[off + 1] * err;
      b2 += jac[off + 2] * err;
      b3 += jac[off + 3] * err;
    }
    finalZncc = dot / N_FINE;

    const dxG = clamp(invH[0] * b0 + invH[1] * b1 + invH[2] * b2 + invH[3] * b3, -3.5, 3.5);
    const dyG = clamp(invH[4] * b0 + invH[5] * b1 + invH[6] * b2 + invH[7] * b3, -2.5, 2.5);
    const dSu = clamp(invH[8] * b0 + invH[9] * b1 + invH[10] * b2 + invH[11] * b3, -0.08, 0.08);
    const dSv = clamp(invH[12] * b0 + invH[13] * b1 + invH[14] * b2 + invH[15] * b3, -0.08, 0.08);

    const invDu = 1 / (1 + dSu);
    const invDv = 1 / (1 + dSv);
    tu -= su * invDu * (dxG / OPT_TW);
    tv -= sv * invDv * (dyG / OPT_TH);
    su = clamp(su * invDu, 0.65, 1.58);
    sv = clamp(sv * invDv, 0.65, 1.58);
    // Keep horizontal/vertical scale coherent so glare cannot squash or stretch the plate unnaturally.
    const ratio = su / sv;
    if (ratio > 1.18) su = sv * 1.18;
    else if (ratio < 0.84) sv = su / 0.84;

    if (Math.abs(dxG) < 0.025 && Math.abs(dyG) < 0.025 && Math.abs(dSu) < 0.002 && Math.abs(dSv) < 0.002) {
      break;
    }
  }

  // Evaluate post-update ZNCC once if the last step was non-trivial.
  {
    let sum = 0, idx = 0;
    for (let j = 0; j < OPT_TH; j++) {
      const eta = (j + 0.5) / OPT_TH - 0.5;
      const v = OPT_SPAN * (tv + sv * eta);
      for (let i = 0; i < OPT_TW; i++) {
        const xi = (i + 0.5) / OPT_TW - 0.5;
        const u = OPT_SPAN * (tu + su * xi);
        const px = mx + u * aux + v * avx + u * v * auvx;
        const py = my + u * auy + v * avy + u * v * auvy;
        const val = sampleFn(px, py);
        sampleBuf[idx++] = val;
        sum += val;
      }
    }
    const mean = sum / N_FINE;
    let sq = 0, dot = 0;
    for (let i = 0; i < N_FINE; i++) {
      const d = sampleBuf[i] - mean;
      sq += d * d;
      dot += d * tpl[i];
    }
    finalZncc = sq > 5.76 * N_FINE ? dot / Math.sqrt(sq * N_FINE) : -1;
  }

  if (finalZncc < minZncc) return { ok: false, zncc: finalZncc };
  const shiftX = Math.abs((tu - initW.tu) * OPT_SPAN * basis.width);
  const shiftY = Math.abs((tv - initW.tv) * OPT_SPAN * basis.height);
  if (shiftX > effRadX * 1.25 + 4 || shiftY > effRadY * 1.25 + 4) {
    return { ok: false, zncc: finalZncc };
  }

  const corners = warpToCorners(basis, tu, tv, su, sv);
  // Ensure convexity and frame containment.
  for (let i = 0; i < 4; i++) {
    const a = corners[i], b = corners[(i + 1) % 4], c = corners[(i + 2) % 4];
    if (!Number.isFinite(a[0]) || !Number.isFinite(a[1])) return { ok: false, zncc: finalZncc };
    if (a[0] < -4 || a[0] > w + 4 || a[1] < -4 || a[1] > h + 4) return { ok: false, zncc: finalZncc };
    if ((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) <= 0) return { ok: false, zncc: finalZncc };
  }
  const x0 = clamp(Math.min(corners[0][0], corners[1][0], corners[2][0], corners[3][0]), 0, w);
  const y0 = clamp(Math.min(corners[0][1], corners[1][1], corners[2][1], corners[3][1]), 0, h);
  const x1 = clamp(Math.max(corners[0][0], corners[1][0], corners[2][0], corners[3][0]), 0, w);
  const y1 = clamp(Math.max(corners[0][1], corners[1][1], corners[2][1], corners[3][1]), 0, h);
  return { ok: true, zncc: finalZncc, corners, box: [x0, y0, x1, y1], tu, tv, su, sv };
}

export function trackPlateOnRgba(ref, rgba, w, h, initCorners, opts) {
  return trackPlateOpticalCore(ref, (x, y) => sampleRgbaLumaBilinear(rgba, w, h, x, y), w, h, initCorners, opts);
}

export function trackPlateOnGray(ref, gray, gw, gh, initCorners, opts) {
  return trackPlateOpticalCore(ref, (x, y) => sampleGrayBilinear(gray, gw, gh, x, y), gw, gh, initCorners, opts);
}

// ==============================================================================
// Lightweight 16x16 Patch-NCC + 24D RGB Histogram + 32D Appearance Vehicle Tracker
// ==============================================================================
export const VEH_TPL_W = 16;
export const VEH_TPL_H = 16;
const VEH_N = VEH_TPL_W * VEH_TPL_H; // 256
const VEH_CW = 8;
const VEH_CH = 8;
const VEH_CN = VEH_CW * VEH_CH; // 64

const vehFineBuf = new Float32Array(VEH_N);
const vehCoarseBuf = new Float32Array(VEH_CN);

function sampleVehicleGridLuma(rgba, w, h, ix0, iy0, iw, ih, dx, dy, gw, gh, outBuf) {
  const stepX = iw / gw;
  const stepY = ih / gh;
  const maxX = w - 1;
  const maxY = h - 1;
  let sum = 0;
  let idx = 0;
  for (let j = 0; j < gh; j++) {
    const py = clamp(iy0 + dy + (j + 0.5) * stepY, 0, maxY);
    const y0 = py | 0;
    const y1 = y0 < maxY ? y0 + 1 : y0;
    const fy = py - y0;
    const row0 = y0 * w;
    const row1 = y1 * w;
    for (let i = 0; i < gw; i++) {
      const px = clamp(ix0 + dx + (i + 0.5) * stepX, 0, maxX);
      const x0 = px | 0;
      const x1 = x0 < maxX ? x0 + 1 : x0;
      const fx = px - x0;
      const a = (row0 + x0) * 4;
      const b = (row0 + x1) * 4;
      const d = (row1 + x0) * 4;
      const e = (row1 + x1) * 4;
      const la = 0.299 * rgba[a] + 0.587 * rgba[a + 1] + 0.114 * rgba[a + 2];
      const lb = 0.299 * rgba[b] + 0.587 * rgba[b + 1] + 0.114 * rgba[b + 2];
      const ld = 0.299 * rgba[d] + 0.587 * rgba[d + 1] + 0.114 * rgba[d + 2];
      const le = 0.299 * rgba[e] + 0.587 * rgba[e + 1] + 0.114 * rgba[e + 2];
      const val =
        la * (1 - fx) * (1 - fy) +
        lb * fx * (1 - fy) +
        ld * (1 - fx) * fy +
        le * fx * fy;
      outBuf[idx++] = val;
      sum += val;
    }
  }
  return sum / (gw * gh);
}

export function extractVehicleAppearanceRgba(rgba, w, h, box) {
  const bw = box[2] - box[0];
  const bh = box[3] - box[1];
  if (bw < 10 || bh < 10) return null;
  // Interior 80% crop avoids background road/sky bleeding into the vehicle descriptor.
  const ix0 = box[0] + 0.10 * bw;
  const iy0 = box[1] + 0.10 * bh;
  const iw = Math.max(4, 0.80 * bw);
  const ih = Math.max(4, 0.80 * bh);
  const stepX = iw / VEH_TPL_W;
  const stepY = ih / VEH_TPL_H;
  const maxX = w - 1;
  const maxY = h - 1;

  const rawLuma = new Float32Array(VEH_N);
  const histCounts = new Float32Array(24);
  const spatial8 = new Float32Array(8);
  let sumY = 0;
  let idx = 0;
  for (let j = 0; j < VEH_TPL_H; j++) {
    const py = clamp(iy0 + (j + 0.5) * stepY, 0, maxY);
    const y0 = py | 0;
    const row = y0 * w;
    const bandY = j >> 2; // 0..3
    for (let i = 0; i < VEH_TPL_W; i++) {
      const px = clamp(ix0 + (i + 0.5) * stepX, 0, maxX);
      const x0 = px | 0;
      const p = (row + x0) * 4;
      const r = rgba[p];
      const g = rgba[p + 1];
      const b = rgba[p + 2];
      const luma = 0.299 * r + 0.587 * g + 0.114 * b;
      rawLuma[idx++] = luma;
      sumY += luma;
      histCounts[Math.min(7, r >> 5)] += 1;
      histCounts[8 + Math.min(7, g >> 5)] += 1;
      histCounts[16 + Math.min(7, b >> 5)] += 1;
      const halfX = i >> 3; // 0..1
      spatial8[bandY * 2 + halfX] += luma;
    }
  }

  // 24D RGB Bhattacharyya-normalized histogram (dot product in [0, 1]).
  const colorHist = new Float32Array(24);
  const inv3N = 1 / (3 * VEH_N);
  for (let k = 0; k < 24; k++) {
    colorHist[k] = Math.sqrt(histCounts[k] * inv3N);
  }

  // 32D L2-normalized appearance descriptor = 24D RGB hist + 8D spatial 4x2 luma profile.
  const app32 = new Float32Array(32);
  let appSq = 0;
  for (let k = 0; k < 24; k++) {
    app32[k] = colorHist[k];
    appSq += app32[k] * app32[k];
  }
  const spInv = 1 / (32 * 255);
  let spMean = 0;
  for (let k = 0; k < 8; k++) {
    spatial8[k] *= spInv;
    spMean += spatial8[k];
  }
  spMean /= 8;
  for (let k = 0; k < 8; k++) {
    const v = 0.85 * (spatial8[k] - spMean);
    app32[24 + k] = v;
    appSq += v * v;
  }
  const appInv = appSq > 1e-12 ? 1 / Math.sqrt(appSq) : 0;
  for (let k = 0; k < 32; k++) app32[k] *= appInv;

  // 16x16 zero-mean unit-L2-norm bilinear grayscale template matching trackVehiclePatchNccRgba exactly.
  const mu = sampleVehicleGridLuma(rgba, w, h, ix0, iy0, iw, ih, 0, 0, VEH_TPL_W, VEH_TPL_H, rawLuma);
  let sq = 0;
  for (let k = 0; k < VEH_N; k++) {
    const d = rawLuma[k] - mu;
    rawLuma[k] = d;
    sq += d * d;
  }
  const tplStd = Math.sqrt(sq / VEH_N);
  if (tplStd < 1.5) {
    return { tpl: null, coarse: null, tplStd, colorHist, app32 };
  }
  const invNorm = 1 / Math.sqrt(sq);
  const tpl = new Float32Array(VEH_N);
  for (let k = 0; k < VEH_N; k++) tpl[k] = rawLuma[k] * invNorm;

  // 8x8 bilinear coarse template matching evalCoarse sampling coordinates identically.
  const coarse = new Float32Array(VEH_CN);
  const cMu = sampleVehicleGridLuma(rgba, w, h, ix0, iy0, iw, ih, 0, 0, VEH_CW, VEH_CH, coarse);
  let cSq = 0;
  for (let k = 0; k < VEH_CN; k++) {
    coarse[k] -= cMu;
    cSq += coarse[k] * coarse[k];
  }
  const cInv = cSq > 1e-12 ? 1 / Math.sqrt(cSq) : 0;
  for (let k = 0; k < VEH_CN; k++) coarse[k] *= cInv;

  return { tpl, coarse, tplStd, colorHist, app32 };
}

export function trackVehiclePatchNccRgba(
  vehRef,
  rgba,
  w,
  h,
  predBox,
  { maxShiftPx = 24, minNcc = 0.42 } = {}
) {
  if (!vehRef?.tpl || !vehRef?.coarse) {
    return { ok: false, ncc: 0, dx: 0, dy: 0, box: predBox };
  }
  const bw = predBox[2] - predBox[0];
  const bh = predBox[3] - predBox[1];
  if (bw < 10 || bh < 10) {
    return { ok: false, ncc: 0, dx: 0, dy: 0, box: predBox };
  }
  const ix0 = predBox[0] + 0.10 * bw;
  const iy0 = predBox[1] + 0.10 * bh;
  const iw = Math.max(4, 0.80 * bw);
  const ih = Math.max(4, 0.80 * bh);
  const tpl = vehRef.tpl;
  const coarse = vehRef.coarse;
  const tplStd = vehRef.tplStd ?? 0;

  const evalCoarse = (dx, dy) => {
    const cx = predBox[0] + 0.5 * bw + dx;
    const cy = predBox[1] + 0.5 * bh + dy;
    if (cx < -0.35 * bw || cx > w + 0.35 * bw || cy < -0.35 * bh || cy > h + 0.35 * bh) return -1;
    const mu = sampleVehicleGridLuma(rgba, w, h, ix0, iy0, iw, ih, dx, dy, VEH_CW, VEH_CH, vehCoarseBuf);
    let sq = 0, dot = 0;
    for (let k = 0; k < VEH_CN; k++) {
      const d = vehCoarseBuf[k] - mu;
      sq += d * d;
      dot += d * coarse[k];
    }
    if (sq < 2.25 * VEH_CN) return -1;
    return dot / Math.sqrt(sq);
  };

  const evalFine = (dx, dy) => {
    const cx = predBox[0] + 0.5 * bw + dx;
    const cy = predBox[1] + 0.5 * bh + dy;
    if (cx < -0.35 * bw || cx > w + 0.35 * bw || cy < -0.35 * bh || cy > h + 0.35 * bh) return 0;
    const mu = sampleVehicleGridLuma(rgba, w, h, ix0, iy0, iw, ih, dx, dy, VEH_TPL_W, VEH_TPL_H, vehFineBuf);
    let sq = 0, dot = 0;
    for (let k = 0; k < VEH_N; k++) {
      const d = vehFineBuf[k] - mu;
      sq += d * d;
      dot += d * tpl[k];
    }
    const candStd = Math.sqrt(sq / VEH_N);
    if (candStd < 1.8) return 0;
    let ncc = dot / Math.sqrt(sq);
    // Prevent high-contrast vehicle templates from matching low-contrast flat asphalt noise.
    if (tplStd > 10.0 && candStd < 0.20 * tplStd) {
      ncc *= candStd / (0.20 * tplStd);
    }
    return ncc;
  };

  const rad = Math.max(4, Math.round(maxShiftPx));
  const invRadSq = 1 / (rad * rad);

  // Stage 1: Coarse 8px grid across [-rad, +rad] on 8x8 pooled template.
  let bestDx = 0, bestDy = 0;
  let bestPen = evalCoarse(0, 0);
  const step1 = Math.max(4, Math.round(rad / 3));
  for (let dy = -rad; dy <= rad; dy += step1) {
    for (let dx = -rad; dx <= rad; dx += step1) {
      if (dx === 0 && dy === 0) continue;
      const sc = evalCoarse(dx, dy);
      const pen = sc - 0.015 * (dx * dx + dy * dy) * invRadSq;
      if (pen > bestPen) {
        bestPen = pen;
        bestDx = dx;
        bestDy = dy;
      }
    }
  }

  // Stage 2: Medium 3px refinement on 8x8 pooled template around coarse peak.
  const b1x = bestDx, b1y = bestDy;
  const step2 = Math.max(2, Math.round(step1 / 2.5));
  for (let iy = -2; iy <= 2; iy++) {
    const dy = clamp(b1y + iy * step2, -rad, rad);
    for (let ix = -2; ix <= 2; ix++) {
      if (ix === 0 && iy === 0) continue;
      const dx = clamp(b1x + ix * step2, -rad, rad);
      const sc = evalCoarse(dx, dy);
      const pen = sc - 0.015 * (dx * dx + dy * dy) * invRadSq;
      if (pen > bestPen) {
        bestPen = pen;
        bestDx = dx;
        bestDy = dy;
      }
    }
  }

  // Stage 3: Exact 16x16 fine search (4px -> 2px -> 1px step).
  let bestFine = evalFine(bestDx, bestDy);
  let bestFinePen = bestFine - 0.010 * (bestDx * bestDx + bestDy * bestDy) * invRadSq;
  for (const step of [4, 2, 1]) {
    const cx0 = bestDx, cy0 = bestDy;
    for (let iy = -1; iy <= 1; iy++) {
      const dy = clamp(cy0 + iy * step, -rad, rad);
      for (let ix = -1; ix <= 1; ix++) {
        if (ix === 0 && iy === 0) continue;
        const dx = clamp(cx0 + ix * step, -rad, rad);
        const sc = evalFine(dx, dy);
        const pen = sc - 0.010 * (dx * dx + dy * dy) * invRadSq;
        if (pen > bestFinePen) {
          bestFinePen = pen;
          bestFine = sc;
          bestDx = dx;
          bestDy = dy;
        }
      }
    }
  }

  const shiftedBox = [
    clamp(predBox[0] + bestDx, 0, w),
    clamp(predBox[1] + bestDy, 0, h),
    clamp(predBox[2] + bestDx, 0, w),
    clamp(predBox[3] + bestDy, 0, h),
  ];
  return {
    ok: bestFine >= minNcc,
    ncc: bestFine,
    dx: bestDx,
    dy: bestDy,
    box: shiftedBox,
  };
}
