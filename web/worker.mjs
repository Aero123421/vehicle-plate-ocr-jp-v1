// Apache-2.0 · vehicle-plate-ocr-jp-v1 v0.6.5.
// JP7029732B2 design-around: no character projection while COLLECTING.
import * as ort from './vendor/ort.wasm.min.mjs';
import { clamp, sigmoid, unpackGrammar, terminalDecision } from './terminal.mjs';

// Shared by the browser WebWorker and local Node parity checks.
export async function createPipeline({
  runtime = ort,
  readJSON,
  progress = () => {},
  autoSealFrames = 0,
  autoSealMinSec = 0,
  autoSealMinPrecision = 0.35,
  autoExitSec = 0.55,
  requireFullFieldsForEarlySeal = false,
  discardShortUnreadable = false,
} = {}) {
  const json = readJSON ?? (async path => {
    const response = await fetch(new URL(path, import.meta.url));
    if (!response.ok) throw Error(`${path}: HTTP ${response.status}`);
    return response.json();
  });
  runtime.env.wasm.wasmPaths = new URL('./vendor/', import.meta.url).href;
  const hardware = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 2;
  runtime.env.wasm.simd = true;
  runtime.env.wasm.numThreads =
    (typeof self !== 'undefined' && self.crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined')
      ? Math.min(4, Math.max(2, Math.floor(hardware / 2)))
      : 1;

  const float = (data, dims) =>
    new runtime.Tensor('float32', data instanceof Float32Array ? data : new Float32Array(data), dims);
  const zero = dims => float(new Float32Array(dims.reduce((a, b) => a * b, 1)), dims);
  const integer = value => new runtime.Tensor('int64', BigInt64Array.of(BigInt(value)), [1]);
  const mean = a => {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i];
    return s / a.length;
  };

  const sessions = {};
  const names = ['temporal_det_nano', 'vehicle_cutout_nano', 'stage2', 'frame_encoder', 'memory_step', 'memory_readout'];
  let grammar, policy;
  try {
    async function createSessions() {
      for (const name of names) {
        progress(`INT8 グラフを準備中: ${name}`);
        sessions[name] = await runtime.InferenceSession.create(
          new URL(`models/${name}.int8.onnx`, import.meta.url).href,
          { executionProviders: ['wasm'], graphOptimizationLevel: 'all' }
        );
      }
    }
    try {
      await createSessions();
    } catch (error) {
      if (runtime.env.wasm.numThreads <= 1) throw error;
      await Promise.allSettled(Object.values(sessions).map(s => s.release()));
      for (const name of names) delete sessions[name];
      runtime.env.wasm.numThreads = 1;
      await createSessions();
    }
    grammar = await unpackGrammar(await json('models/grammar_tables.json'));
    policy = await json('models/gen6.json');
  } catch (error) {
    await Promise.all(Object.values(sessions).map(s => s.release()));
    throw error;
  }

  const baseHeartbeat = policy?.pipeline?.heartbeat_interval ?? 6;
  const maxHeartbeat = policy?.pipeline?.max_heartbeat_interval ?? 12;
  const plateThresh = 0.22;
  const weakPlateThresh = 0.08;
  const vehThresh = 0.54;
  const minIsPlateProb = 0.34;
  const plateLogitMin = Math.log(plateThresh / (1 - plateThresh));
  const weakPlateLogitMin = Math.log(weakPlateThresh / (1 - weakPlateThresh));
  const vehLogitMin = Math.log(vehThresh / (1 - vehThresh));

  const LB_W = 768, LB_H = 448;
  const ZOOM_W = 384, ZOOM_H = 256;
  const letterboxBuf = new Float32Array(3 * LB_W * LB_H);
  const fovea1Buf = new Float32Array(3 * 96 * 160);
  const surround1Buf = new Float32Array(3 * 64 * 96);
  const policy1Buf = new BigInt64Array(1);
  const zoomTileBuf = new Uint8ClampedArray(ZOOM_W * ZOOM_H * 4);
  const zeroH3 = zero([1, 36, 56, 96]);
  const zeroH4 = zero([1, 36, 28, 48]);
  const cropBuf = new Float32Array(3 * 80 * 160);
  const lumBuf = new Float32Array(12800);
  const chromaBuf = new Float32Array(12800);
  const SCOUT_W = 48, SCOUT_H = 28;
  const scoutPrev = new Float32Array(SCOUT_W * SCOUT_H);
  const scoutCur = new Float32Array(SCOUT_W * SCOUT_H);
  const scoutMask = new Uint8Array(SCOUT_W * SCOUT_H);

  let hasScoutPrev = false;
  let emittedResults = [];
  let nextPlateId = 1;
  let nextVehicleId = 1;
  let tracks = [];
  let vtracks = [];
  let frameIndex = 0;
  let h3, h4;
  let frame, wOrig, hOrig;
  let lastFull = -Infinity;
  let lastTime = -Infinity;
  let phase = 'IDLE';
  let fullDetections = 0;
  let fastFrames = 0;
  let zoomCooldown = 0;
  let prevLumaMean = null;

  function configure(options = {}) {
    if (options.autoSealFrames !== undefined) autoSealFrames = options.autoSealFrames;
    if (options.autoSealMinSec !== undefined) autoSealMinSec = options.autoSealMinSec;
    if (options.autoSealMinPrecision !== undefined) autoSealMinPrecision = options.autoSealMinPrecision;
    if (options.autoExitSec !== undefined) autoExitSec = options.autoExitSec;
    if (options.requireFullFieldsForEarlySeal !== undefined) {
      requireFullFieldsForEarlySeal = Boolean(options.requireFullFieldsForEarlySeal);
    }
    if (options.discardShortUnreadable !== undefined) {
      discardShortUnreadable = Boolean(options.discardShortUnreadable);
    }
  }

  function minFieldPrecision(memPData) {
    const slotMean = s => {
      let sum = 0;
      const base = s * 112;
      for (let i = 0; i < 112; i++) sum += Math.max(0, memPData[base + i]);
      return sum / 112;
    };
    const pReg = (slotMean(0) + slotMean(1) + slotMean(2) + slotMean(3)) / 4;
    const pCls = (slotMean(4) + slotMean(5) + slotMean(6)) / 3;
    const pKana = slotMean(7);
    const pSer = (slotMean(8) + slotMean(9) + slotMean(10) + slotMean(11)) / 4;
    return Math.min(pReg, pCls, pKana, pSer);
  }

  function reset(options = {}) {
    if (phase === 'COLLECTING' || phase === 'SEALED') throw Error('Finish the current collection first');
    configure(options);
    emittedResults = [];
    nextPlateId = 1;
    nextVehicleId = 1;
    tracks = [];
    vtracks = [];
    frameIndex = 0;
    lastFull = -Infinity;
    lastTime = -Infinity;
    hasScoutPrev = false;
    prevLumaMean = null;
    fullDetections = 0;
    fastFrames = 0;
    zoomCooldown = 0;
    h3 = zero([1, 36, 56, 96]);
    h4 = zero([1, 36, 28, 48]);
    phase = 'COLLECTING';
  }

  function iou(a, b) {
    const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const inter = ix * iy;
    if (inter <= 0) return 0;
    const areaA = Math.max(1, (a[2] - a[0]) * (a[3] - a[1]));
    const areaB = Math.max(1, (b[2] - b[0]) * (b[3] - b[1]));
    return inter / Math.max(1, areaA + areaB - inter);
  }

  function nms(items, limit = 16, threshold = 0.45) {
    const sorted = items.slice().sort((a, b) => b.score - a.score);
    const result = [];
    for (const item of sorted) {
      if (result.every(other => iou(other.box, item.box) < threshold)) result.push(item);
      if (result.length >= limit) break;
    }
    return result;
  }

  function normalizeVec(arr) {
    const out = new Float32Array(arr.length);
    let sq = 0;
    for (let i = 0; i < arr.length; i++) sq += arr[i] * arr[i];
    const inv = sq > 1e-12 ? 1 / Math.sqrt(sq) : 0;
    for (let i = 0; i < arr.length; i++) out[i] = arr[i] * inv;
    return out;
  }

  function dotVec(a, b) {
    if (!a || !b || a.length !== b.length) return 0;
    let s = 0;
    for (let i = 0; i < a.length; i++) s += a[i] * b[i];
    return s;
  }

  function estimateBorderColor(box) {
    const x0 = Math.max(0, Math.min(wOrig - 1, Math.floor(box[0])));
    const y0 = Math.max(0, Math.min(hOrig - 1, Math.floor(box[1])));
    const x1 = Math.max(x0 + 1, Math.min(wOrig, Math.ceil(box[2])));
    const y1 = Math.max(y0 + 1, Math.min(hOrig, Math.ceil(box[3])));
    let r = 0, g = 0, b = 0, count = 0;
    const stepX = Math.max(1, Math.floor((x1 - x0) / 8));
    const stepY = Math.max(1, Math.floor((y1 - y0) / 6));
    for (let x = x0; x < x1; x += stepX) {
      for (const y of [y0, y1 - 1]) {
        const idx = (y * wOrig + x) * 4;
        r += frame[idx]; g += frame[idx + 1]; b += frame[idx + 2]; count++;
      }
    }
    for (let y = y0; y < y1; y += stepY) {
      for (const x of [x0, x1 - 1]) {
        const idx = (y * wOrig + x) * 4;
        r += frame[idx]; g += frame[idx + 1]; b += frame[idx + 2]; count++;
      }
    }
    if (!count) return [40 / 255, 44 / 255, 50 / 255];
    return [
      clamp(r / count, 24, 136) / 255,
      clamp(g / count, 24, 136) / 255,
      clamp(b / count, 24, 136) / 255,
    ];
  }

  function sampleBilinear(src, sw, sh, nx, ny, border) {
    if (nx < 0 || nx > sw - 1 || ny < 0 || ny > sh - 1) return border;
    const x0 = nx | 0;
    const y0 = ny | 0;
    const x1 = x0 + 1 < sw ? x0 + 1 : x0;
    const y1 = y0 + 1 < sh ? y0 + 1 : y0;
    const dx = nx - x0;
    const dy = ny - y0;
    const wa = (1 - dx) * (1 - dy) / 255;
    const wb = dx * (1 - dy) / 255;
    const wd = (1 - dx) * dy / 255;
    const we = dx * dy / 255;
    const a = (y0 * sw + x0) * 4;
    const b = (y0 * sw + x1) * 4;
    const d = (y1 * sw + x0) * 4;
    const e = (y1 * sw + x1) * 4;
    return [
      src[a] * wa + src[b] * wb + src[d] * wd + src[e] * we,
      src[a + 1] * wa + src[b + 1] * wb + src[d + 1] * wd + src[e + 1] * we,
      src[a + 2] * wa + src[b + 2] * wb + src[d + 2] * wd + src[e + 2] * we,
    ];
  }

  function fillPatch(win, w, h, out, offset, border) {
    const [rx0, ry0, rw, rh] = win;
    const stride = w * h;
    const ratio = Math.max(rw / w, rh / h);
    const ss = ratio > 1.35 ? 2 : 1;
    if (ss === 1) {
      for (let j = 0; j < h; j++) {
        const ny = ry0 + (j + 0.5) * rh / h - 0.5;
        for (let i = 0; i < w; i++) {
          const nx = rx0 + (i + 0.5) * rw / w - 0.5;
          const idx = offset + j * w + i;
          if (nx < 0 || nx > wOrig - 1 || ny < 0 || ny > hOrig - 1) {
            out[idx] = border[0];
            out[idx + stride] = border[1];
            out[idx + 2 * stride] = border[2];
          } else {
            const x0 = nx | 0, y0 = ny | 0;
            const x1 = x0 + 1 < wOrig ? x0 + 1 : x0;
            const y1 = y0 + 1 < hOrig ? y0 + 1 : y0;
            const dx = nx - x0, dy = ny - y0;
            const wa = (1 - dx) * (1 - dy) / 255;
            const wb = dx * (1 - dy) / 255;
            const wd = (1 - dx) * dy / 255;
            const we = dx * dy / 255;
            const a = (y0 * wOrig + x0) * 4;
            const b = (y0 * wOrig + x1) * 4;
            const d = (y1 * wOrig + x0) * 4;
            const e = (y1 * wOrig + x1) * 4;
            out[idx] = frame[a] * wa + frame[b] * wb + frame[d] * wd + frame[e] * we;
            out[idx + stride] = frame[a + 1] * wa + frame[b + 1] * wb + frame[d + 1] * wd + frame[e + 1] * we;
            out[idx + 2 * stride] = frame[a + 2] * wa + frame[b + 2] * wb + frame[d + 2] * wd + frame[e + 2] * we;
          }
        }
      }
      return;
    }
    // 2x2 box-integrated supersampling when minifying (approximates cv2.INTER_AREA), zero-allocation.
    const br = border[0], bg = border[1], bb = border[2];
    const wm1 = wOrig - 1, hm1 = hOrig - 1;
    for (let j = 0; j < h; j++) {
      const nyA = ry0 + (j + 0.25) * rh / h - 0.5;
      const nyB = ry0 + (j + 0.75) * rh / h - 0.5;
      for (let i = 0; i < w; i++) {
        const nxA = rx0 + (i + 0.25) * rw / w - 0.5;
        const nxB = rx0 + (i + 0.75) * rw / w - 0.5;
        let sr = 0, sg = 0, sb = 0;
        for (let k = 0; k < 4; k++) {
          const nx = (k & 1) ? nxB : nxA;
          const ny = (k & 2) ? nyB : nyA;
          if (nx < 0 || nx > wm1 || ny < 0 || ny > hm1) {
            sr += br; sg += bg; sb += bb;
          } else {
            const x0 = nx | 0, y0 = ny | 0;
            const x1 = x0 < wm1 ? x0 + 1 : x0, y1 = y0 < hm1 ? y0 + 1 : y0;
            const dx = nx - x0, dy = ny - y0;
            const wa = (1 - dx) * (1 - dy) / 255, wb = dx * (1 - dy) / 255;
            const wd = (1 - dx) * dy / 255, we = dx * dy / 255;
            const a = (y0 * wOrig + x0) * 4, b = (y0 * wOrig + x1) * 4;
            const d = (y1 * wOrig + x0) * 4, e = (y1 * wOrig + x1) * 4;
            sr += frame[a] * wa + frame[b] * wb + frame[d] * wd + frame[e] * we;
            sg += frame[a + 1] * wa + frame[b + 1] * wb + frame[d + 1] * wd + frame[e + 1] * we;
            sb += frame[a + 2] * wa + frame[b + 2] * wb + frame[d + 2] * wd + frame[e + 2] * we;
          }
        }
        const idx = offset + j * w + i;
        out[idx] = 0.25 * sr;
        out[idx + stride] = 0.25 * sg;
        out[idx + 2 * stride] = 0.25 * sb;
      }
    }
  }

  function letterbox() {
    const scale = Math.min(LB_W / wOrig, LB_H / hOrig);
    const nw = Math.max(1, Math.round(wOrig * scale));
    const nh = Math.max(1, Math.round(hOrig * scale));
    const ox = Math.floor((LB_W - nw) / 2);
    const oy = Math.floor((LB_H - nh) / 2);
    const data = letterboxBuf;
    const stride = LB_W * LB_H;
    if (wOrig === LB_W && hOrig === LB_H) {
      for (let i = 0; i < stride; i++) {
        const k = i * 4;
        data[i] = frame[k] / 255;
        data[i + stride] = frame[k + 1] / 255;
        data[i + 2 * stride] = frame[k + 2] / 255;
      }
      return { images: float(data, [1, 3, LB_H, LB_W]), scale, ox, oy };
    }
    data.fill(114 / 255);
    const invScale = 1 / scale;
    const useArea = scale < 0.92;
    for (let y = 0; y < nh; y++) {
      const dy = oy + y;
      const sy0 = y * invScale;
      const sy1 = (y + 1) * invScale;
      for (let x = 0; x < nw; x++) {
        const dx = ox + x;
        const idx = dy * LB_W + dx;
        if (useArea) {
          const sx0 = x * invScale;
          const sx1 = (x + 1) * invScale;
          const ix0 = Math.max(0, Math.floor(sx0));
          const ix1 = Math.min(wOrig - 1, Math.floor(sx1 - 1e-4));
          const iy0 = Math.max(0, Math.floor(sy0));
          const iy1 = Math.min(hOrig - 1, Math.floor(sy1 - 1e-4));
          let r = 0, g = 0, b = 0, cnt = 0;
          for (let yy = iy0; yy <= iy1; yy++) {
            const row = yy * wOrig;
            for (let xx = ix0; xx <= ix1; xx++) {
              const p = (row + xx) * 4;
              r += frame[p]; g += frame[p + 1]; b += frame[p + 2]; cnt++;
            }
          }
          const inv = 1 / (255 * Math.max(1, cnt));
          data[idx] = r * inv;
          data[idx + stride] = g * inv;
          data[idx + 2 * stride] = b * inv;
        } else {
          const nx = clamp((x + 0.5) * invScale - 0.5, 0, wOrig - 1);
          const ny = clamp((y + 0.5) * invScale - 0.5, 0, hOrig - 1);
          const x0 = nx | 0, y0 = ny | 0;
          const x1 = x0 + 1 < wOrig ? x0 + 1 : x0;
          const y1 = y0 + 1 < hOrig ? y0 + 1 : y0;
          const fx = nx - x0, fy = ny - y0;
          const wa = (1 - fx) * (1 - fy) / 255;
          const wb = fx * (1 - fy) / 255;
          const wd = (1 - fx) * fy / 255;
          const we = fx * fy / 255;
          const a = (y0 * wOrig + x0) * 4;
          const b = (y0 * wOrig + x1) * 4;
          const d = (y1 * wOrig + x0) * 4;
          const e = (y1 * wOrig + x1) * 4;
          data[idx] = frame[a] * wa + frame[b] * wb + frame[d] * wd + frame[e] * we;
          data[idx + stride] = frame[a + 1] * wa + frame[b + 1] * wb + frame[d + 1] * wd + frame[e + 1] * we;
          data[idx + 2 * stride] = frame[a + 2] * wa + frame[b + 2] * wb + frame[d + 2] * wd + frame[e + 2] * we;
        }
      }
    }
    return { images: float(data, [1, 3, LB_H, LB_W]), scale, ox, oy };
  }

  function windows(box, gamma = 1.0) {
    const [a, b, c, d] = box;
    const bw = Math.max(5, c - a), bh = Math.max(3, d - b);
    const cx = 0.5 * (a + c), cy = 0.5 * (b + d);
    const eff = 1.7 * gamma;
    const rw = Math.max(bw * eff, bh * eff * 160 / 96, 18 * gamma);
    const rh = rw * 96 / 160;
    const sw = rw * 4.5 / 1.7;
    const sh = sw * 64 / 96;
    return [
      [cx - rw / 2, cy - rh / 2, rw, rh],
      [cx - sw / 2, cy - sh / 2, sw, sh],
    ];
  }

  function projective(corners) {
    const matrix = [], rhs = [];
    for (let i = 0; i < 4; i++) {
      const u = [0, 1, 1, 0][i], v = [0, 0, 1, 1][i], [x, y] = corners[i];
      matrix.push([u, v, 1, 0, 0, 0, -u * x, -v * x]); rhs.push(x);
      matrix.push([0, 0, 0, u, v, 1, -u * y, -v * y]); rhs.push(y);
    }
    for (let k = 0; k < 8; k++) {
      let p = k;
      for (let i = k + 1; i < 8; i++) if (Math.abs(matrix[i][k]) > Math.abs(matrix[p][k])) p = i;
      [matrix[k], matrix[p]] = [matrix[p], matrix[k]];
      [rhs[k], rhs[p]] = [rhs[p], rhs[k]];
      const d = matrix[k][k];
      if (Math.abs(d) < 1e-10) throw Error('Degenerate plate homography');
      for (let j = k; j < 8; j++) matrix[k][j] /= d;
      rhs[k] /= d;
      for (let i = 0; i < 8; i++) {
        if (i === k) continue;
        const f = matrix[i][k];
        for (let j = k; j < 8; j++) matrix[i][j] -= f * matrix[k][j];
        rhs[i] -= f * rhs[k];
      }
    }
    const wt = Math.hypot(corners[1][0] - corners[0][0], corners[1][1] - corners[0][1]);
    const wb = Math.hypot(corners[2][0] - corners[3][0], corners[2][1] - corners[3][1]);
    const hl = Math.hypot(corners[3][0] - corners[0][0], corners[3][1] - corners[0][1]);
    const hr = Math.hypot(corners[2][0] - corners[1][0], corners[2][1] - corners[1][1]);
    const maxRatio = Math.max(0.5 * (wt + wb) / (160 * 0.9), 0.5 * (hl + hr) / (80 * 0.9));
    const ss = maxRatio > 1.25 ? 2 : 1;
    const data = cropBuf;
    const stride = 12800;
    const neutral = [40 / 255, 44 / 255, 50 / 255];

    const sampleUV = (u, v) => {
      const z = rhs[6] * u + rhs[7] * v + 1;
      const nx = (rhs[0] * u + rhs[1] * v + rhs[2]) / z;
      const ny = (rhs[3] * u + rhs[4] * v + rhs[5]) / z;
      return sampleBilinear(frame, wOrig, hOrig, clamp(nx, 0, wOrig - 1), clamp(ny, 0, hOrig - 1), neutral);
    };

    for (let y = 0; y < 80; y++) {
      for (let x = 0; x < 160; x++) {
        const index = y * 160 + x;
        if (ss === 1) {
          const u = ((x + 0.5) / 160 - 0.05) / 0.9;
          const v = ((y + 0.5) / 80 - 0.05) / 0.9;
          const z = rhs[6] * u + rhs[7] * v + 1;
          const nx = (rhs[0] * u + rhs[1] * v + rhs[2]) / z;
          const ny = (rhs[3] * u + rhs[4] * v + rhs[5]) / z;
          if (nx < 0 || nx > wOrig - 1 || ny < 0 || ny > hOrig - 1) {
            const c = sampleBilinear(frame, wOrig, hOrig, clamp(nx, 0, wOrig - 1), clamp(ny, 0, hOrig - 1), neutral);
            data[index] = c[0]; data[index + stride] = c[1]; data[index + 2 * stride] = c[2];
          } else {
            const x0 = nx | 0, y0 = ny | 0;
            const x1 = x0 + 1 < wOrig ? x0 + 1 : x0;
            const y1 = y0 + 1 < hOrig ? y0 + 1 : y0;
            const dx = nx - x0, dy = ny - y0;
            const wa = (1 - dx) * (1 - dy) / 255;
            const wb2 = dx * (1 - dy) / 255;
            const wd = (1 - dx) * dy / 255;
            const we = dx * dy / 255;
            const a = (y0 * wOrig + x0) * 4;
            const b = (y0 * wOrig + x1) * 4;
            const d = (y1 * wOrig + x0) * 4;
            const e = (y1 * wOrig + x1) * 4;
            data[index] = frame[a] * wa + frame[b] * wb2 + frame[d] * wd + frame[e] * we;
            data[index + stride] = frame[a + 1] * wa + frame[b + 1] * wb2 + frame[d + 1] * wd + frame[e + 1] * we;
            data[index + 2 * stride] = frame[a + 2] * wa + frame[b + 2] * wb2 + frame[d + 2] * wd + frame[e + 2] * we;
          }
        } else {
          const uA = ((x + 0.25) / 160 - 0.05) / 0.9, uB = ((x + 0.75) / 160 - 0.05) / 0.9;
          const vA = ((y + 0.25) / 80 - 0.05) / 0.9, vB = ((y + 0.75) / 80 - 0.05) / 0.9;
          const c00 = sampleUV(uA, vA), c10 = sampleUV(uB, vA);
          const c01 = sampleUV(uA, vB), c11 = sampleUV(uB, vB);
          data[index] = 0.25 * (c00[0] + c10[0] + c01[0] + c11[0]);
          data[index + stride] = 0.25 * (c00[1] + c10[1] + c01[1] + c11[1]);
          data[index + 2 * stride] = 0.25 * (c00[2] + c10[2] + c01[2] + c11[2]);
        }
      }
    }
    return float(data, [1, 3, 80, 160]);
  }

  function geometry(q, crop, previous) {
    const norm = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
    const wt = norm(q[0], q[1]), wr = norm(q[1], q[2]), wb = norm(q[2], q[3]), wl = norm(q[3], q[0]);
    const w = Math.max(2, (wt + wb) / 2), h = Math.max(1, (wl + wr) / 2), pixels = 12800;
    const lum = lumBuf, chroma = chromaBuf;
    let brightness = 0, highlights = 0, lapSum = 0, lapSquare = 0;
    for (let i = 0; i < pixels; i++) {
      const r = crop.data[i], g = crop.data[pixels + i], b = crop.data[2 * pixels + i];
      const v = 0.299 * r + 0.587 * g + 0.114 * b;
      lum[i] = v; brightness += v;
      if (v > 0.96) highlights++;
      chroma[i] = Math.max(r, g, b) - Math.min(r, g, b);
    }
    for (let y = 0; y < 80; y++) {
      const ym = (y === 0 ? 1 : y - 1) * 160, yp = (y === 79 ? 78 : y + 1) * 160;
      const yrow = y * 160;
      for (let x = 0; x < 160; x++) {
        const xm = x === 0 ? 1 : x - 1, xp = x === 159 ? 158 : x + 1;
        const v = 2 * (lum[ym + xm] + lum[ym + xp] + lum[yp + xm] + lum[yp + xp]) - 8 * lum[yrow + x];
        lapSum += v; lapSquare += v * v;
      }
    }
    const variance = Math.max(0, lapSquare / pixels - (lapSum / pixels) ** 2);
    chroma.sort();
    let chromaSum = 0;
    for (let i = pixels - 1920; i < pixels; i++) chromaSum += chroma[i];
    let displacement = 0;
    if (previous) for (let i = 0; i < 4; i++) for (let k = 0; k < 2; k++) displacement += (q[i][k] - previous[i][k]) ** 2;
    const delta = previous ? clamp(Math.sqrt(displacement) / (2 * w)) : 0;
    return float([
      clamp(0.5 * Math.log2(w / 64), -1, 1),
      clamp(0.5 * Math.log2(h / 32), -1, 1),
      clamp(w / (2 * h), 0.25, 2) - 1,
      Math.tanh(2.5 * Math.log((wl + 0.001) / (wr + 0.001))),
      Math.tanh(2.5 * Math.log((wb + 0.001) / (wt + 0.001))),
      delta,
      clamp(0.25 * Math.log1p(100 * variance)),
      brightness / pixels,
      chromaSum / 1920,
      highlights / pixels,
    ], [1, 10]);
  }

  function isRedLampCrop(crop) {
    // Reject pure red brake-light / reflector false positives (valid JP plates are white, yellow, green, or black).
    const stride = 12800;
    let sumR = 0, sumG = 0, sumB = 0, redPixels = 0, count = 0;
    for (let y = 16; y < 64; y += 2) {
      const row = y * 160;
      for (let x = 24; x < 136; x += 2) {
        const i = row + x;
        const r = crop.data[i], g = crop.data[stride + i], b = crop.data[2 * stride + i];
        sumR += r;
        sumG += g;
        sumB += b;
        if (r > 0.30 && r - Math.max(g, b) > 0.17) redPixels++;
        count++;
      }
    }
    const meanR = sumR / count, meanG = sumG / count, meanB = sumB / count;
    return meanR > 0.30 && meanR - Math.max(meanG, meanB) > 0.16 && redPixels > 0.42 * count;
  }

  function thumbnail(crop) {
    const rgba = new Uint8ClampedArray(160 * 80 * 4);
    for (let i = 0; i < 12800; i++) {
      for (let c = 0; c < 3; c++) rgba[4 * i + c] = Math.round(clamp(crop.data[c * 12800 + i]) * 255);
      rgba[4 * i + 3] = 255;
    }
    return rgba;
  }

  function validCorners(q) {
    if (!q || q.length !== 4) return false;
    for (let i = 0; i < 4; i++) {
      if (!Number.isFinite(q[i][0]) || !Number.isFinite(q[i][1])) return false;
      if (q[i][0] < -2 || q[i][0] > wOrig + 2 || q[i][1] < -2 || q[i][1] > hOrig + 2) return false;
      const a = q[i], b = q[(i + 1) % 4], c = q[(i + 2) % 4];
      if ((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) <= 0) return false;
    }
    return true;
  }

  function cornersToBox(corners) {
    const x0 = Math.max(0, Math.min(corners[0][0], corners[1][0], corners[2][0], corners[3][0]));
    const y0 = Math.max(0, Math.min(corners[0][1], corners[1][1], corners[2][1], corners[3][1]));
    const x1 = Math.min(wOrig, Math.max(corners[0][0], corners[1][0], corners[2][0], corners[3][0]));
    const y1 = Math.min(hOrig, Math.max(corners[0][1], corners[1][1], corners[2][1], corners[3][1]));
    return [x0, y0, x1, y1];
  }

  function newTrack(box, time) {
    if (tracks.length >= 256) throw Error('トラック上限です。終了して読み出してください。');
    const memory = {
      mem_h: zero([1, 12, 112]),
      mem_P: zero([1, 12, 112]),
      mem_conflict: zero([1, 12]),
      color_h: zero([1, 32]),
      color_P: zero([1, 32]),
      pol_h: zero([1]),
      pol_P: zero([1]),
      pose_ema: zero([1, 4135]),
      last_timestamp_s: float([-1], [1]),
    };
    const track = {
      id: nextPlateId++,
      box,
      corners: [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
      firstTime: time,
      last: time,
      lastFrame: frameIndex,
      state: 'COLLECTING',
      dormant: false,
      dormantSince: Infinity,
      memory,
      frames: 0,
      readouts: 0,
      odds: 0,
      firstSupport: 0,
      sharpness: -Infinity,
      velocity: [0, 0],
      sizeVelocity: [0, 0],
      reidEmb: null,
      vehicleId: null,
      missed: 0,
    };
    tracks.push(track);
    progress({ track_id: track.id, state: track.state, readouts: 0 });
    return track;
  }

  function updateMotionScout(monitoredBoxes) {
    scoutMask.fill(0);
    for (const b of monitoredBoxes) {
      const gx0 = Math.max(0, Math.floor((b[0] / wOrig) * SCOUT_W) - 1);
      const gy0 = Math.max(0, Math.floor((b[1] / hOrig) * SCOUT_H) - 1);
      const gx1 = Math.min(SCOUT_W - 1, Math.floor((b[2] / wOrig) * SCOUT_W) + 1);
      const gy1 = Math.min(SCOUT_H - 1, Math.floor((b[3] / hOrig) * SCOUT_H) + 1);
      for (let gy = gy0; gy <= gy1; gy++) {
        const row = gy * SCOUT_W;
        for (let gx = gx0; gx <= gx1; gx++) scoutMask[row + gx] = 1;
      }
    }
    let idx = 0;
    let totalLuma = 0;
    let allDiff = 0;
    let unmonDiff = 0, unmonCnt = 0, hotCells = 0;
    let borderDiff = 0, borderCnt = 0;
    for (let gy = 0; gy < SCOUT_H; gy++) {
      const sy = Math.min(hOrig - 1, Math.floor(((gy + 0.5) * hOrig) / SCOUT_H));
      const row = sy * wOrig;
      const isBorderY = gy <= 2 || gy >= SCOUT_H - 3;
      for (let gx = 0; gx < SCOUT_W; gx++) {
        const sx = Math.min(wOrig - 1, Math.floor(((gx + 0.5) * wOrig) / SCOUT_W));
        const p = (row + sx) * 4;
        const val = (0.299 * frame[p] + 0.587 * frame[p + 1] + 0.114 * frame[p + 2]) / 255;
        scoutCur[idx] = val;
        totalLuma += val;
        if (hasScoutPrev) {
          const d = Math.abs(val - scoutPrev[idx]);
          allDiff += d;
          if (isBorderY || gx <= 3 || gx >= SCOUT_W - 4) {
            borderDiff += d;
            borderCnt++;
          }
          if (!scoutMask[idx]) {
            unmonDiff += d;
            unmonCnt++;
            if (d > 0.085) hotCells++;
          }
        }
        idx++;
      }
    }
    scoutPrev.set(scoutCur);
    const cellTotal = SCOUT_W * SCOUT_H;
    const lumaMean = totalLuma / cellTotal;
    const blackout = prevLumaMean !== null && prevLumaMean >= 0.095 && lumaMean < 0.25 * prevLumaMean;
    prevLumaMean = lumaMean;
    if (!hasScoutPrev) {
      hasScoutPrev = true;
      return { emptyTriggered: true, busyTriggered: false, sceneCut: false, blackout };
    }
    const allMean = allDiff / cellTotal;
    const unmonMean = unmonCnt > 0 ? unmonDiff / unmonCnt : 0;
    const borderMean = borderCnt > 0 ? borderDiff / borderCnt : 0;
    const sceneCut = allMean > 0.165 && borderMean > 0.14;
    const emptyTriggered = unmonMean > 0.04 || hotCells >= 14 || borderMean > 0.06;
    const busyTriggered = unmonMean > 0.06 || hotCells >= 26 || borderMean > 0.085;
    return { emptyTriggered, busyTriggered, sceneCut, blackout };
  }

  function predictedBox(track, time) {
    const dt = clamp(time - track.last, 0, 0.45);
    const decay = track.state === 'FINALIZED' ? 0.5 * Math.exp(-1.5 * dt) : (track.missed > 0 ? Math.exp(-2.2 * dt) : 1);
    const [vx, vy] = track.velocity;
    const [vw, vh] = (track.state === 'FINALIZED' || track.missed > 0) ? [0, 0] : (track.sizeVelocity ?? [0, 0]);
    const [a, b, c, d] = track.box;
    const cx = 0.5 * (a + c) + vx * dt * decay;
    const cy = 0.5 * (b + d) + vy * dt * decay;
    const w = Math.max(5, (c - a) + vw * dt * 0.35 * decay);
    const h = Math.max(3, (d - b) + vh * dt * 0.35 * decay);
    return [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h];
  }

  function predictedCorners(track, time) {
    const dt = clamp(time - track.last, 0, 0.45);
    const decay = track.missed > 0 ? Math.exp(-2.2 * dt) : 1;
    const dx = track.velocity[0] * dt * decay;
    const dy = track.velocity[1] * dt * decay;
    return track.corners.map(([x, y]) => [x + dx, y + dy]);
  }

  function predictedVehicleBox(vt, time) {
    const dt = clamp(time - vt.last, 0, 0.5);
    const decay = vt.missed > 0 ? Math.exp(-1.8 * dt) : 1;
    const cx = vt.cx + vt.vx * dt * decay;
    const cy = vt.cy + vt.vy * dt * decay;
    const w = Math.max(12, vt.w + vt.vw * dt * 0.5 * decay);
    const h = Math.max(12, vt.h + vt.vh * dt * 0.5 * decay);
    return [
      clamp(cx - 0.5 * w, 0, wOrig),
      clamp(cy - 0.5 * h, 0, hOrig),
      clamp(cx + 0.5 * w, 0, wOrig),
      clamp(cy + 0.5 * h, 0, hOrig),
    ];
  }

  function updateVehicleTracks(dets, time) {
    const matchedTrack = new Set();
    const matchedDet = new Set();
    const pairs = [];
    for (let ti = 0; ti < vtracks.length; ti++) {
      const vt = vtracks[ti];
      const pbox = predictedVehicleBox(vt, time);
      const pcx = 0.5 * (pbox[0] + pbox[2]);
      const pcy = 0.5 * (pbox[1] + pbox[3]);
      const diag = Math.max(32, Math.hypot(pbox[2] - pbox[0], pbox[3] - pbox[1]));
      for (let di = 0; di < dets.length; di++) {
        const d = dets[di];
        const ov = iou(pbox, d.box);
        const dcx = 0.5 * (d.box[0] + d.box[2]);
        const dcy = 0.5 * (d.box[1] + d.box[3]);
        const dist = Math.hypot(dcx - pcx, dcy - pcy) / diag;
        const embSim = dotVec(vt.emb, d.emb);
        if (ov >= 0.12 || (dist <= 0.75 && embSim >= 0.45)) {
          const score = 2.6 * ov + 1.6 * Math.max(0, embSim) - dist;
          pairs.push({ ti, di, score });
        }
      }
    }
    pairs.sort((a, b) => b.score - a.score);
    for (const { ti, di } of pairs) {
      if (matchedTrack.has(ti) || matchedDet.has(di)) continue;
      matchedTrack.add(ti);
      matchedDet.add(di);
      const vt = vtracks[ti];
      const d = dets[di];
      const dt = Math.max(0.03, time - vt.last);
      const ncx = 0.5 * (d.box[0] + d.box[2]);
      const ncy = 0.5 * (d.box[1] + d.box[3]);
      const nw = Math.max(10, d.box[2] - d.box[0]);
      const nh = Math.max(10, d.box[3] - d.box[1]);
      const predCx = vt.cx + vt.vx * dt;
      const predCy = vt.cy + vt.vy * dt;
      const predW = Math.max(10, vt.w + vt.vw * dt * 0.5);
      const predH = Math.max(10, vt.h + vt.vh * dt * 0.5);
      const alpha = 0.68, beta = 0.44;
      vt.cx = predCx + alpha * (ncx - predCx);
      vt.cy = predCy + alpha * (ncy - predCy);
      vt.vx = clamp(vt.vx + (beta / dt) * (ncx - predCx), -wOrig * 1.5, wOrig * 1.5);
      vt.vy = clamp(vt.vy + (beta / dt) * (ncy - predCy), -hOrig * 1.5, hOrig * 1.5);
      vt.w = predW + 0.55 * (nw - predW);
      vt.h = predH + 0.55 * (nh - predH);
      vt.vw = clamp(vt.vw + (0.28 / dt) * (nw - predW), -wOrig, wOrig);
      vt.vh = clamp(vt.vh + (0.28 / dt) * (nh - predH), -hOrig, hOrig);
      vt.box = [
        clamp(vt.cx - 0.5 * vt.w, 0, wOrig),
        clamp(vt.cy - 0.5 * vt.h, 0, hOrig),
        clamp(vt.cx + 0.5 * vt.w, 0, wOrig),
        clamp(vt.cy + 0.5 * vt.h, 0, hOrig),
      ];
      vt.score = 0.65 * vt.score + 0.35 * d.score;
      vt.maxScore = Math.max(vt.maxScore ?? vt.score, d.score);
      vt.maxDisp = Math.max(vt.maxDisp ?? 0, Math.hypot(vt.cx - vt.firstCx, vt.cy - vt.firstCy));
      vt.mask = d.mask ?? vt.mask;
      vt.maxMountProb = Math.max(vt.maxMountProb ?? 0, d.mountProb ?? 0);
      if (d.mountBox && d.mountProb >= 0.25) {
        vt.mountRel = [
          (d.mountBox[0] - d.box[0]) / nw,
          (d.mountBox[1] - d.box[1]) / nh,
          (d.mountBox[2] - d.box[0]) / nw,
          (d.mountBox[3] - d.box[1]) / nh,
        ];
        vt.mountProb = d.mountProb;
      }
      if (d.emb) {
        const blended = new Float32Array(d.emb.length);
        for (let i = 0; i < blended.length; i++) blended[i] = 0.72 * vt.emb[i] + 0.28 * d.emb[i];
        vt.emb = normalizeVec(blended);
      }
      vt.last = time;
      vt.lastFrame = frameIndex;
      vt.hits++;
      vt.missed = 0;
    }

    for (let ti = 0; ti < vtracks.length; ti++) {
      if (matchedTrack.has(ti)) continue;
      vtracks[ti].missed++;
    }
    vtracks = vtracks.filter(vt => vt.missed <= 6 && time - vt.last <= 0.9);

    for (let di = 0; di < dets.length; di++) {
      if (matchedDet.has(di)) continue;
      const d = dets[di];
      const cx = 0.5 * (d.box[0] + d.box[2]);
      const cy = 0.5 * (d.box[1] + d.box[3]);
      const w = Math.max(10, d.box[2] - d.box[0]);
      const h = Math.max(10, d.box[3] - d.box[1]);
      vtracks.push({
        id: nextVehicleId++,
        cx, cy, w, h,
        firstCx: cx, firstCy: cy, maxDisp: 0,
        vx: 0, vy: 0, vw: 0, vh: 0,
        box: d.box,
        score: d.score,
        maxScore: d.score,
        mask: d.mask ?? null,
        emb: d.emb ?? null,
        mountRel: d.mountBox && d.mountProb >= 0.25 ? [
          (d.mountBox[0] - d.box[0]) / w,
          (d.mountBox[1] - d.box[1]) / h,
          (d.mountBox[2] - d.box[0]) / w,
          (d.mountBox[3] - d.box[1]) / h,
        ] : null,
        mountProb: d.mountProb ?? 0,
        maxMountProb: d.mountProb ?? 0,
        last: time,
        lastFrame: frameIndex,
        hits: 1,
        missed: 0,
        mountCooldown: 0,
      });
    }
  }

  function decodePlatesFromOutput(output, mapBox, mapCorners, provenance, minScore, maxScore = 1.01) {
    const cls = output.cls_logits.data;
    const boxes = output.pred_boxes.data;
    const cornersData = output.pred_corners?.data;
    const out = [];
    const total = cls.length / 3;
    const minLogit = Math.log(minScore / (1 - minScore));
    for (let i = 0; i < total; i++) {
      const pLogit = cls[i * 3];
      const vLogit = cls[i * 3 + 1];
      const sLogit = cls[i * 3 + 2];
      if (pLogit < minLogit || pLogit <= vLogit || pLogit <= sLogit) continue;
      const score = sigmoid(pLogit);
      if (score >= maxScore) continue;
      const box = mapBox(boxes, i * 4);
      const bw = box[2] - box[0];
      const bh = box[3] - box[1];
      if (bw < 5 || bh < 2.8 || bw > 0.96 * wOrig || bh > 0.90 * hOrig) continue;
      const aspect = bw / Math.max(1e-3, bh);
      if (aspect < 0.45 || aspect > 5.0) continue;
      const corners = cornersData ? mapCorners(cornersData, i * 8) : null;
      if (corners) {
        const qxs = [corners[0][0], corners[1][0], corners[2][0], corners[3][0]];
        const qys = [corners[0][1], corners[1][1], corners[2][1], corners[3][1]];
        const qw = Math.max(...qxs) - Math.min(...qxs);
        const qh = Math.max(...qys) - Math.min(...qys);
        if (qw < 0.32 * bw || qw > 2.0 * bw || qh < 0.32 * bh || qh > 2.0 * bh) continue;
      }
      out.push({ box, score, corners, provenance });
    }
    return out;
  }

  async function runSmartZoomMosaic(rois, curStrongBoxes) {
    if (!rois.length) return [];
    const subRegions = [];
    const subBuf = zoomTileBuf;
    const isMosaic = rois.length >= 2;
    if (isMosaic) {
      subBuf.fill(114);
      const qw = Math.floor(ZOOM_W / 2) - 1;
      const qh = Math.floor(ZOOM_H / 2) - 1;
      const slots = [
        [0, 0, qw, qh],
        [qw + 2, 0, ZOOM_W, qh],
        [0, qh + 2, qw, ZOOM_H],
        [qw + 2, qh + 2, ZOOM_W, ZOOM_H],
      ];
      const count = Math.min(4, rois.length);
      for (let i = 0; i < count; i++) {
        const [rx0, ry0, rx1, ry1] = rois[i];
        const rw = Math.max(16, rx1 - rx0);
        const rh = Math.max(12, ry1 - ry0);
        const [qx0, qy0, qx1, qy1] = slots[i];
        const sw = qx1 - qx0, sh = qy1 - qy0;
        const sc = Math.min(sw / rw, sh / rh);
        const nw = Math.max(1, Math.round(rw * sc));
        const nh = Math.max(1, Math.round(rh * sc));
        const px = qx0 + Math.floor((sw - nw) / 2);
        const py = qy0 + Math.floor((sh - nh) / 2);
        for (let y = 0; y < nh; y++) {
          const sy = clamp(ry0 + (y + 0.5) / sc - 0.5, 0, hOrig - 1);
          const y0 = sy | 0, y1 = y0 + 1 < hOrig ? y0 + 1 : y0, fy = sy - y0;
          for (let x = 0; x < nw; x++) {
            const sx = clamp(rx0 + (x + 0.5) / sc - 0.5, 0, wOrig - 1);
            const x0 = sx | 0, x1 = x0 + 1 < wOrig ? x0 + 1 : x0, fx = sx - x0;
            const wa = (1 - fx) * (1 - fy), wb2 = fx * (1 - fy), wd = (1 - fx) * fy, we = fx * fy;
            const a = (y0 * wOrig + x0) * 4, b = (y0 * wOrig + x1) * 4;
            const d = (y1 * wOrig + x0) * 4, e = (y1 * wOrig + x1) * 4;
            const dst = ((py + y) * ZOOM_W + (px + x)) * 4;
            subBuf[dst] = frame[a] * wa + frame[b] * wb2 + frame[d] * wd + frame[e] * we;
            subBuf[dst + 1] = frame[a + 1] * wa + frame[b + 1] * wb2 + frame[d + 1] * wd + frame[e + 1] * we;
            subBuf[dst + 2] = frame[a + 2] * wa + frame[b + 2] * wb2 + frame[d + 2] * wd + frame[e + 2] * we;
          }
        }
        subRegions.push({ qx0, qy0, qx1, qy1, sc, px, py, rx0, ry0, rx1, ry1 });
      }
    } else {
      subBuf.fill(114);
      const [rx0, ry0, rx1, ry1] = rois[0];
      const rw = Math.max(16, rx1 - rx0), rh = Math.max(12, ry1 - ry0);
      const sc = Math.min(ZOOM_W / rw, ZOOM_H / rh);
      const nw = Math.max(1, Math.round(rw * sc)), nh = Math.max(1, Math.round(rh * sc));
      const px = Math.floor((ZOOM_W - nw) / 2), py = Math.floor((ZOOM_H - nh) / 2);
      for (let y = 0; y < nh; y++) {
        const sy = clamp(ry0 + (y + 0.5) / sc - 0.5, 0, hOrig - 1);
        const y0 = sy | 0, y1 = y0 + 1 < hOrig ? y0 + 1 : y0, fy = sy - y0;
        for (let x = 0; x < nw; x++) {
          const sx = clamp(rx0 + (x + 0.5) / sc - 0.5, 0, wOrig - 1);
          const x0 = sx | 0, x1 = x0 + 1 < wOrig ? x0 + 1 : x0, fx = sx - x0;
          const wa = (1 - fx) * (1 - fy), wb2 = fx * (1 - fy), wd = (1 - fx) * fy, we = fx * fy;
          const a = (y0 * wOrig + x0) * 4, b = (y0 * wOrig + x1) * 4;
          const d = (y1 * wOrig + x0) * 4, e = (y1 * wOrig + x1) * 4;
          const dst = ((py + y) * ZOOM_W + (px + x)) * 4;
          subBuf[dst] = frame[a] * wa + frame[b] * wb2 + frame[d] * wd + frame[e] * we;
          subBuf[dst + 1] = frame[a + 1] * wa + frame[b + 1] * wb2 + frame[d + 1] * wd + frame[e + 1] * we;
          subBuf[dst + 2] = frame[a + 2] * wa + frame[b + 2] * wb2 + frame[d + 2] * wd + frame[e + 2] * we;
        }
      }
      subRegions.push({ qx0: 0, qy0: 0, qx1: ZOOM_W, qy1: ZOOM_H, sc, px, py, rx0, ry0, rx1, ry1 });
    }
    // Upsample 384x256 zoom mosaic cleanly into fixed 768x448 input tensor (stateless: zero h3/h4).
    const zFloat = new Float32Array(3 * LB_W * LB_H);
    const zStride = LB_W * LB_H;
    const sxTile = ZOOM_W / LB_W;
    const syTile = ZOOM_H / LB_H;
    for (let y = 0; y < LB_H; y++) {
      const sy = (y + 0.5) * syTile - 0.5;
      const y0 = Math.max(0, sy | 0), y1 = Math.min(ZOOM_H - 1, y0 + 1), fy = Math.max(0, sy - y0);
      for (let x = 0; x < LB_W; x++) {
        const sx = (x + 0.5) * sxTile - 0.5;
        const x0 = Math.max(0, sx | 0), x1 = Math.min(ZOOM_W - 1, x0 + 1), fx = Math.max(0, sx - x0);
        const wa = (1 - fx) * (1 - fy) / 255, wb2 = fx * (1 - fy) / 255;
        const wd = (1 - fx) * fy / 255, we = fx * fy / 255;
        const a = (y0 * ZOOM_W + x0) * 4, b = (y0 * ZOOM_W + x1) * 4;
        const d = (y1 * ZOOM_W + x0) * 4, e = (y1 * ZOOM_W + x1) * 4;
        const dst = y * LB_W + x;
        zFloat[dst] = subBuf[a] * wa + subBuf[b] * wb2 + subBuf[d] * wd + subBuf[e] * we;
        zFloat[dst + zStride] = subBuf[a + 1] * wa + subBuf[b + 1] * wb2 + subBuf[d + 1] * wd + subBuf[e + 1] * we;
        zFloat[dst + 2 * zStride] = subBuf[a + 2] * wa + subBuf[b + 2] * wb2 + subBuf[d + 2] * wd + subBuf[e + 2] * we;
      }
    }
    const zOut = await sessions.temporal_det_nano.run({
      images: float(zFloat, [1, 3, LB_H, LB_W]),
      h_p3_in: zeroH3,
      h_p4_in: zeroH4,
    });
    const cls = zOut.cls_logits.data;
    const boxes = zOut.pred_boxes.data;
    const cornersData = zOut.pred_corners?.data;
    const total = cls.length / 3;
    const zoomThresh = Math.max(plateThresh, 0.28);
    const zoomLogitMin = Math.log(zoomThresh / (1 - zoomThresh));
    const found = [];
    for (let i = 0; i < total; i++) {
      const pL = cls[i * 3], vL = cls[i * 3 + 1], sL = cls[i * 3 + 2];
      if (pL < zoomLogitMin || pL <= vL || pL <= sL) continue;
      const tcx = 0.5 * (boxes[i * 4] + boxes[i * 4 + 2]) * sxTile;
      const tcy = 0.5 * (boxes[i * 4 + 1] + boxes[i * 4 + 3]) * syTile;
      for (const reg of subRegions) {
        if (tcx < reg.qx0 || tcx > reg.qx1 || tcy < reg.qy0 || tcy > reg.qy1) continue;
        const invZ = 1 / Math.max(1e-6, reg.sc);
        const box = [
          clamp((boxes[i * 4] * sxTile - reg.px) * invZ + reg.rx0, 0, wOrig),
          clamp((boxes[i * 4 + 1] * syTile - reg.py) * invZ + reg.ry0, 0, hOrig),
          clamp((boxes[i * 4 + 2] * sxTile - reg.px) * invZ + reg.rx0, 0, wOrig),
          clamp((boxes[i * 4 + 3] * syTile - reg.py) * invZ + reg.ry0, 0, hOrig),
        ];
        const bw = box[2] - box[0], bh = box[3] - box[1];
        if (bw < 5 || bh < 2.8 || bw / Math.max(1e-3, bh) < 0.5 || bw / Math.max(1e-3, bh) > 4.8) continue;
        if (curStrongBoxes.some(sb => iou(sb, box) >= 0.35)) continue;
        const corners = cornersData
          ? Array.from({ length: 4 }, (_, k) => [
              clamp((cornersData[i * 8 + k * 2] * sxTile - reg.px) * invZ + reg.rx0, 0, wOrig),
              clamp((cornersData[i * 8 + k * 2 + 1] * syTile - reg.py) * invZ + reg.ry0, 0, hOrig),
            ])
          : null;
        found.push({ box, score: sigmoid(pL), corners, provenance: 1 });
      }
    }
    return found;
  }

  async function runStage2Batch(items) {
    const n = items.length;
    if (!n) return [];
    const fovBatch = n === 1 ? fovea1Buf : new Float32Array(n * 3 * 96 * 160);
    const surBatch = n === 1 ? surround1Buf : new Float32Array(n * 3 * 64 * 96);
    const policies = n === 1 ? policy1Buf : new BigInt64Array(n);
    const wins = [];
    for (let i = 0; i < n; i++) {
      const [fw, sw] = windows(items[i].box, items[i].gamma ?? 1.0);
      wins.push(fw);
      const border = estimateBorderColor(items[i].box);
      fillPatch(fw, 160, 96, fovBatch, i * 3 * 96 * 160, border);
      fillPatch(sw, 96, 64, surBatch, i * 3 * 64 * 96, border);
      policies[i] = BigInt(items[i].provenance ?? 0);
    }
    const out = await sessions.stage2.run({
      fovea_rois: float(fovBatch, [n, 3, 96, 160]),
      surround_rois: float(surBatch, [n, 3, 64, 96]),
      sampling_policy: new runtime.Tensor('int64', policies, [n]),
    });
    const results = [];
    for (let i = 0; i < n; i++) {
      const fw = wins[i];
      const valid = Boolean(out.geometry_valid.data[i]);
      const llr = out.plate_null_llr.data[i];
      const support = sigmoid(llr);
      const nc = Math.max(0, out.support_nonconformity.data[i]);
      const visOffset = i * 4;
      let visSum = 0;
      for (let k = 0; k < 4; k++) visSum += sigmoid(out.corner_vis_logits.data[visOffset + k]);
      const visibility = visSum / 4;
      const cnOffset = i * 8;
      let minU = 1, maxU = 0, minV = 1, maxV = 0;
      const corners = [];
      for (let k = 0; k < 4; k++) {
        const u = out.corners_norm.data[cnOffset + k * 2];
        const v = out.corners_norm.data[cnOffset + k * 2 + 1];
        if (u < minU) minU = u;
        if (u > maxU) maxU = u;
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
        corners.push([u * fw[2] + fw[0], v * fw[3] + fw[1]]);
      }
      const needsRecrop =
        valid && support >= minIsPlateProb &&
        (minU < 0.02 || maxU > 0.98 || minV < 0.02 || maxV > 0.98) &&
        corners.every(([x, y]) => x >= 2 && x <= wOrig - 2 && y >= 2 && y <= hOrig - 2);
      const reidEmb = out.reid_emb ? normalizeVec(out.reid_emb.data.subarray(i * 48, (i + 1) * 48)) : null;
      results.push({ valid, llr, support, nc, visibility, corners, needsRecrop, reidEmb });
    }
    return results;
  }

  async function process({ rgbaBuffer, wOrig: width, hOrig: height, timestamp: time }) {
    if (phase !== 'COLLECTING') throw Error('Collection is closed');
    if (!Number.isInteger(width) || !Number.isInteger(height) || Math.min(width, height) <= 0 || Math.max(width, height) > 1920)
      throw Error('Invalid native frame dimensions');
    frame = rgbaBuffer instanceof Uint8ClampedArray ? rgbaBuffer : new Uint8ClampedArray(rgbaBuffer);
    if (frame.length !== width * height * 4 || !Number.isFinite(time) || time <= lastTime)
      throw Error('Invalid frame or non-monotonic timestamp');
    wOrig = width;
    hOrig = height;
    lastTime = time;
    const started = performance.now();

    // Purge stale finalized tracks (keep accepted longer to suppress re-reads cleanly; expire unreadable quickly).
    tracks = tracks.filter(t => {
      if (t.state !== 'FINALIZED') return true;
      return t.result.status === 'ACCEPTED' ? time - t.last < 2.2 : time - t.last < 0.35;
    });

    const active = tracks.filter(t => t.state === 'COLLECTING' && !t.dormant);
    const dormant = tracks.filter(t => t.state === 'COLLECTING' && t.dormant);
    const finalizedActive = tracks.filter(
      t => t.state === 'FINALIZED' && t.result.status === 'ACCEPTED' && frameIndex - t.lastFrame <= 2 && time - t.last < 2.2
    );
    const monitoredBoxes = [
      ...active.map(t => predictedBox(t, time)),
      ...finalizedActive.map(t => predictedBox(t, time)),
      ...vtracks.map(vt => predictedVehicleBox(vt, time)),
    ];
    const scout = updateMotionScout(monitoredBoxes);
    const preSealedOnCut = [];
    if (scout.sceneCut) {
      if (autoSealFrames > 0) {
        for (const t of tracks) {
          if (t.state !== 'COLLECTING') continue;
          if (discardShortUnreadable && t.frames <= 2) t.state = 'DISCARDED';
          else preSealedOnCut.push(await sealTrack(t));
        }
        tracks = [];
      }
      vtracks = [];
      h3 = zero([1, 36, 56, 96]);
      h4 = zero([1, 36, 28, 48]);
    }
    const effHeartbeat = active.length >= 4 ? maxHeartbeat : baseHeartbeat;
    const activeConfirmed = !scout.sceneCut && (active.some(t => t.frames >= 2) || finalizedActive.length > 0);
    const missingConfirmed = !scout.sceneCut && active.some(t => t.frames >= 2 && t.missed > 0);
    const framesSinceFull = frameIndex - lastFull;

    const full =
      frameIndex === 0 ||
      scout.sceneCut ||
      framesSinceFull >= effHeartbeat ||
      (!activeConfirmed && ((scout.emptyTriggered && framesSinceFull >= 2) || framesSinceFull >= 3)) ||
      (scout.busyTriggered && framesSinceFull >= 4) ||
      (missingConfirmed && framesSinceFull >= 3);

    const proposals = [];
    let lbInfo = null;

    if (full) {
      lbInfo = letterbox();
      const { images, scale, ox, oy } = lbInfo;
      const invS = 1 / Math.max(1e-6, scale);
      const nativeBox = (buf, base) => [
        clamp((buf[base] - ox) * invS, 0, wOrig),
        clamp((buf[base + 1] - oy) * invS, 0, hOrig),
        clamp((buf[base + 2] - ox) * invS, 0, wOrig),
        clamp((buf[base + 3] - oy) * invS, 0, hOrig),
      ];
      const nativeCorners = (buf, base) =>
        Array.from({ length: 4 }, (_, k) => [
          clamp((buf[base + k * 2] - ox) * invS, 0, wOrig),
          clamp((buf[base + k * 2 + 1] - oy) * invS, 0, hOrig),
        ]);

      const output = await sessions.temporal_det_nano.run({ images, h_p3_in: h3, h_p4_in: h4 });
      if (!scout.blackout) {
        h3 = output.h_p3_out;
        h4 = output.h_p4_out;
      }
      lastFull = frameIndex;
      fullDetections++;

      const cls = output.cls_logits.data;
      const numQ = output.veh_log_odds.data.length;
      const aPyr = cls.length / 3 - numQ;

      // 1. Decode D-FINE vehicle queries using calibrated `cls_logits` + `pred_boxes` (not `raw_q_boxes_px`).
      const rawVehCandidates = [];
      const frameArea = wOrig * hOrig;
      for (let q = 0; q < numQ; q++) {
        const idx = aPyr + q;
        const vLogit = cls[idx * 3 + 1];
        const sLogit = cls[idx * 3 + 2];
        if (vLogit < vehLogitMin || vLogit <= sLogit) continue;
        const box = nativeBox(output.pred_boxes.data, idx * 4);
        const bw = box[2] - box[0], bh = box[3] - box[1];
        if (bw < 14 || bh < 14) continue;
        const aspect = bw / bh;
        if (aspect < 0.36 || aspect > 3.5) continue;
        const score = sigmoid(vLogit);
        if (bw * bh > 0.28 * frameArea && score < 0.78) continue;
        const mountBox = output.pred_q_mount_boxes
          ? nativeBox(output.pred_q_mount_boxes.data, q * 4)
          : null;
        const mountProb = output.pred_q_mount_vis_logit
          ? sigmoid(output.pred_q_mount_vis_logit.data[q])
          : 0;
        const emb = output.appearance_emb
          ? normalizeVec(output.appearance_emb.data.subarray(q * 48, (q + 1) * 48))
          : null;
        rawVehCandidates.push({
          box,
          lbBox: [
            clamp(box[0] * scale + ox, 0, LB_W),
            clamp(box[1] * scale + oy, 0, LB_H),
            clamp(box[2] * scale + ox, 0, LB_W),
            clamp(box[3] * scale + oy, 0, LB_H),
          ],
          score,
          q,
          emb,
          mountBox,
          mountProb,
        });
      }
      const nmsVehs = nms(rawVehCandidates, 12, 0.52);
      // Step 1: Drop giant scene-spanning multi-object wrapper hallucinations before sub-part pruning.
      const nonWrapperVehs = nmsVehs.filter((a, i) => {
        const bwA = a.box[2] - a.box[0], bhA = a.box[3] - a.box[1];
        const areaA = Math.max(1, bwA * bhA);
        let enclosedStrong = 0;
        for (let j = 0; j < nmsVehs.length; j++) {
          if (i === j) continue;
          const b = nmsVehs[j];
          const ix = Math.max(0, Math.min(a.box[2], b.box[2]) - Math.max(a.box[0], b.box[0]));
          const iy = Math.max(0, Math.min(a.box[3], b.box[3]) - Math.max(a.box[1], b.box[1]));
          const inter = ix * iy;
          if (inter <= 0) continue;
          const areaB = Math.max(1, (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]));
          if (b.score >= 0.62 && inter / areaB >= 0.72 && areaA >= 2.0 * areaB) enclosedStrong++;
          if (a.score < 0.85 && b.score >= a.score && inter / areaB >= 0.70 && areaA >= 2.2 * areaB) return false;
          if (
            areaA > 0.22 * frameArea &&
            bwA > 0.44 * wOrig &&
            b.score >= 0.76 &&
            inter / areaB >= 0.76 &&
            areaA >= 2.5 * areaB
          ) {
            return false;
          }
        }
        if (enclosedStrong >= 2) return false;
        return true;
      });
      // Step 2: Drop interior window/cab/door sub-part duplicates inside a stronger real vehicle.
      const keptVehs = nonWrapperVehs.filter((a, i) => {
        const areaA = Math.max(1, (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]));
        for (let j = 0; j < nonWrapperVehs.length; j++) {
          if (i === j) continue;
          const b = nonWrapperVehs[j];
          if (b.score < a.score) continue;
          const ix = Math.max(0, Math.min(a.box[2], b.box[2]) - Math.max(a.box[0], b.box[0]));
          const iy = Math.max(0, Math.min(a.box[3], b.box[3]) - Math.max(a.box[1], b.box[1]));
          const inter = ix * iy;
          if (inter <= 0) continue;
          const areaB = Math.max(1, (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]));
          if (b.score >= 0.72 && inter / areaA >= 0.72 && areaA <= 0.55 * areaB) return false;
        }
        return true;
      });
      if (keptVehs.length) {
        const cutout = await sessions.vehicle_cutout_nano.run({
          p2_feat: output.p2_feat,
          aux_edge_3ch: output.aux_edge_3ch,
          veh_rois: float(keptVehs.flatMap(v => [0, ...v.lbBox]), [keptVehs.length, 5]),
          query_condition: float(
            keptVehs.flatMap(v => Array.from(output.query_condition.data.subarray(v.q * 64, (v.q + 1) * 64))),
            [keptVehs.length, 64]
          ),
        });
        keptVehs.forEach((v, i) => {
          v.mask = cutout.cutout_logits.data.slice(i * 4096, (i + 1) * 4096);
        });
      }
      updateVehicleTracks(keptVehs, time);

      // 2. Decode Pass-1 pyramid/query plate candidates.
      const pass1Plates = decodePlatesFromOutput(output, nativeBox, nativeCorners, 0, plateThresh);
      proposals.push(...pass1Plates);

      // 3. Pass-2 Smart-Zoom (HD/4K videos where scale < 0.88) for unplated vehicles & weak plate peaks.
      if (zoomCooldown > 0) zoomCooldown--;
      if (scale < 0.88 && zoomCooldown === 0) {
        const pass1StrongBoxes = pass1Plates.filter(p => p.score >= 0.55).map(p => p.box);
        const zoomRois = [];
        const enqBox = (rx0, ry0, rx1, ry1) => {
          if (zoomRois.length >= 4) return;
          const x0 = Math.max(0, Math.floor(rx0)), y0 = Math.max(0, Math.floor(ry0));
          const x1 = Math.min(wOrig, Math.ceil(rx1)), y1 = Math.min(hOrig, Math.ceil(ry1));
          if (x1 - x0 < 28 || y1 - y0 < 20) return;
          const win = [x0, y0, x1, y1];
          if (zoomRois.some(prev => iou(prev, win) >= 0.6)) return;
          zoomRois.push(win);
        };
        for (const vt of vtracks) {
          if (zoomRois.length >= 4) break;
          const vb = predictedVehicleBox(vt, time);
          const vw = vb[2] - vb[0], vh = vb[3] - vb[1];
          const hasStrong = pass1StrongBoxes.some(pb => {
            const pcx = 0.5 * (pb[0] + pb[2]), pcy = 0.5 * (pb[1] + pb[3]);
            return pcx >= vb[0] - 0.06 * vw && pcx <= vb[2] + 0.06 * vw && pcy >= vb[1] && pcy <= vb[3] + 0.16 * vh;
          });
          if (!hasStrong && vw * scale <= 360) {
            enqBox(vb[0] - 0.1 * vw, vb[1] + 0.25 * vh, vb[2] + 0.1 * vw, vb[3] + 0.1 * vh);
          }
        }
        if (zoomRois.length < 4) {
          const weakPeaks = nms(
            decodePlatesFromOutput(output, nativeBox, nativeCorners, 1, weakPlateThresh, plateThresh),
            4,
            0.35
          );
          for (const wp of weakPeaks) {
            const wcx = 0.5 * (wp.box[0] + wp.box[2]), wcy = 0.5 * (wp.box[1] + wp.box[3]);
            const covered = pass1StrongBoxes.some(sb => wcx >= sb[0] - 8 && wcx <= sb[2] + 8 && wcy >= sb[1] - 8 && wcy <= sb[3] + 8);
            if (covered) continue;
            const ww = Math.max(96, 4.5 * (wp.box[2] - wp.box[0]));
            const wh = Math.max(64, 5.0 * (wp.box[3] - wp.box[1]));
            enqBox(wcx - 0.5 * ww, wcy - 0.5 * wh, wcx + 0.5 * ww, wcy + 0.5 * wh);
          }
        }
        if (zoomRois.length) {
          const zoomPlates = await runSmartZoomMosaic(zoomRois, pass1StrongBoxes);
          if (!zoomPlates.length) zoomCooldown = 2;
          proposals.push(...zoomPlates);
        }
      }
    } else {
      fastFrames++;
    }

    const strongPlatedVehicleIds = new Set(
      tracks
        .filter(t =>
          (t.state === 'COLLECTING' && t.frames >= 4 && sigmoid(t.odds) >= 0.65) ||
          (t.state === 'FINALIZED' && t.result?.status === 'ACCEPTED')
        )
        .map(t => t.vehicleId)
        .filter(v => v !== null && v !== undefined)
    );
    const predictedVehicles = vtracks
      .filter(vt => {
        // Require 2+ hits before displaying low/medium-confidence candidates (prevents single-frame storefront flashes).
        if (vt.hits < 2 && vt.maxScore < 0.82) return false;
        // Suppress motionless background storefronts/kiosks/props lacking strong vehicle confidence and verified plates.
        if (
          vt.hits >= 2 &&
          vt.maxDisp < Math.max(10, 0.018 * wOrig) &&
          vt.maxScore < 0.88 &&
          !strongPlatedVehicleIds.has(vt.id)
        ) {
          return false;
        }
        return true;
      })
      .map(vt => {
        const box = predictedVehicleBox(vt, time);
        const vw = box[2] - box[0], vh = box[3] - box[1];
        const mountBox = vt.mountRel
          ? [
              clamp(box[0] + vt.mountRel[0] * vw, 0, wOrig),
              clamp(box[1] + vt.mountRel[1] * vh, 0, hOrig),
              clamp(box[0] + vt.mountRel[2] * vw, 0, wOrig),
              clamp(box[1] + vt.mountRel[3] * vh, 0, hOrig),
            ]
          : null;
        return {
          id: vt.id,
          box,
          velocity: [vt.vx, vt.vy, vt.vw, vt.vh],
          mask: vt.mask,
          score: vt.score,
          mountBox,
          mountProb: vt.mountProb,
        };
      });

    const ownerOf = box => {
      const cx = 0.5 * (box[0] + box[2]), cy = 0.5 * (box[1] + box[3]);
      let best = null, bestArea = Infinity;
      for (const v of predictedVehicles) {
        const vw = v.box[2] - v.box[0], vh = v.box[3] - v.box[1];
        const minY = v.box[1] + (vh > 56 ? 0.14 * vh : -0.02 * vh);
        if (
          cx >= v.box[0] - 0.06 * vw - 6 &&
          cx <= v.box[2] + 0.06 * vw + 6 &&
          cy >= minY &&
          cy <= v.box[3] + 0.18 * vh + 14
        ) {
          const area = vw * vh;
          if (area < bestArea) {
            bestArea = area;
            best = v;
          }
        }
      }
      return best;
    };

    // Smoothly predict active & dormant tracks and generate Stage-2 tracking search proposals.
    for (const track of active) {
      proposals.push({
        box: predictedBox(track, time),
        score: full ? 0.45 : 1.0,
        track,
        provenance: 2,
      });
    }
    for (const track of finalizedActive) {
      const box = predictedBox(track, time);
      if (!full || ownerOf(box) !== null) {
        proposals.push({
          box,
          score: full ? 0.01 : 0.9,
          track,
          provenance: 2,
        });
      }
    }
    for (const track of dormant) {
      if (time - track.dormantSince <= 0.65) {
        proposals.push({
          box: predictedBox(track, time),
          score: 0.35,
          track,
          provenance: 2,
        });
      }
    }

    // Score-free mount search on full frames for confirmed vehicles missing a plate.
    if (full) {
      let addedMounts = 0;
      for (const vt of vtracks) {
        if (vt.mountCooldown > 0) {
          vt.mountCooldown--;
          continue;
        }
        if (addedMounts >= 1) break;
        if (vt.hits < 2 || vt.mountProb < 0.42 || !vt.mountRel) continue;
        const vb = predictedVehicleBox(vt, time);
        const vw = vb[2] - vb[0], vh = vb[3] - vb[1];
        if (vw < 44 || vh < 34) continue;
        const covered = proposals.some(p => {
          const pcx = 0.5 * (p.box[0] + p.box[2]), pcy = 0.5 * (p.box[1] + p.box[3]);
          return pcx >= vb[0] - 0.06 * vw && pcx <= vb[2] + 0.06 * vw && pcy >= vb[1] && pcy <= vb[3] + 0.18 * vh;
        });
        if (covered) continue;
        const roi = [
          clamp(vb[0] + vt.mountRel[0] * vw, 0, wOrig),
          clamp(vb[1] + vt.mountRel[1] * vh, 0, hOrig),
          clamp(vb[0] + vt.mountRel[2] * vw, 0, wOrig),
          clamp(vb[1] + vt.mountRel[3] * vh, 0, hOrig),
        ];
        if (roi[2] - roi[0] >= 12 && roi[3] - roi[1] >= 6) {
          proposals.push({ box: roi, score: 0.15, provenance: 2, vehicleId: vt.id, speculative: true });
          vt.mountCooldown = 4;
          addedMounts++;
        }
      }
    }

    // Keep track-backed proposals or proposals with readable size contained within vehicles.
    // Never let Smart-Zoom (`provenance === 1`) bypass vehicle containment when vehicles exist!
    const contained = proposals.filter(({ box, score, track, provenance }) =>
      track || (
        (box[2] - box[0] >= 10 && box[3] - box[1] >= 5) && (
          provenance === 2 ||
          (predictedVehicles.length ? ownerOf(box) !== null : score >= 0.52)
        )
      )
    );

    // Link unbacked detector proposals to existing collectors before NMS so we never spawn duplicate tracks.
    for (const c of contained) {
      if (c.track) continue;
      let bestTrack = null, bestIou = 0.25;
      for (const t of tracks) {
        if (t.state !== 'COLLECTING') continue;
        const ov = iou(predictedBox(t, time), c.box);
        if (ov >= bestIou) {
          bestIou = ov;
          bestTrack = t;
        }
      }
      if (bestTrack) c.track = bestTrack;
    }

    // Suppress proposals already covered by ACCEPTED FINALIZED tracks (so terminal readout stays 1/track).
    // Notice UNREADABLE premature exits do NOT lock out future readable sightings!
    const used = new Set();
    const acceptedOwners = new Set();
    const remaining = [];
    for (const candidate of nms(contained, 16, 0.45)) {
      const cq = candidate.corners;
      const cBox = validCorners(cq) ? cornersToBox(cq) : candidate.box;
      const acceptedCover = tracks.find(t => {
        if (t.state !== 'FINALIZED' || t.result.status !== 'ACCEPTED') return false;
        const pBox = predictedBox(t, time);
        if (iou(pBox, cBox) < 0.20) return false;
        const ratio = Math.max(1, cBox[2] - cBox[0]) / Math.max(1, pBox[2] - pBox[0]);
        return ratio >= 0.45 && ratio <= 2.2;
      });
      if (acceptedCover) {
        if (!used.has(acceptedCover.id)) {
          const dt = Math.max(0.03, time - acceptedCover.last);
          const dx = 0.5 * (cBox[0] + cBox[2] - acceptedCover.box[0] - acceptedCover.box[2]);
          const dy = 0.5 * (cBox[1] + cBox[3] - acceptedCover.box[1] - acceptedCover.box[3]);
          if (validCorners(cq)) {
            acceptedCover.velocity = [
              clamp(0.6 * acceptedCover.velocity[0] + 0.4 * (dx / dt), -480, 480),
              clamp(0.6 * acceptedCover.velocity[1] + 0.4 * (dy / dt), -480, 480),
            ];
            acceptedCover.corners = cq;
          } else {
            acceptedCover.corners = acceptedCover.corners.map(([x, y]) => [x + dx, y + dy]);
          }
          acceptedCover.box = cBox;
          acceptedCover.last = time;
          acceptedCover.lastFrame = frameIndex;
          acceptedCover.missed = 0;
          used.add(acceptedCover.id);
          const owner = ownerOf(acceptedCover.box);
          if (owner) acceptedOwners.add(owner.id);
        }
        continue;
      }
      remaining.push(candidate);
    }

    // Prioritize track-backed proposals, then detector proposals; allow up to 2 candidates per unaccepted vehicle
    // into Stage-2 so a taillight never blocks the real plate before Stage-2 scores both!
    remaining.sort((a, b) => Number(Boolean(b.track)) - Number(Boolean(a.track)) || b.score - a.score);
    const confirmedPlatedVehicles = new Set(
      tracks.filter(t => t.state === 'COLLECTING' && t.frames >= 2 && t.vehicleId !== null).map(t => t.vehicleId)
    );
    const vehCandCounts = new Map();
    const claimedTrackIds = new Set();
    const selected = [];
    for (const candidate of remaining) {
      if (candidate.track) {
        if (used.has(candidate.track.id) || claimedTrackIds.has(candidate.track.id)) continue;
      }
      if (tracks.some(t => used.has(t.id) && iou(t.box, candidate.box) >= 0.18)) continue;
      if (selected.some(s => iou(s.box, candidate.box) >= 0.18)) continue;
      const owner = ownerOf(candidate.box);
      const vid = candidate.track?.vehicleId ?? owner?.id;
      if (vid !== undefined && vid !== null) {
        if (acceptedOwners.has(vid)) continue;
        const maxPerVeh = !confirmedPlatedVehicles.has(vid) && candidate.score >= 0.32 ? 2 : 1;
        const cnt = vehCandCounts.get(vid) ?? 0;
        if (cnt >= maxPerVeh) continue;
        vehCandCounts.set(vid, cnt + 1);
      }
      if (candidate.track) claimedTrackIds.add(candidate.track.id);
      selected.push(candidate);
      if (selected.length >= 6) break;
    }

    // Run batched Stage-2 refinement (NEVER reuse stale corners across frames!).
    const stage2Results = await runStage2Batch(selected);
    const recropIndices = [];
    for (let i = 0; i < stage2Results.length; i++) {
      if (stage2Results[i].needsRecrop) recropIndices.push(i);
    }
    if (recropIndices.length) {
      const recropItems = recropIndices.map(i => ({
        box: cornersToBox(stage2Results[i].corners),
        provenance: selected[i].provenance ?? 0,
        gamma: 1.45,
      }));
      const recropped = await runStage2Batch(recropItems);
      recropIndices.forEach((idx, j) => {
        if (recropped[j].valid && validCorners(recropped[j].corners)) stage2Results[idx] = recropped[j];
      });
    }

    // Process highest Stage-2 plate support first, keeping at most 1 verified plate per vehicle.
    const stage2Order = selected
      .map((_, i) => i)
      .sort((a, b) => stage2Results[b].support - stage2Results[a].support);
    const postStage2Owners = new Set(acceptedOwners);

    for (const idx of stage2Order) {
      const candidate = selected[idx];
      const loc = stage2Results[idx];
      const prior = candidate.track;
      const reqSupport = candidate.speculative ? 0.5 : minIsPlateProb;
      if (!loc.valid || loc.support < reqSupport) continue;
      const corners = loc.corners;
      if (!validCorners(corners)) continue;
      const box = cornersToBox(corners);
      const boxW = box[2] - box[0], boxH = box[3] - box[1];
      const aspect = boxW / Math.max(1, boxH);
      if (boxW < 11 || boxH < 5.5 || aspect < 0.85 || aspect > 4.6) continue;
      if (tracks.some(t => used.has(t.id) && iou(t.box, box) >= 0.15)) continue;
      const postOwner = ownerOf(box);
      if (postOwner && postStage2Owners.has(postOwner.id)) continue;

      // Associate with tracked collectors (active OR dormant) using IoU + cosine ReID + distance gate.
      let track = prior;
      if (!track || used.has(track.id) || track.state !== 'COLLECTING') {
        track = null;
        let bestAssocScore = -Infinity;
        const pcx = 0.5 * (box[0] + box[2]), pcy = 0.5 * (box[1] + box[3]);
        for (const t of tracks) {
          if (t.state !== 'COLLECTING' || used.has(t.id)) continue;
          const pBox = predictedBox(t, time);
          const ov = iou(pBox, box);
          const tcx = 0.5 * (pBox[0] + pBox[2]), tcy = 0.5 * (pBox[1] + pBox[3]);
          const lastCx = 0.5 * (t.box[0] + t.box[2]), lastCy = 0.5 * (t.box[1] + t.box[3]);
          const dtGap = Math.max(0.03, time - t.last);
          const maxJump = Math.max(2.4 * (t.box[2] - t.box[0]), 60) + 420 * dtGap;
          if (Math.hypot(pcx - lastCx, pcy - lastCy) > maxJump) continue;
          const dist = Math.hypot(pcx - tcx, pcy - tcy) / Math.max(18, pBox[2] - pBox[0]);
          const sim = dotVec(t.reidEmb, loc.reidEmb);
          if (ov >= 0.18 || (sim >= 0.65 && dist <= 0.95)) {
            const sc = 3.0 * ov + 1.4 * Math.max(0, sim) - 0.6 * dist;
            if (sc > bestAssocScore) {
              bestAssocScore = sc;
              track = t;
            }
          }
        }
      }
      if (track && used.has(track.id)) continue;
      const crop = projective(corners);
      if (!track) {
        const minBirth = predictedVehicles.length ? 0.48 : 0.68;
        if (loc.support < minBirth || loc.nc > 3.8 || isRedLampCrop(crop)) continue;
        track = newTrack(box, time);
      }
      used.add(track.id);
      track.dormant = false;
      track.dormantSince = Infinity;
      track.missed = 0;

      if (track.state !== 'COLLECTING' || track.readouts !== 0) {
        throw Error('Sealed collector cannot receive evidence');
      }

      const geom = geometry(corners, crop, track.frames ? track.corners : null);
      const encoded = await sessions.frame_encoder.run({
        crops: crop,
        plate_sides: integer(0),
        geom_meta: geom,
      });
      const damping = Math.exp(-Math.min(30, loc.nc + Math.max(0, -loc.llr))) * (0.05 + 0.95 * loc.visibility);
      for (const key of ['lambda_prec', 'lambda_color', 'pol_prec']) {
        const src = encoded[key].data;
        const scaled = new Float32Array(src.length);
        for (let k = 0; k < src.length; k++) scaled[k] = src[k] * damping;
        encoded[key] = float(scaled, encoded[key].dims);
      }
      const next = await sessions.memory_step.run({
        ...track.memory,
        ...encoded,
        geom_meta: geom,
        timestamp_s: float([time], [1]),
      });
      for (const name of Object.keys(track.memory)) track.memory[name] = next[`new_${name}`];

      const qc = clamp(loc.support, 1e-6, 1 - 1e-6);
      const increment = Math.log(qc) - Math.log1p(-qc);
      if (!track.frames) {
        track.firstSupport = loc.support;
        track.odds = increment;
        track.corners = corners;
        track.box = box;
      } else {
        const dt = Math.max(0.03, time - track.last);
        track.odds = clamp(track.odds + Math.min(1, Math.max(0.05, dt / 0.5)) * (0.05 + 0.95 * loc.visibility) * increment, -30, 30);

        // Alpha-beta Kalman update for plate center, size, and velocity.
        const oldCx = 0.5 * (track.box[0] + track.box[2]);
        const oldCy = 0.5 * (track.box[1] + track.box[3]);
        const oldW = track.box[2] - track.box[0];
        const oldH = track.box[3] - track.box[1];
        const predCx = oldCx + track.velocity[0] * dt;
        const predCy = oldCy + track.velocity[1] * dt;
        const measCx = 0.5 * (box[0] + box[2]);
        const measCy = 0.5 * (box[1] + box[3]);
        const measW = box[2] - box[0];
        const measH = box[3] - box[1];
        const alpha = 0.72, beta = track.frames < 2 ? 0.25 : 0.45;
        const rCx = measCx - predCx, rCy = measCy - predCy;
        const maxPlateV = Math.min(480, wOrig * 0.65);
        track.velocity = [
          clamp(track.velocity[0] + (beta / dt) * rCx, -maxPlateV, maxPlateV),
          clamp(track.velocity[1] + (beta / dt) * rCy, -maxPlateV, maxPlateV),
        ];
        track.sizeVelocity = [
          clamp(track.sizeVelocity[0] + (0.28 / dt) * (measW - oldW), -wOrig, wOrig),
          clamp(track.sizeVelocity[1] + (0.28 / dt) * (measH - oldH), -hOrig, hOrig),
        ];
        const smCx = predCx + alpha * rCx;
        const smCy = predCy + alpha * rCy;
        const smW = oldW + 0.65 * (measW - oldW);
        const smH = oldH + 0.65 * (measH - oldH);
        const prevPredCorners = predictedCorners(track, time);
        track.corners = corners.map((c, k) => [
          0.25 * prevPredCorners[k][0] + 0.75 * c[0],
          0.25 * prevPredCorners[k][1] + 0.75 * c[1],
        ]);
        track.box = [smCx - 0.5 * smW, smCy - 0.5 * smH, smCx + 0.5 * smW, smCy + 0.5 * smH];
      }

      if (loc.reidEmb) {
        if (!track.reidEmb) track.reidEmb = loc.reidEmb;
        else {
          const blended = new Float32Array(48);
          for (let k = 0; k < 48; k++) blended[k] = 0.75 * track.reidEmb[k] + 0.25 * loc.reidEmb[k];
          track.reidEmb = normalizeVec(blended);
        }
      }
      const owner = postOwner;
      if (owner) {
        track.vehicleId = owner.id;
        postStage2Owners.add(owner.id);
      }
      if (geom.data[6] > track.sharpness) {
        track.sharpness = geom.data[6];
        track.thumbnail = thumbnail(crop);
      }
      track.frames++;
      track.last = time;
      track.lastFrame = frameIndex;
    }

    // Move missed collectors cleanly into DORMANT instead of dropping identity immediately.
    for (const t of tracks) {
      if (t.state === 'COLLECTING' && !used.has(t.id)) {
        t.missed++;
        if (!t.dormant) {
          t.dormant = true;
          t.dormantSince = time;
        }
      }
    }

    const newlyFinalized = [...preSealedOnCut];
    for (const t of tracks) {
      if (t.state !== 'COLLECTING') continue;
      if (autoSealFrames > 0) {
        const exited = autoExitSec > 0 && time - t.last >= autoExitSec;
        if (exited && discardShortUnreadable && t.frames <= 2) {
          t.state = 'DISCARDED';
          continue;
        }
        let readyToSeal = false;
        if (requireFullFieldsForEarlySeal) {
          const pw = t.box[2] - t.box[0];
          const growing = (t.sizeVelocity?.[0] ?? 0) > 3.5;
          const allFieldsCrisp =
            t.frames >= autoSealFrames &&
            time - t.firstTime >= autoSealMinSec &&
            !growing &&
            pw >= 48 &&
            minFieldPrecision(t.memory.mem_P.data) >= 1.1;
          const longDwell =
            t.frames >= 42 &&
            time - t.firstTime >= 3.2 &&
            mean(t.memory.mem_P.data) >= autoSealMinPrecision;
          readyToSeal = allFieldsCrisp || longDwell;
        } else {
          readyToSeal =
            t.frames >= autoSealFrames &&
            time - t.firstTime >= autoSealMinSec &&
            mean(t.memory.mem_P.data) >= autoSealMinPrecision;
        }
        if (exited || readyToSeal) newlyFinalized.push(await sealTrack(t));
      }
    }
    tracks = tracks.filter(t => t.state !== 'DISCARDED');

    frameIndex++;
    return {
      timestamp: time,
      wOrig,
      hOrig,
      frameIndex,
      fullDetections,
      fastFrames,
      latencyMs: performance.now() - started,
      newlyFinalized,
      finalizedTracks: tracks
        .filter(t => t.state === 'FINALIZED' && used.has(t.id))
        .map(t => ({
          id: t.id,
          box: t.box,
          corners: t.corners,
          velocity: t.velocity,
          vehicleId: t.vehicleId,
          status: t.result.status,
          text: t.result.text,
          confidence_tier: t.result.confidence_tier,
          frames: t.result.frames,
          readouts: t.result.readouts,
        })),
      proposalCount: contained.length,
      localizedCount: selected.length,
      vehicles: predictedVehicles,
      tracks: tracks
        .filter(
          t =>
            t.state === 'COLLECTING' &&
            (used.has(t.id) || t.missed <= 2) &&
            (!discardShortUnreadable || t.frames >= 2)
        )
        .map(t => ({
          id: t.id,
          box: used.has(t.id) ? t.box : predictedBox(t, time),
          corners: used.has(t.id) ? t.corners : predictedCorners(t, time),
          velocity: t.velocity,
          vehicleId: t.vehicleId,
          coasted: !used.has(t.id),
          state: t.state,
          frames: t.frames,
          readouts: t.readouts,
        })),
    };
  }

  async function sealTrack(track) {
    if (track.state !== 'COLLECTING' || track.readouts !== 0) throw Error('Terminal capability already consumed');
    track.state = 'SEALED';
    track.readouts = 1;
    progress({ track_id: track.id, state: track.state, readouts: track.readouts });
    let decision;
    try {
      const ceiling = policy.terminal.readout_precision_ceiling;
      const precision = mean(track.memory.mem_P.data);
      if (ceiling > 0 && precision > ceiling) {
        for (const name of ['mem_h', 'mem_P']) {
          const data = track.memory[name].data;
          for (let i = 0; i < data.length; i++) data[i] *= ceiling / precision;
        }
      }
      const feed = Object.fromEntries(
        ['mem_h', 'mem_P', 'mem_conflict', 'color_h', 'color_P', 'pol_h', 'pol_P'].map(k => [k, track.memory[k]])
      );
      const readout = await sessions.memory_readout.run(feed);
      decision = terminalDecision(
        readout,
        track.memory,
        track.frames,
        track.frames === 1 ? track.firstSupport : sigmoid(track.odds),
        grammar,
        policy
      );
    } catch (error) {
      decision = { status: 'UNREADABLE', confidence_tier: 'NONE', text: null, error: error.message };
    }
    track.state = 'FINALIZED';
    track.result = {
      track_id: track.id,
      ...decision,
      frames: track.frames,
      readouts: 1,
      state: 'FINALIZED',
      thumbnail: track.thumbnail,
    };
    emittedResults.push(track.result);
    progress({ track_id: track.id, state: track.state, readouts: track.readouts });
    return track.result;
  }

  async function resegment(options = {}) {
    if (phase === 'IDLE' || phase === 'SEALED') throw Error('Collection is unavailable');
    const newlySealed = [];
    for (const t of tracks) if (t.state === 'COLLECTING' && t.frames >= 2) newlySealed.push(await sealTrack(t));
    tracks = [];
    vtracks = [];
    h3 = zero([1, 36, 56, 96]);
    h4 = zero([1, 36, 28, 48]);
    lastFull = -Infinity;
    hasScoutPrev = false;
    prevLumaMean = null;
    lastTime = -Infinity;
    zoomCooldown = 0;
    configure(options);
    phase = 'COLLECTING';
    return { finalizedBeforeSeek: newlySealed };
  }

  async function finalize() {
    if (phase !== 'COLLECTING') throw Error('Terminal capability already consumed');
    phase = 'SEALED';
    for (const t of tracks) {
      if (t.state !== 'COLLECTING') continue;
      if (discardShortUnreadable && t.frames <= 2) {
        t.state = 'DISCARDED';
        continue;
      }
      await sealTrack(t);
    }
    phase = 'FINALIZED';
    return emittedResults.slice();
  }

  async function dispose() {
    await Promise.all(Object.values(sessions).map(s => s.release()));
  }

  return { reset, process, finalize, resegment, dispose };
}

// RPC commands serialize frames, timeline changes and terminal operations.
if (typeof self !== 'undefined' && typeof self.postMessage === 'function' && typeof document === 'undefined') {
  let pipeline, queue = Promise.resolve();
  self.onmessage = ({ data }) => {
    queue = queue.then(async () => {
      try {
        let result;
        if (data.type === 'init') {
          pipeline ??= await createPipeline({
            ...data.options,
            progress: message => self.postMessage({ type: 'progress', message }),
          });
          result = {
            threads: ort.env.wasm.numThreads,
            isolated: Boolean(self.crossOriginIsolated),
          };
        } else if (data.type === 'reset') {
          pipeline.reset(data.options);
          result = true;
        } else if (data.type === 'frame') {
          let rgbaBuffer = data.rgbaBuffer;
          if (!rgbaBuffer && data.bitmap) {
            const canvas = new OffscreenCanvas(data.wOrig, data.hOrig);
            const ctx = canvas.getContext('2d', { willReadFrequently: true });
            ctx.drawImage(data.bitmap, 0, 0, data.wOrig, data.hOrig);
            data.bitmap.close();
            rgbaBuffer = ctx.getImageData(0, 0, data.wOrig, data.hOrig).data;
          }
          result = await pipeline.process({
            rgbaBuffer,
            wOrig: data.wOrig,
            hOrig: data.hOrig,
            timestamp: data.timestamp,
          });
        } else if (data.type === 'resegment') {
          result = await pipeline.resegment(data.options);
        } else if (data.type === 'finalize') {
          result = await pipeline.finalize();
        } else {
          throw Error('Unknown worker command');
        }
        self.postMessage({ id: data.id, type: data.type, result });
      } catch (error) {
        self.postMessage({ id: data.id, error: error.message });
      }
    });
  };
}
