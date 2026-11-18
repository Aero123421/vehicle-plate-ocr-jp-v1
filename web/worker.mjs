// Apache-2.0 · vehicle-plate-ocr-jp-v1 v0.6.5.
// JP7029732B2 design-around: no character projection while COLLECTING.
import * as ort from './vendor/ort.wasm.min.mjs';
import { clamp, sigmoid, unpackGrammar, terminalDecision } from './terminal.mjs';
import { extractPatchFromRgba, createOpticalRef, trackPlateOnRgba, sampleRgbaLumaBilinear, quadBasis } from './optical_tracker.mjs';

// Shared by the browser WebWorker and local Node parity checks.
export async function createPipeline({
  runtime = ort,
  readJSON,
  progress = () => {},
  autoSealFrames = 0,
  autoSealMinSec = 0,
  autoSealMinPrecision = 0.35,
  autoExitSec = 1.10,
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
  let grammar, policy, stage2Cal;
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
    stage2Cal = await json('models/stage2_int8_cal.json').catch(() => null);
  } catch (error) {
    await Promise.all(Object.values(sessions).map(s => s.release()));
    throw error;
  }

  const getStage2TauNc = policyIdx => {
    const q =
      stage2Cal?.support_quantile?.[policyIdx] ??
      stage2Cal?.policies?.[String(policyIdx)]?.support_quantile ??
      stage2Cal?.policies?.[policyIdx]?.support_quantile;
    return Number.isFinite(q) && q > 0 ? q : 1.65;
  };

  const baseHeartbeat = policy?.pipeline?.heartbeat_interval ?? 6;
  const maxHeartbeat = policy?.pipeline?.max_heartbeat_interval ?? 12;
  const plateThresh = policy?.pipeline?.plate_thresh ?? 0.18;
  const weakPlateThresh = 0.08;
  const vehLowThresh = 0.20;
  const vehHighThresh = 0.34;
  const vehBirthThresh = 0.36;
  const minIsPlateProb = policy?.pipeline?.min_is_plate_prob ?? 0.34;
  const vehLowLogitMin = Math.log(vehLowThresh / (1 - vehLowThresh));

  const LB_W = 768, LB_H = 448;
  const letterboxBuf = new Float32Array(3 * LB_W * LB_H);
  const zoomFloatBuf = new Float32Array(3 * LB_W * LB_H);
  const fovea1Buf = new Float32Array(3 * 96 * 160);
  const surround1Buf = new Float32Array(3 * 64 * 96);
  const policy1Buf = new BigInt64Array(1);
  const zeroH3 = zero([1, 36, 56, 96]);
  const zeroH4 = zero([1, 36, 28, 48]);
  const cropBuf = new Float32Array(3 * 80 * 160);
  const srOutBuf = new Float32Array(3 * 80 * 160);
  const srWarpLumaBuf = new Float32Array(40 * 80);
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
  let lastFullTime = -Infinity;
  let lastDetIntervalSec = 0.12;
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
    lastFullTime = -Infinity;
    lastDetIntervalSec = 0.12;
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

  function iomin(a, b) {
    const ix = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
    const iy = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
    const inter = ix * iy;
    if (inter <= 0) return 0;
    const areaA = Math.max(1, (a[2] - a[0]) * (a[3] - a[1]));
    const areaB = Math.max(1, (b[2] - b[0]) * (b[3] - b[1]));
    return inter / Math.max(1, Math.min(areaA, areaB));
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

  function sampleBicubic(src, sw, sh, nx, ny, border, out, idx, stride) {
    if (nx < 0 || nx > sw - 1 || ny < 0 || ny > sh - 1) {
      out[idx] = border[0];
      out[idx + stride] = border[1];
      out[idx + 2 * stride] = border[2];
      return;
    }
    const ix = Math.floor(nx), iy = Math.floor(ny);
    const tx = nx - ix, ty = ny - iy;
    const tx2 = tx * tx, tx3 = tx2 * tx;
    const ty2 = ty * ty, ty3 = ty2 * ty;
    const wx0 = -0.75 * tx3 + 1.5 * tx2 - 0.75 * tx;
    const wx1 = 1.25 * tx3 - 2.25 * tx2 + 1.0;
    const wx2 = -1.25 * tx3 + 1.5 * tx2 + 0.75 * tx;
    const wx3 = 0.75 * tx3 - 0.75 * tx2;
    const wy0 = -0.75 * ty3 + 1.5 * ty2 - 0.75 * ty;
    const wy1 = 1.25 * ty3 - 2.25 * ty2 + 1.0;
    const wy2 = -1.25 * ty3 + 1.5 * ty2 + 0.75 * ty;
    const wy3 = 0.75 * ty3 - 0.75 * ty2;
    const wm1 = sw - 1, hm1 = sh - 1;
    const x0 = Math.max(0, ix - 1), x1 = Math.max(0, Math.min(wm1, ix));
    const x2 = Math.min(wm1, ix + 1), x3 = Math.min(wm1, ix + 2);
    const wyArr = [wy0, wy1, wy2, wy3];
    let r = 0, g = 0, b = 0;
    for (let m = 0; m < 4; m++) {
      const yy = Math.max(0, Math.min(hm1, iy - 1 + m)) * sw;
      const wy = wyArr[m];
      const p0 = (yy + x0) * 4, p1 = (yy + x1) * 4, p2 = (yy + x2) * 4, p3 = (yy + x3) * 4;
      r += wy * (wx0 * src[p0] + wx1 * src[p1] + wx2 * src[p2] + wx3 * src[p3]);
      g += wy * (wx0 * src[p0 + 1] + wx1 * src[p1 + 1] + wx2 * src[p2 + 1] + wx3 * src[p3 + 1]);
      b += wy * (wx0 * src[p0 + 2] + wx1 * src[p1 + 2] + wx2 * src[p2 + 2] + wx3 * src[p3 + 2]);
    }
    const inv255 = 1 / 255;
    out[idx] = clamp(r * inv255, 0, 1);
    out[idx + stride] = clamp(g * inv255, 0, 1);
    out[idx + 2 * stride] = clamp(b * inv255, 0, 1);
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
    const eff = 1.85 * gamma;
    const rw = Math.max(bw * eff, bh * eff * 160 / 96, 24 * gamma);
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
    const data = new Float32Array(3 * 80 * 160);
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

  const HAD_SIGNS = [
    [1, -1, -1, 1],
    [1, 1, -1, -1],
    [1, 1, 1, 1],
    [1, -1, 1, -1],
  ];

  function cornersToHadamard(q) {
    const m = new Float64Array(8);
    for (let k = 0; k < 4; k++) {
      let sx = 0, sy = 0;
      for (let c = 0; c < 4; c++) {
        const s = HAD_SIGNS[c][k];
        sx += s * q[c][0];
        sy += s * q[c][1];
      }
      m[k * 2] = 0.25 * sx;
      m[k * 2 + 1] = 0.25 * sy;
    }
    return m;
  }

  function hadamardToCorners(m) {
    const out = [];
    for (let c = 0; c < 4; c++) {
      let x = 0, y = 0;
      for (let k = 0; k < 4; k++) {
        const s = HAD_SIGNS[c][k];
        x += s * m[k * 2];
        y += s * m[k * 2 + 1];
      }
      out.push([x, y]);
    }
    return out;
  }

  function cornerCovToHadamard8x8(cornerCov, wFallback) {
    const R = new Float64Array(64);
    const fbVar = Math.max(0.25, (0.025 * Math.max(12, wFallback)) ** 2);
    for (let a = 0; a < 4; a++) {
      for (let b = 0; b < 4; b++) {
        let s00 = 0, s01 = 0, s11 = 0;
        for (let c = 0; c < 4; c++) {
          const prod = HAD_SIGNS[c][a] * HAD_SIGNS[c][b];
          const blk = cornerCov?.[c];
          const c00 = blk ? blk[0] : fbVar;
          const c01 = blk ? blk[1] : 0;
          const c11 = blk ? blk[2] : fbVar;
          s00 += prod * c00;
          s01 += prod * c01;
          s11 += prod * c11;
        }
        const r0 = a * 2, c0 = b * 2;
        R[r0 * 8 + c0] = 0.0625 * s00;
        R[r0 * 8 + c0 + 1] = 0.0625 * s01;
        R[(r0 + 1) * 8 + c0] = 0.0625 * s01;
        R[(r0 + 1) * 8 + c0 + 1] = 0.0625 * s11;
      }
    }
    for (let i = 0; i < 8; i++) R[i * 8 + i] += 0.04;
    return R;
  }

  function invert8x8(mat) {
    const a = Float64Array.from(mat);
    const inv = new Float64Array(64);
    for (let i = 0; i < 8; i++) {
      a[i * 8 + i] += 1e-6;
      inv[i * 8 + i] = 1;
    }
    for (let k = 0; k < 8; k++) {
      let p = k;
      let maxAbs = Math.abs(a[k * 8 + k]);
      for (let i = k + 1; i < 8; i++) {
        const v = Math.abs(a[i * 8 + k]);
        if (v > maxAbs) { maxAbs = v; p = i; }
      }
      if (maxAbs < 1e-11) return null;
      if (p !== k) {
        for (let j = 0; j < 8; j++) {
          const ta = a[k * 8 + j]; a[k * 8 + j] = a[p * 8 + j]; a[p * 8 + j] = ta;
          const ti = inv[k * 8 + j]; inv[k * 8 + j] = inv[p * 8 + j]; inv[p * 8 + j] = ti;
        }
      }
      const invDiag = 1 / a[k * 8 + k];
      for (let j = 0; j < 8; j++) {
        a[k * 8 + j] *= invDiag;
        inv[k * 8 + j] *= invDiag;
      }
      for (let i = 0; i < 8; i++) {
        if (i === k) continue;
        const f = a[i * 8 + k];
        if (f === 0) continue;
        for (let j = 0; j < 8; j++) {
          a[i * 8 + j] -= f * a[k * 8 + j];
          inv[i * 8 + j] -= f * inv[k * 8 + j];
        }
      }
    }
    return inv;
  }

  function stepHadamardFilter(track, rawCorners, cornerCov, dt) {
    const zM = cornersToHadamard(rawCorners);
    const wObs = Math.max(6, 2 * Math.hypot(zM[2], zM[3]));
    if (!track.hadModes || !track.hadCov) {
      track.hadModes = zM;
      track.hadCov = cornerCovToHadamard8x8(cornerCov, wObs);
      track.hadVel = new Float64Array(6);
      track.hadValidShape = zM.slice(2, 8);
      return rawCorners;
    }
    const prevM = track.hadModes;
    const wRef = Math.max(8, 2 * Math.hypot(prevM[2], prevM[3]));
    const dtClamped = clamp(dt, 0.01, 0.5);
    const mPred = new Float64Array(8);
    mPred[0] = prevM[0] + track.hadVel[0] * dtClamped;
    mPred[1] = prevM[1] + track.hadVel[1] * dtClamped;
    mPred[2] = prevM[2] + track.hadVel[2] * dtClamped * 0.35;
    mPred[3] = prevM[3] + track.hadVel[3] * dtClamped * 0.35;
    mPred[4] = prevM[4] + track.hadVel[4] * dtClamped * 0.35;
    mPred[5] = prevM[5] + track.hadVel[5] * dtClamped * 0.35;
    mPred[6] = prevM[6];
    mPred[7] = prevM[7];

    const q0 = Math.max(0.85, (0.18 * wRef) ** 2 * (dtClamped / 0.05));
    const jumpSq = (zM[0] - mPred[0]) ** 2 + (zM[1] - mPred[1]) ** 2;
    const shock0 = jumpSq > (0.24 * wRef) ** 2 ? jumpSq : 0;
    const qDiag = [
      q0 + shock0, q0 + shock0,
      0.032 * q0, 0.032 * q0,
      0.032 * q0, 0.032 * q0,
      0.008 * q0, 0.008 * q0,
    ];
    const Ppred = Float64Array.from(track.hadCov);
    for (let i = 0; i < 8; i++) Ppred[i * 8 + i] += qDiag[i];

    const Rm = cornerCovToHadamard8x8(cornerCov, wRef);
    const S = new Float64Array(64);
    for (let i = 0; i < 64; i++) S[i] = Ppred[i] + Rm[i];
    const Sinv = invert8x8(S);
    if (!Sinv) {
      track.hadModes = zM;
      track.hadCov = Rm;
      return rawCorners;
    }

    const K = new Float64Array(64);
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        let sum = 0;
        for (let l = 0; l < 8; l++) sum += Ppred[i * 8 + l] * Sinv[l * 8 + j];
        K[i * 8 + j] = sum;
      }
    }

    const mPost = new Float64Array(8);
    for (let i = 0; i < 8; i++) {
      let corr = 0;
      for (let j = 0; j < 8; j++) corr += K[i * 8 + j] * (zM[j] - mPred[j]);
      mPost[i] = mPred[i] + corr;
    }

    // Joseph-form covariance update: (I - K) P_pred (I - K)^T + K R_m K^T
    const ImK = new Float64Array(64);
    for (let i = 0; i < 64; i++) ImK[i] = -K[i];
    for (let i = 0; i < 8; i++) ImK[i * 8 + i] += 1;
    const tmpP = new Float64Array(64);
    const tmpR = new Float64Array(64);
    for (let i = 0; i < 8; i++) {
      for (let j = 0; j < 8; j++) {
        let sp = 0, sr = 0;
        for (let l = 0; l < 8; l++) {
          sp += ImK[i * 8 + l] * Ppred[l * 8 + j];
          sr += K[i * 8 + l] * Rm[l * 8 + j];
        }
        tmpP[i * 8 + j] = sp;
        tmpR[i * 8 + j] = sr;
      }
    }
    const Ppost = new Float64Array(64);
    for (let i = 0; i < 8; i++) {
      for (let j = i; j < 8; j++) {
        let val = 0;
        for (let l = 0; l < 8; l++) val += tmpP[i * 8 + l] * ImK[j * 8 + l] + tmpR[i * 8 + l] * K[j * 8 + l];
        Ppost[i * 8 + j] = val;
        Ppost[j * 8 + i] = val;
      }
    }

    const maxV = Math.min(480, wOrig * 0.65);
    const beta0 = track.frames < 2 ? 0.28 : 0.45;
    track.hadVel[0] = clamp(track.hadVel[0] + (beta0 / dtClamped) * (zM[0] - mPred[0]), -maxV, maxV);
    track.hadVel[1] = clamp(track.hadVel[1] + (beta0 / dtClamped) * (zM[1] - mPred[1]), -maxV, maxV);
    for (let d = 0; d < 4; d++) {
      track.hadVel[2 + d] = clamp(
        0.72 * track.hadVel[2 + d] + 0.28 * ((mPost[2 + d] - prevM[2 + d]) / dtClamped),
        -wOrig * 0.4,
        wOrig * 0.4
      );
    }
    track.hadModes = mPost;
    track.hadCov = Ppost;

    const qHad = hadamardToCorners(mPost);
    if (validCorners(qHad)) {
      track.hadValidShape = mPost.slice(2, 8);
      return qHad;
    }
    if (track.hadValidShape) {
      const safeM = new Float64Array(8);
      safeM[0] = mPost[0];
      safeM[1] = mPost[1];
      safeM.set(track.hadValidShape, 2);
      const qSafe = hadamardToCorners(safeM);
      if (validCorners(qSafe)) return qSafe;
    }
    return rawCorners;
  }

  // Low-pass 2x-upsampled ZNCC translation lock for < 36 px low-res plates (suppresses H.264 block jitter).
  const LR_W = 48, LR_H = 24, LR_N = LR_W * LR_H;
  const lrRawBuf = new Float32Array(LR_N);
  const lrTmpBuf = new Float32Array(LR_N);
  const lrBlurBuf = new Float32Array(LR_N);

  function sampleLowPass2x(corners, dx, dy, outNorm) {
    const b = quadBasis(corners);
    const span = 1.12;
    let idx = 0;
    for (let j = 0; j < LR_H; j++) {
      const v = span * ((j + 0.5) / LR_H - 0.5);
      for (let i = 0; i < LR_W; i++) {
        const u = span * ((i + 0.5) / LR_W - 0.5);
        const px = b.mx + dx + u * b.aux + v * b.avx + u * v * b.auvx;
        const py = b.my + dy + u * b.auy + v * b.avy + u * v * b.auvy;
        lrRawBuf[idx++] = sampleRgbaLumaBilinear(frame, wOrig, hOrig, px, py);
      }
    }
    for (let j = 0; j < LR_H; j++) {
      const r = j * LR_W;
      for (let i = 0; i < LR_W; i++) {
        const im = i > 0 ? i - 1 : 0, ip = i + 1 < LR_W ? i + 1 : LR_W - 1;
        lrTmpBuf[r + i] = 0.25 * (lrRawBuf[r + im] + 2 * lrRawBuf[r + i] + lrRawBuf[r + ip]);
      }
    }
    let sum = 0;
    for (let j = 0; j < LR_H; j++) {
      const rm = (j > 0 ? j - 1 : 0) * LR_W;
      const rc = j * LR_W;
      const rp = (j + 1 < LR_H ? j + 1 : LR_H - 1) * LR_W;
      for (let i = 0; i < LR_W; i++) {
        const v = 0.25 * (lrTmpBuf[rm + i] + 2 * lrTmpBuf[rc + i] + lrTmpBuf[rp + i]);
        lrBlurBuf[rc + i] = v;
        sum += v;
      }
    }
    const mu = sum / LR_N;
    let sq = 0;
    for (let i = 0; i < LR_N; i++) {
      const d = lrBlurBuf[i] - mu;
      outNorm[i] = d;
      sq += d * d;
    }
    if (sq < 4.0 * LR_N) return false;
    const inv = 1 / Math.sqrt(sq);
    for (let i = 0; i < LR_N; i++) outNorm[i] *= inv;
    return true;
  }

  function createLowResRef2x(corners) {
    const tpl = new Float32Array(LR_N);
    if (!sampleLowPass2x(corners, 0, 0, tpl)) return null;
    return { tpl };
  }

  const lrCandBuf = new Float32Array(LR_N);
  function trackLowResTranslation2x(ref2x, corners, minZncc = 0.48) {
    if (!ref2x?.tpl) return { ok: false, zncc: -1, dx: 0, dy: 0 };
    const tpl = ref2x.tpl;
    const evalShift = (dx, dy) => {
      if (!sampleLowPass2x(corners, dx, dy, lrCandBuf)) return -1;
      let dot = 0;
      for (let i = 0; i < LR_N; i++) dot += lrCandBuf[i] * tpl[i];
      return dot;
    };
    let bestDx = 0, bestDy = 0, bestZ = evalShift(0, 0), bestPen = bestZ;
    const rad = 2.4, step = 0.8;
    for (let iy = -2; iy <= 2; iy++) {
      const dy = iy * 0.6;
      for (let ix = -3; ix <= 3; ix++) {
        if (ix === 0 && iy === 0) continue;
        const dx = ix * step;
        const z = evalShift(dx, dy);
        const pen = z - 0.03 * ((dx / rad) ** 2 + (dy / 1.5) ** 2);
        if (pen > bestPen) {
          bestPen = pen;
          bestZ = z;
          bestDx = dx;
          bestDy = dy;
        }
      }
    }
    const baseDx = bestDx, baseDy = bestDy;
    const fStep = 0.25;
    const zGrid = new Float64Array(9);
    for (let iy = -1; iy <= 1; iy++) {
      for (let ix = -1; ix <= 1; ix++) {
        const dx = baseDx + ix * fStep, dy = baseDy + iy * fStep;
        const z = evalShift(dx, dy);
        zGrid[(iy + 1) * 3 + (ix + 1)] = z;
        if (z > bestZ) {
          bestZ = z;
          bestDx = dx;
          bestDy = dy;
        }
      }
    }
    // Parabolic sub-pixel peak around center of 3x3 if center is peak
    const zc = zGrid[4];
    if (zc >= zGrid[3] && zc >= zGrid[5]) {
      const denomX = zGrid[3] - 2 * zc + zGrid[5];
      if (denomX < -1e-5) bestDx = baseDx + clamp(0.5 * (zGrid[3] - zGrid[5]) / denomX, -0.5, 0.5) * fStep;
    }
    if (zc >= zGrid[1] && zc >= zGrid[7]) {
      const denomY = zGrid[1] - 2 * zc + zGrid[7];
      if (denomY < -1e-5) bestDy = baseDy + clamp(0.5 * (zGrid[1] - zGrid[7]) / denomY, -0.5, 0.5) * fStep;
    }
    const finalZ = evalShift(bestDx, bestDy);
    if (finalZ < minZncc) return { ok: false, zncc: finalZ, dx: 0, dy: 0 };
    return { ok: true, zncc: finalZ, dx: bestDx, dy: bestDy };
  }

  function solveHomographyPtsToDst(srcCorners, dstCorners) {
    const matrix = [], rhs = [];
    for (let i = 0; i < 4; i++) {
      const [x, y] = srcCorners[i];
      const [u, v] = dstCorners[i];
      matrix.push([x, y, 1, 0, 0, 0, -x * u, -y * u]); rhs.push(u);
      matrix.push([0, 0, 0, x, y, 1, -x * v, -y * v]); rhs.push(v);
    }
    for (let k = 0; k < 8; k++) {
      let p = k;
      for (let i = k + 1; i < 8; i++) if (Math.abs(matrix[i][k]) > Math.abs(matrix[p][k])) p = i;
      [matrix[k], matrix[p]] = [matrix[p], matrix[k]];
      [rhs[k], rhs[p]] = [rhs[p], rhs[k]];
      const d = matrix[k][k];
      if (Math.abs(d) < 1e-10) return null;
      for (let j = k; j < 8; j++) matrix[k][j] /= d;
      rhs[k] /= d;
      for (let i = 0; i < 8; i++) {
        if (i === k) continue;
        const f = matrix[i][k];
        for (let j = k; j < 8; j++) matrix[i][j] -= f * matrix[k][j];
        rhs[i] -= f * rhs[k];
      }
    }
    return rhs;
  }

  const CANON_DST_160 = [[7.5, 3.5], [151.5, 3.5], [151.5, 75.5], [7.5, 75.5]];

  function updateDrizzleRefTables(track, luma40x80) {
    track.srRefLuma.set(luma40x80);
    const N = 3200;
    let sum = 0;
    for (let i = 0; i < N; i++) sum += luma40x80[i];
    const mu = sum / N;
    let sq = 0;
    const tpl = track.srTpl ?? (track.srTpl = new Float32Array(N));
    for (let i = 0; i < N; i++) {
      const d = luma40x80[i] - mu;
      tpl[i] = d;
      sq += d * d;
    }
    const std = Math.sqrt(sq / N);
    if (std < 0.012) {
      track.srInvH = null;
      return;
    }
    const invStd = 1 / std;
    for (let i = 0; i < N; i++) tpl[i] *= invStd;
    const jac = track.srJac ?? (track.srJac = new Float32Array(N * 4));
    const H = new Float64Array(16);
    for (let j = 0; j < 40; j++) {
      const ym = (j > 0 ? j - 1 : 0) * 80;
      const yp = (j + 1 < 40 ? j + 1 : 39) * 80;
      const yr = j * 80;
      const yc = (j + 0.5) / 40 - 0.5;
      for (let i = 0; i < 80; i++) {
        const xm = i > 0 ? i - 1 : 0;
        const xp = i + 1 < 80 ? i + 1 : 79;
        const xc = (i + 0.5) / 80 - 0.5;
        const gx = (i > 0 && i < 79 ? 0.5 : 1.0) * (tpl[yr + xp] - tpl[yr + xm]);
        const gy = (j > 0 && j < 39 ? 0.5 : 1.0) * (tpl[yp + i] - tpl[ym + i]);
        const j0 = gx, j1 = gy, j2 = gx * xc, j3 = gy * yc;
        const off = (yr + i) * 4;
        jac[off] = j0; jac[off + 1] = j1; jac[off + 2] = j2; jac[off + 3] = j3;
        H[0] += j0 * j0; H[1] += j0 * j1; H[2] += j0 * j2; H[3] += j0 * j3;
        H[5] += j1 * j1; H[6] += j1 * j2; H[7] += j1 * j3;
        H[10] += j2 * j2; H[11] += j2 * j3;
        H[15] += j3 * j3;
      }
    }
    H[4] = H[1]; H[8] = H[2]; H[12] = H[3];
    H[9] = H[6]; H[13] = H[7]; H[14] = H[11];
    H[0] += 2.0; H[5] += 2.0; H[10] += 80.0; H[15] += 60.0;
    // Invert 4x4 H
    const a = Float64Array.from(H);
    const inv = new Float64Array(16);
    inv[0] = 1; inv[5] = 1; inv[10] = 1; inv[15] = 1;
    for (let k = 0; k < 4; k++) {
      let p = k;
      for (let i = k + 1; i < 4; i++) if (Math.abs(a[i * 4 + k]) > Math.abs(a[p * 4 + k])) p = i;
      if (Math.abs(a[p * 4 + k]) < 1e-9) { track.srInvH = null; return; }
      if (p !== k) {
        for (let j = 0; j < 4; j++) {
          const ta = a[k * 4 + j]; a[k * 4 + j] = a[p * 4 + j]; a[p * 4 + j] = ta;
          const ti = inv[k * 4 + j]; inv[k * 4 + j] = inv[p * 4 + j]; inv[p * 4 + j] = ti;
        }
      }
      const d = a[k * 4 + k];
      for (let j = 0; j < 4; j++) { a[k * 4 + j] /= d; inv[k * 4 + j] /= d; }
      for (let i = 0; i < 4; i++) {
        if (i === k) continue;
        const f = a[i * 4 + k];
        for (let j = 0; j < 4; j++) { a[i * 4 + j] -= f * a[k * 4 + j]; inv[i * 4 + j] -= f * inv[k * 4 + j]; }
      }
    }
    track.srInvH = inv;
  }

  function alignCornersToDrizzleRef(track, corners) {
    if (!track.srInvH || !track.srTpl || !track.srJac) return { zncc: -1, corners };
    const b = quadBasis(corners);
    const tpl = track.srTpl, jac = track.srJac, invH = track.srInvH;
    const N = 3200;

    const sampleWarp40 = (tu, tv, su, sv) => {
      const u0 = tu - 0.5 * su, u1 = tu + 0.5 * su;
      const v0 = tv - 0.5 * sv, v1 = tv + 0.5 * sv;
      const q = [
        [b.mx + u0 * b.aux + v0 * b.avx + u0 * v0 * b.auvx, b.my + u0 * b.auy + v0 * b.avy + u0 * v0 * b.auvy],
        [b.mx + u1 * b.aux + v0 * b.avx + u1 * v0 * b.auvx, b.my + u1 * b.auy + v0 * b.avy + u1 * v0 * b.auvy],
        [b.mx + u1 * b.aux + v1 * b.avx + u1 * v1 * b.auvx, b.my + u1 * b.auy + v1 * b.avy + u1 * v1 * b.auvy],
        [b.mx + u0 * b.aux + v1 * b.avx + u0 * v1 * b.auvx, b.my + u0 * b.auy + v1 * b.avy + u0 * v1 * b.auvy],
      ];
      // Canonical (u_c, v_c) -> sensor homography
      const hC2S = solveHomographyPtsToDst([[0, 0], [1, 0], [1, 1], [0, 1]], q);
      if (!hC2S) return null;
      let idx = 0;
      for (let j = 0; j < 40; j++) {
        const vc = ((j + 0.5) / 40 - 0.05) / 0.9;
        for (let i = 0; i < 80; i++) {
          const uc = ((i + 0.5) / 80 - 0.05) / 0.9;
          const z = hC2S[6] * uc + hC2S[7] * vc + 1;
          const nx = (hC2S[0] * uc + hC2S[1] * vc + hC2S[2]) / z;
          const ny = (hC2S[3] * uc + hC2S[4] * vc + hC2S[5]) / z;
          srWarpLumaBuf[idx++] = sampleRgbaLumaBilinear(frame, wOrig, hOrig, nx, ny) / 255;
        }
      }
      return q;
    };

    const evalZ = () => {
      let sum = 0;
      for (let i = 0; i < N; i++) sum += srWarpLumaBuf[i];
      const mu = sum / N;
      let sq = 0, dot = 0;
      for (let i = 0; i < N; i++) {
        const d = srWarpLumaBuf[i] - mu;
        sq += d * d;
        dot += d * tpl[i];
      }
      return sq > 1e-4 * N ? dot / Math.sqrt(sq * N) : -1;
    };

    let bestTu = 0, bestTv = 0, bestZ = -1;
    const stepU = Math.min(0.08, 1.2 / Math.max(10, b.width));
    const stepV = Math.min(0.10, 1.0 / Math.max(5, b.height));
    for (let iv = -1; iv <= 1; iv++) {
      for (let iu = -1; iu <= 1; iu++) {
        if (!sampleWarp40(iu * stepU, iv * stepV, 1, 1)) continue;
        const z = evalZ();
        if (z > bestZ) { bestZ = z; bestTu = iu * stepU; bestTv = iv * stepV; }
      }
    }
    if (bestZ < 0.32) return { zncc: bestZ, corners };
    let tu = bestTu, tv = bestTv, su = 1, sv = 1;
    for (let iter = 0; iter < 4; iter++) {
      if (!sampleWarp40(tu, tv, su, sv)) return { zncc: -1, corners };
      let sum = 0;
      for (let i = 0; i < N; i++) sum += srWarpLumaBuf[i];
      const mu = sum / N;
      let sq = 0;
      for (let i = 0; i < N; i++) {
        const d = srWarpLumaBuf[i] - mu;
        srWarpLumaBuf[i] = d;
        sq += d * d;
      }
      const std = Math.sqrt(sq / N);
      if (std < 0.01) return { zncc: -1, corners };
      const invStd = 1 / std;
      let b0 = 0, b1 = 0, b2 = 0, b3 = 0;
      for (let i = 0; i < N; i++) {
        const err = srWarpLumaBuf[i] * invStd - tpl[i];
        const off = i * 4;
        b0 += jac[off] * err;
        b1 += jac[off + 1] * err;
        b2 += jac[off + 2] * err;
        b3 += jac[off + 3] * err;
      }
      const dxG = clamp(invH[0] * b0 + invH[1] * b1 + invH[2] * b2 + invH[3] * b3, -2.5, 2.5);
      const dyG = clamp(invH[4] * b0 + invH[5] * b1 + invH[6] * b2 + invH[7] * b3, -2.0, 2.0);
      const dSu = clamp(invH[8] * b0 + invH[9] * b1 + invH[10] * b2 + invH[11] * b3, -0.06, 0.06);
      const dSv = clamp(invH[12] * b0 + invH[13] * b1 + invH[14] * b2 + invH[15] * b3, -0.06, 0.06);
      tu -= su * (dxG / 72.0);
      tv -= sv * (dyG / 36.0);
      su = clamp(su / (1 + dSu / 80.0), 0.85, 1.18);
      sv = clamp(sv / (1 + dSv / 40.0), 0.85, 1.18);
      if (Math.abs(dxG) < 0.03 && Math.abs(dyG) < 0.03) break;
    }
    const qFinal = sampleWarp40(tu, tv, su, sv);
    if (!qFinal) return { zncc: -1, corners };
    return { zncc: evalZ(), corners: qFinal };
  }

  const srPool40Buf = new Float32Array(3200);
  function poolLuma40x80(rgb160Data, out40) {
    const stride = 12800;
    for (let cy = 0; cy < 40; cy++) {
      const r0 = (cy * 2) * 160, r1 = r0 + 160;
      const outRow = cy * 80;
      for (let cx = 0; cx < 80; cx++) {
        const c0 = cx * 2, c1 = c0 + 1;
        const i00 = r0 + c0, i01 = r0 + c1, i10 = r1 + c0, i11 = r1 + c1;
        const l00 = 0.299 * rgb160Data[i00] + 0.587 * rgb160Data[stride + i00] + 0.114 * rgb160Data[2 * stride + i00];
        const l01 = 0.299 * rgb160Data[i01] + 0.587 * rgb160Data[stride + i01] + 0.114 * rgb160Data[2 * stride + i01];
        const l10 = 0.299 * rgb160Data[i10] + 0.587 * rgb160Data[stride + i10] + 0.114 * rgb160Data[2 * stride + i10];
        const l11 = 0.299 * rgb160Data[i11] + 0.587 * rgb160Data[stride + i11] + 0.114 * rgb160Data[2 * stride + i11];
        out40[outRow + cx] = 0.25 * (l00 + l01 + l10 + l11);
      }
    }
  }

  function stepTrackDrizzle(track, initCorners) {
    const wt = 0.5 * (
      Math.hypot(initCorners[1][0] - initCorners[0][0], initCorners[1][1] - initCorners[0][1]) +
      Math.hypot(initCorners[2][0] - initCorners[3][0], initCorners[2][1] - initCorners[3][1])
    );
    track.srEffViews = wt < 56 ? Math.min(16, (track.srEffViews ?? 0) + 1) : 0;
    return {
      corners: initCorners,
      crop: projective(initCorners),
      srActive: wt < 56 && track.srEffViews >= 2,
    };
  }

  function newTrack(box, corners, cornerCov, time) {
    if (tracks.length >= 256) throw Error('トラック上限です。終了して読み出してください。');
    const initCorners = corners ?? [[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]];
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
      corners: initCorners.map(([x, y]) => [x, y]),
      measBox: box.slice(),
      measCorners: initCorners.map(([x, y]) => [x, y]),
      prevMeasCorners: null,
      hadModes: null,
      hadCov: null,
      hadVel: null,
      hadValidShape: null,
      lowResRef: null,
      srNum: null,
      srDen: null,
      srRefLuma: null,
      srEffViews: 0,
      srWidthEma: 0,
      bestSrViews: 0,
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
      maxLlr: -Infinity,
      firstSupport: 0,
      sharpness: -Infinity,
      velocity: [0, 0],
      sizeVelocity: [0, 0],
      reidEmb: null,
      optRef: null,
      optPatch: null,
      curOpt: null,
      vehicleId: null,
      missed: 0,
    };
    stepHadamardFilter(track, initCorners, cornerCov, 0.05);
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

  function predictedBox(track, time, useDisplay = false) {
    const dt = clamp(time - track.last, 0, 0.45);
    const decay = track.state === 'FINALIZED' ? 0.5 * Math.exp(-1.5 * dt) : (track.missed > 0 ? Math.exp(-2.2 * dt) : 1);
    const [vx, vy] = track.velocity;
    const [vw, vh] = (track.state === 'FINALIZED' || track.missed > 0) ? [0, 0] : (track.sizeVelocity ?? [0, 0]);
    const srcBox = useDisplay ? track.box : (track.measBox ?? track.box);
    const [a, b, c, d] = srcBox;
    const cx = 0.5 * (a + c) + vx * dt * decay;
    const cy = 0.5 * (b + d) + vy * dt * decay;
    const w = Math.max(5, (c - a) + vw * dt * 0.35 * decay);
    const h = Math.max(2.8, (d - b) + vh * dt * 0.35 * decay);
    return [cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h];
  }

  function predictedCorners(track, time, useDisplay = false) {
    const dt = clamp(time - track.last, 0, 0.45);
    const decay = track.state === 'FINALIZED' ? 0.5 * Math.exp(-1.5 * dt) : (track.missed > 0 ? Math.exp(-2.2 * dt) : 1);
    const dx = track.velocity[0] * dt * decay;
    const dy = track.velocity[1] * dt * decay;
    const [vw, vh] = (track.state === 'FINALIZED' || track.missed > 0) ? [0, 0] : (track.sizeVelocity ?? [0, 0]);
    const srcBox = useDisplay ? track.box : (track.measBox ?? track.box);
    const srcCorners = useDisplay ? track.corners : (track.measCorners ?? track.corners);
    const bw = Math.max(6, srcBox[2] - srcBox[0]);
    const bh = Math.max(3, srcBox[3] - srcBox[1]);
    const sx = clamp(1 + (vw / bw) * dt * 0.45 * decay, 0.84, 1.26);
    const sy = clamp(1 + (vh / bh) * dt * 0.45 * decay, 0.84, 1.26);
    const mx = 0.25 * (srcCorners[0][0] + srcCorners[1][0] + srcCorners[2][0] + srcCorners[3][0]);
    const my = 0.25 * (srcCorners[0][1] + srcCorners[1][1] + srcCorners[2][1] + srcCorners[3][1]);
    return srcCorners.map(([x, y]) => [mx + (x - mx) * sx + dx, my + (y - my) * sy + dy]);
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
    const applyVehicleMatch = (vt, d) => {
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
      if (d.emb && vt.emb) {
        const blended = new Float32Array(d.emb.length);
        for (let i = 0; i < blended.length; i++) blended[i] = 0.72 * vt.emb[i] + 0.28 * d.emb[i];
        vt.emb = normalizeVec(blended);
      } else if (d.emb && !vt.emb) {
        vt.emb = d.emb;
      }
      vt.last = time;
      vt.lastFrame = frameIndex;
      vt.hits++;
      vt.missed = 0;
      d.vtRef = vt;
    };

    const matchedTrack = new Set();
    const matchedDet = new Set();
    const predBoxes = vtracks.map(vt => predictedVehicleBox(vt, time));

    // ByteTrack Pass 1: associate high-confidence vehicle detections (score >= 0.36).
    const pairs1 = [];
    for (let ti = 0; ti < vtracks.length; ti++) {
      const vt = vtracks[ti];
      const pbox = predBoxes[ti];
      const pcx = 0.5 * (pbox[0] + pbox[2]);
      const pcy = 0.5 * (pbox[1] + pbox[3]);
      const diag = Math.max(32, Math.hypot(pbox[2] - pbox[0], pbox[3] - pbox[1]));
      for (let di = 0; di < dets.length; di++) {
        const d = dets[di];
        if (d.score < vehHighThresh) continue;
        const ov = iou(pbox, d.box);
        const dcx = 0.5 * (d.box[0] + d.box[2]);
        const dcy = 0.5 * (d.box[1] + d.box[3]);
        const dist = Math.hypot(dcx - pcx, dcy - pcy) / diag;
        const embSim = dotVec(vt.emb, d.emb);
        if (ov >= 0.12 || (dist <= 0.75 && embSim >= 0.45)) {
          const score = 2.6 * ov + 1.6 * Math.max(0, embSim) - dist;
          pairs1.push({ ti, di, score });
        }
      }
    }
    pairs1.sort((a, b) => b.score - a.score);
    for (const { ti, di } of pairs1) {
      if (matchedTrack.has(ti) || matchedDet.has(di)) continue;
      matchedTrack.add(ti);
      matchedDet.add(di);
      applyVehicleMatch(vtracks[ti], dets[di]);
    }

    // ByteTrack Pass 2: associate low-confidence detections (0.20 <= score < 0.36) to established vtracks (hits >= 2, IoU >= 0.30).
    const pairs2 = [];
    for (let ti = 0; ti < vtracks.length; ti++) {
      if (matchedTrack.has(ti) || vtracks[ti].hits < 2) continue;
      const pbox = predBoxes[ti];
      for (let di = 0; di < dets.length; di++) {
        if (matchedDet.has(di) || dets[di].score >= vehHighThresh) continue;
        const ov = iou(pbox, dets[di].box);
        if (ov >= 0.30) pairs2.push({ ti, di, score: ov });
      }
    }
    pairs2.sort((a, b) => b.score - a.score);
    for (const { ti, di } of pairs2) {
      if (matchedTrack.has(ti) || matchedDet.has(di)) continue;
      matchedTrack.add(ti);
      matchedDet.add(di);
      applyVehicleMatch(vtracks[ti], dets[di]);
    }

    for (let ti = 0; ti < vtracks.length; ti++) {
      if (matchedTrack.has(ti)) continue;
      vtracks[ti].missed++;
    }
    const maxVehAge = Math.max(1.8, 3.5 * lastDetIntervalSec);
    vtracks = vtracks.filter(vt => vt.missed <= 6 && time - vt.last <= maxVehAge);

    // Birth new vtracks ONLY when score >= 0.42.
    for (let di = 0; di < dets.length; di++) {
      if (matchedDet.has(di)) continue;
      const d = dets[di];
      if (d.score < vehBirthThresh) continue;
      const cx = 0.5 * (d.box[0] + d.box[2]);
      const cy = 0.5 * (d.box[1] + d.box[3]);
      const w = Math.max(10, d.box[2] - d.box[0]);
      const h = Math.max(10, d.box[3] - d.box[1]);
      const vt = {
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
      };
      vtracks.push(vt);
      d.vtRef = vt;
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
    // Render zoom ROIs directly into native 768x448 detector resolution (eliminates 384x256 blur bottleneck).
    const zFloat = zoomFloatBuf;
    zFloat.fill(114 / 255);
    const zStride = LB_W * LB_H;
    const subRegions = [];
    const fillTile = (rx0, ry0, rx1, ry1, qx0, qy0, qx1, qy1) => {
      const rw = Math.max(16, rx1 - rx0);
      const rh = Math.max(12, ry1 - ry0);
      const sw = qx1 - qx0;
      const sh = qy1 - qy0;
      const sc = Math.min(sw / rw, sh / rh);
      const nw = Math.max(1, Math.round(rw * sc));
      const nh = Math.max(1, Math.round(rh * sc));
      const px = qx0 + Math.floor((sw - nw) / 2);
      const py = qy0 + Math.floor((sh - nh) / 2);
      if (sc > 1.3) {
        // Separable Catmull-Rom (a = -0.5) bicubic interpolation for crisp low-res Smart-Zoom magnification.
        const colIdx = new Int32Array(nw * 4);
        const colW = new Float32Array(nw * 4);
        for (let x = 0; x < nw; x++) {
          const sx = clamp(rx0 + (x + 0.5) / sc - 0.5, 0, wOrig - 1);
          const ix = Math.floor(sx);
          const tx = sx - ix;
          const tx2 = tx * tx, tx3 = tx2 * tx;
          const off = x * 4;
          colIdx[off] = clamp(ix - 1, 0, wOrig - 1) * 4;
          colIdx[off + 1] = clamp(ix, 0, wOrig - 1) * 4;
          colIdx[off + 2] = clamp(ix + 1, 0, wOrig - 1) * 4;
          colIdx[off + 3] = clamp(ix + 2, 0, wOrig - 1) * 4;
          colW[off] = -0.5 * tx + tx2 - 0.5 * tx3;
          colW[off + 1] = 1.0 - 2.5 * tx2 + 1.5 * tx3;
          colW[off + 2] = 0.5 * tx + 2.0 * tx2 - 1.5 * tx3;
          colW[off + 3] = -0.5 * tx2 + 0.5 * tx3;
        }
        const rowStride = wOrig * 4;
        const inv255 = 1 / 255;
        for (let y = 0; y < nh; y++) {
          const sy = clamp(ry0 + (y + 0.5) / sc - 0.5, 0, hOrig - 1);
          const iy = Math.floor(sy);
          const ty = sy - iy;
          const ty2 = ty * ty, ty3 = ty2 * ty;
          const wy0 = (-0.5 * ty + ty2 - 0.5 * ty3) * inv255;
          const wy1 = (1.0 - 2.5 * ty2 + 1.5 * ty3) * inv255;
          const wy2 = (0.5 * ty + 2.0 * ty2 - 1.5 * ty3) * inv255;
          const wy3 = (-0.5 * ty2 + 0.5 * ty3) * inv255;
          const r0 = clamp(iy - 1, 0, hOrig - 1) * rowStride;
          const r1 = clamp(iy, 0, hOrig - 1) * rowStride;
          const r2 = clamp(iy + 1, 0, hOrig - 1) * rowStride;
          const r3 = clamp(iy + 2, 0, hOrig - 1) * rowStride;
          const dstRow = (py + y) * LB_W + px;
          for (let x = 0; x < nw; x++) {
            const off = x * 4;
            const c0 = colIdx[off], c1 = colIdx[off + 1], c2 = colIdx[off + 2], c3 = colIdx[off + 3];
            const wx0 = colW[off], wx1 = colW[off + 1], wx2 = colW[off + 2], wx3 = colW[off + 3];
            const rVal =
              wy0 * (frame[r0 + c0] * wx0 + frame[r0 + c1] * wx1 + frame[r0 + c2] * wx2 + frame[r0 + c3] * wx3) +
              wy1 * (frame[r1 + c0] * wx0 + frame[r1 + c1] * wx1 + frame[r1 + c2] * wx2 + frame[r1 + c3] * wx3) +
              wy2 * (frame[r2 + c0] * wx0 + frame[r2 + c1] * wx1 + frame[r2 + c2] * wx2 + frame[r2 + c3] * wx3) +
              wy3 * (frame[r3 + c0] * wx0 + frame[r3 + c1] * wx1 + frame[r3 + c2] * wx2 + frame[r3 + c3] * wx3);
            const gVal =
              wy0 * (frame[r0 + c0 + 1] * wx0 + frame[r0 + c1 + 1] * wx1 + frame[r0 + c2 + 1] * wx2 + frame[r0 + c3 + 1] * wx3) +
              wy1 * (frame[r1 + c0 + 1] * wx0 + frame[r1 + c1 + 1] * wx1 + frame[r1 + c2 + 1] * wx2 + frame[r1 + c3 + 1] * wx3) +
              wy2 * (frame[r2 + c0 + 1] * wx0 + frame[r2 + c1 + 1] * wx1 + frame[r2 + c2 + 1] * wx2 + frame[r2 + c3 + 1] * wx3) +
              wy3 * (frame[r3 + c0 + 1] * wx0 + frame[r3 + c1 + 1] * wx1 + frame[r3 + c2 + 1] * wx2 + frame[r3 + c3 + 1] * wx3);
            const bVal =
              wy0 * (frame[r0 + c0 + 2] * wx0 + frame[r0 + c1 + 2] * wx1 + frame[r0 + c2 + 2] * wx2 + frame[r0 + c3 + 2] * wx3) +
              wy1 * (frame[r1 + c0 + 2] * wx0 + frame[r1 + c1 + 2] * wx1 + frame[r1 + c2 + 2] * wx2 + frame[r1 + c3 + 2] * wx3) +
              wy2 * (frame[r2 + c0 + 2] * wx0 + frame[r2 + c1 + 2] * wx1 + frame[r2 + c2 + 2] * wx2 + frame[r2 + c3 + 2] * wx3) +
              wy3 * (frame[r3 + c0 + 2] * wx0 + frame[r3 + c1 + 2] * wx1 + frame[r3 + c2 + 2] * wx2 + frame[r3 + c3 + 2] * wx3);
            const dst = dstRow + x;
            zFloat[dst] = clamp(rVal, 0, 1);
            zFloat[dst + zStride] = clamp(gVal, 0, 1);
            zFloat[dst + 2 * zStride] = clamp(bVal, 0, 1);
          }
        }
      } else {
        for (let y = 0; y < nh; y++) {
          const sy = clamp(ry0 + (y + 0.5) / sc - 0.5, 0, hOrig - 1);
          const y0 = sy | 0, y1 = y0 + 1 < hOrig ? y0 + 1 : y0, fy = sy - y0;
          const dstRow = (py + y) * LB_W + px;
          for (let x = 0; x < nw; x++) {
            const sx = clamp(rx0 + (x + 0.5) / sc - 0.5, 0, wOrig - 1);
            const x0 = sx | 0, x1 = x0 + 1 < wOrig ? x0 + 1 : x0, fx = sx - x0;
            const wa = ((1 - fx) * (1 - fy)) / 255, wb2 = (fx * (1 - fy)) / 255;
            const wd = ((1 - fx) * fy) / 255, we = (fx * fy) / 255;
            const a = (y0 * wOrig + x0) * 4, b = (y0 * wOrig + x1) * 4;
            const d = (y1 * wOrig + x0) * 4, e = (y1 * wOrig + x1) * 4;
            const dst = dstRow + x;
            zFloat[dst] = frame[a] * wa + frame[b] * wb2 + frame[d] * wd + frame[e] * we;
            zFloat[dst + zStride] = frame[a + 1] * wa + frame[b + 1] * wb2 + frame[d + 1] * wd + frame[e + 1] * we;
            zFloat[dst + 2 * zStride] = frame[a + 2] * wa + frame[b + 2] * wb2 + frame[d + 2] * wd + frame[e + 2] * we;
          }
        }
      }
      subRegions.push({ qx0, qy0, qx1, qy1, sc, px, py, rx0, ry0, rx1, ry1 });
    };

    if (rois.length >= 2) {
      const qw = Math.floor(LB_W / 2) - 2;
      const qh = Math.floor(LB_H / 2) - 2;
      const slots = [
        [0, 0, qw, qh],
        [qw + 4, 0, LB_W, qh],
        [0, qh + 4, qw, LB_H],
        [qw + 4, qh + 4, LB_W, LB_H],
      ];
      const count = Math.min(4, rois.length);
      for (let i = 0; i < count; i++) {
        const [rx0, ry0, rx1, ry1] = rois[i];
        const [qx0, qy0, qx1, qy1] = slots[i];
        fillTile(rx0, ry0, rx1, ry1, qx0, qy0, qx1, qy1);
      }
    } else {
      const [rx0, ry0, rx1, ry1] = rois[0];
      fillTile(rx0, ry0, rx1, ry1, 0, 0, LB_W, LB_H);
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
    const zoomThresh = 0.20;
    const zoomLogitMin = Math.log(zoomThresh / (1 - zoomThresh));
    const found = [];
    for (let i = 0; i < total; i++) {
      const pL = cls[i * 3], vL = cls[i * 3 + 1], sL = cls[i * 3 + 2];
      if (pL < zoomLogitMin || pL <= vL || pL <= sL) continue;
      const tcx = 0.5 * (boxes[i * 4] + boxes[i * 4 + 2]);
      const tcy = 0.5 * (boxes[i * 4 + 1] + boxes[i * 4 + 3]);
      for (const reg of subRegions) {
        if (tcx < reg.qx0 || tcx > reg.qx1 || tcy < reg.qy0 || tcy > reg.qy1) continue;
        const invZ = 1 / Math.max(1e-6, reg.sc);
        const box = [
          clamp((boxes[i * 4] - reg.px) * invZ + reg.rx0, 0, wOrig),
          clamp((boxes[i * 4 + 1] - reg.py) * invZ + reg.ry0, 0, hOrig),
          clamp((boxes[i * 4 + 2] - reg.px) * invZ + reg.rx0, 0, wOrig),
          clamp((boxes[i * 4 + 3] - reg.py) * invZ + reg.ry0, 0, hOrig),
        ];
        const bw = box[2] - box[0], bh = box[3] - box[1];
        if (bw < 4.5 || bh < 2.4 || bw / Math.max(1e-3, bh) < 0.45 || bw / Math.max(1e-3, bh) > 5.0) continue;
        if (curStrongBoxes.some(sb => iou(sb, box) >= 0.35)) continue;
        const corners = cornersData
          ? Array.from({ length: 4 }, (_, k) => [
              clamp((cornersData[i * 8 + k * 2] - reg.px) * invZ + reg.rx0, 0, wOrig),
              clamp((cornersData[i * 8 + k * 2 + 1] - reg.py) * invZ + reg.ry0, 0, hOrig),
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
      const rawCov = out.corner_cov?.data;
      const cornerCov = [];
      const rw2 = fw[2] * fw[2], rwh = fw[2] * fw[3], rh2 = fw[3] * fw[3];
      for (let k = 0; k < 4; k++) {
        if (rawCov) {
          const co = i * 16 + k * 4;
          const c00 = rawCov[co] * rw2;
          const c01 = 0.5 * (rawCov[co + 1] + rawCov[co + 2]) * rwh;
          const c11 = rawCov[co + 3] * rh2;
          if (Number.isFinite(c00) && Number.isFinite(c01) && Number.isFinite(c11) && c00 >= 0 && c11 >= 0) {
            cornerCov.push([c00 + 0.0625, c01, c11 + 0.0625]);
            continue;
          }
        }
        const fb = Math.max(0.25, (0.022 * fw[2]) ** 2);
        cornerCov.push([fb, 0, fb]);
      }
      const needsRecrop =
        (!items[i].track || items[i].track.missed > 0) &&
        valid && support >= 0.58 && nc <= 1.85 &&
        (minU < 0.015 || maxU > 0.985 || minV < 0.015 || maxV > 0.985) &&
        corners.every(([x, y]) => x >= 2 && x <= wOrig - 2 && y >= 2 && y <= hOrig - 2);
      const reidEmb = out.reid_emb ? normalizeVec(out.reid_emb.data.subarray(i * 48, (i + 1) * 48)) : null;
      results.push({ valid, llr, support, nc, visibility, corners, cornerCov, needsRecrop, reidEmb });
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
          if (discardShortUnreadable && t.frames <= 4 && (t.frames <= 2 || sigmoid(t.odds) < 0.78)) {
            t.state = 'DISCARDED';
          } else {
            const sealed = await sealTrack(t);
            if (!t.suppressedUnreadable) preSealedOnCut.push(sealed);
          }
        }
        tracks = [];
      }
      vtracks = [];
      h3 = zero([1, 36, 56, 96]);
      h4 = zero([1, 36, 28, 48]);
    }
    const singleThread = runtime.env.wasm.numThreads <= 1;
    const isLowRes = wOrig * hOrig <= 235000;
    const activeConfirmed = !scout.sceneCut && (active.some(t => t.frames >= 2) || finalizedActive.length > 0);
    const missingConfirmed = !scout.sceneCut && active.some(t => t.frames >= 2 && t.missed > 0);
    const effHeartbeat = isLowRes
      ? (!activeConfirmed || missingConfirmed ? 2 : (singleThread ? maxHeartbeat : 4))
      : (active.length >= 4 ? maxHeartbeat : baseHeartbeat);
    const framesSinceFull = frameIndex - lastFull;

    const full =
      frameIndex === 0 ||
      scout.sceneCut ||
      framesSinceFull >= effHeartbeat ||
      (!activeConfirmed && (isLowRes ? framesSinceFull >= 1 : framesSinceFull >= 2)) ||
      (scout.busyTriggered && framesSinceFull >= (isLowRes ? 2 : 3)) ||
      (missingConfirmed && framesSinceFull >= (isLowRes ? 1 : 2));

    const proposals = [];
    let lbInfo = null;

    if (full) {
      if (Number.isFinite(lastFullTime) && time > lastFullTime) {
        lastDetIntervalSec = clamp(time - lastFullTime, 0.03, 1.5);
      }
      lastFullTime = time;
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

      // 1. Decode D-FINE vehicle queries down to vehLowThresh = 0.20 for ByteTrack two-pass association.
      const rawVehCandidates = [];
      const frameArea = wOrig * hOrig;
      for (let q = 0; q < numQ; q++) {
        const idx = aPyr + q;
        const vLogit = cls[idx * 3 + 1];
        const sLogit = cls[idx * 3 + 2];
        if (vLogit < vehLowLogitMin || vLogit <= sLogit) continue;
        const box = nativeBox(output.pred_boxes.data, idx * 4);
        const bw = box[2] - box[0], bh = box[3] - box[1];
        if (bw < 14 || bh < 14) continue;
        const aspect = bw / bh;
        if (aspect < 0.36 || aspect > 3.5) continue;
        const score = sigmoid(vLogit);
        if (bw * bh > 0.78 * frameArea && score < 0.55) continue;
        if (bw * bh > 0.45 * frameArea && score < 0.48) continue;
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
          vtRef: null,
        });
      }
      const nmsVehs = nms(rawVehCandidates, 16, 0.52);
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
      // Step 2: Drop interior window/cab/door sub-boxes inside a larger vehicle.
      const keptVehs = nonWrapperVehs.filter((a, i) => {
        const areaA = Math.max(1, (a.box[2] - a.box[0]) * (a.box[3] - a.box[1]));
        for (let j = 0; j < nonWrapperVehs.length; j++) {
          if (i === j) continue;
          const b = nonWrapperVehs[j];
          if (b.score < vehHighThresh) continue;
          const ix = Math.max(0, Math.min(a.box[2], b.box[2]) - Math.max(a.box[0], b.box[0]));
          const iy = Math.max(0, Math.min(a.box[3], b.box[3]) - Math.max(a.box[1], b.box[1]));
          const inter = ix * iy;
          if (inter <= 0) continue;
          const areaB = Math.max(1, (b.box[2] - b.box[0]) * (b.box[3] - b.box[1]));
          if (
            areaB >= 1.50 * areaA &&
            inter / areaA >= 0.80 &&
            (a.box[3] < b.box[1] + 0.82 * (b.box[3] - b.box[1]) || b.score >= 0.88 * a.score)
          ) {
            return false;
          }
          if (b.score >= a.score && b.score >= 0.72 && inter / areaA >= 0.72 && areaA <= 0.55 * areaB) return false;
        }
        return true;
      });
      updateVehicleTracks(keptVehs, time);
      const cutoutVehs = keptVehs.filter(v => v.vtRef !== null);
      if (cutoutVehs.length) {
        const cutout = await sessions.vehicle_cutout_nano.run({
          p2_feat: output.p2_feat,
          aux_edge_3ch: output.aux_edge_3ch,
          veh_rois: float(cutoutVehs.flatMap(v => [0, ...v.lbBox]), [cutoutVehs.length, 5]),
          query_condition: float(
            cutoutVehs.flatMap(v => Array.from(output.query_condition.data.subarray(v.q * 64, (v.q + 1) * 64))),
            [cutoutVehs.length, 64]
          ),
        });
        cutoutVehs.forEach((v, i) => {
          const m = cutout.cutout_logits.data.slice(i * 4096, (i + 1) * 4096);
          v.mask = m;
          if (v.vtRef) v.vtRef.mask = m;
        });
      }

      // 2. Decode Pass-1 pyramid/query plate candidates.
      const pass1Plates = decodePlatesFromOutput(output, nativeBox, nativeCorners, 0, plateThresh);
      proposals.push(...pass1Plates);

      // 3. Pass-2 Smart-Zoom (all resolutions, Catmull-Rom bicubic) for unplated vehicles & weak plate peaks.
      const pass1CoveredBoxes = pass1Plates.filter(p => p.score >= 0.26).map(p => p.box);
      const alreadyLockedVehIds = new Set(
        tracks
          .filter(
            t =>
              (t.state === 'COLLECTING' &&
                t.frames >= 3 &&
                t.missed === 0 &&
                mean(t.memory.mem_P.data) >= 0.16 &&
                sigmoid(t.odds) >= 0.72) ||
              (t.state === 'FINALIZED' && t.result?.status === 'ACCEPTED')
          )
          .map(t => t.vehicleId)
          .filter(v => v !== null && v !== undefined)
      );
      const zoomableVehs = vtracks.filter(vt => {
        if (alreadyLockedVehIds.has(vt.id)) return false;
        if (vt.hits < 2 && vt.score < 0.62 && pass1Plates.length > 0) return false;
        if ((vt.zoomSkip ?? 0) > 0) {
          vt.zoomSkip--;
          return false;
        }
        const vb = predictedVehicleBox(vt, time);
        const vw = vb[2] - vb[0], vh = vb[3] - vb[1];
        const coveredByPass1 = pass1CoveredBoxes.some(pb => {
          const pcx = 0.5 * (pb[0] + pb[2]), pcy = 0.5 * (pb[1] + pb[3]);
          return pcx >= vb[0] - 0.06 * vw && pcx <= vb[2] + 0.06 * vw && pcy >= vb[1] - 0.02 * vh && pcy <= vb[3] + 0.16 * vh;
        });
        return !coveredByPass1;
      });
      if (zoomableVehs.length > 0) zoomCooldown = 0;
      else if (zoomCooldown > 0) zoomCooldown--;

      if (zoomCooldown === 0 && !(singleThread && scale >= 0.92 && !isLowRes && pass1Plates.length > 0)) {
        const pass1StrongBoxes = pass1Plates.filter(p => p.score >= 0.50).map(p => p.box);
        const zoomRois = [];
        const zoomedVts = [];
        const enqBox = (rx0, ry0, rx1, ry1, vtOwner = null) => {
          if (zoomRois.length >= 4) return;
          const x0 = Math.max(0, Math.floor(rx0)), y0 = Math.max(0, Math.floor(ry0));
          const x1 = Math.min(wOrig, Math.ceil(rx1)), y1 = Math.min(hOrig, Math.ceil(ry1));
          if (x1 - x0 < 20 || y1 - y0 < 14) return;
          const win = [x0, y0, x1, y1];
          if (zoomRois.some(prev => iou(prev, win) >= 0.6)) return;
          zoomRois.push(win);
          if (vtOwner) zoomedVts.push(vtOwner);
        };
        for (const vt of zoomableVehs) {
          if (zoomRois.length >= 4) break;
          const vb = predictedVehicleBox(vt, time);
          const vw = vb[2] - vb[0], vh = vb[3] - vb[1];
          if (vt.mountRel && vt.mountProb >= 0.32 && (vw * scale < 380 || scale < 0.88 || isLowRes)) {
            const mx0 = vb[0] + vt.mountRel[0] * vw, my0 = vb[1] + vt.mountRel[1] * vh;
            const mx1 = vb[0] + vt.mountRel[2] * vw, my1 = vb[1] + vt.mountRel[3] * vh;
            const mcx = 0.5 * (mx0 + mx1), mcy = 0.5 * (my0 + my1);
            const zw = Math.max(0.58 * vw, 3.2 * (mx1 - mx0), 72);
            const zh = Math.max(0.48 * vh, 3.6 * (my1 - my0), 48);
            enqBox(
              Math.max(vb[0] - 0.08 * vw, mcx - 0.5 * zw),
              Math.max(vb[1] - 0.02 * vh, mcy - 0.5 * zh),
              Math.min(vb[2] + 0.08 * vw, mcx + 0.5 * zw),
              Math.min(vb[3] + 0.12 * vh, mcy + 0.5 * zh),
              vt
            );
          } else if (vw * scale < 280 || (scale < 0.88 && vw * scale <= 540)) {
            enqBox(vb[0] - 0.08 * vw, vb[1] - 0.02 * vh, vb[2] + 0.08 * vw, vb[3] + 0.10 * vh, vt);
          }
        }
        if (
          zoomRois.length < 4 &&
          (zoomRois.length > 0 || scale < 0.88 || isLowRes || (vtracks.length === 0 && pass1StrongBoxes.length === 0))
        ) {
          const minWeak = zoomableVehs.length > 0 ? weakPlateThresh : (isLowRes || scale < 0.88 ? 0.16 : 0.28);
          const weakPeaks = nms(
            decodePlatesFromOutput(output, nativeBox, nativeCorners, 1, minWeak, plateThresh),
            4,
            0.35
          );
          for (const wp of weakPeaks) {
            const wbw = wp.box[2] - wp.box[0], wbh = wp.box[3] - wp.box[1];
            if (wbw * scale > 40 && zoomableVehs.length === 0) continue;
            const wcx = 0.5 * (wp.box[0] + wp.box[2]), wcy = 0.5 * (wp.box[1] + wp.box[3]);
            const covered = pass1CoveredBoxes.some(sb => wcx >= sb[0] - 8 && wcx <= sb[2] + 8 && wcy >= sb[1] - 8 && wcy <= sb[3] + 8);
            if (covered) continue;
            const lockedVehHit = vtracks.some(vt => {
              if (!alreadyLockedVehIds.has(vt.id)) return false;
              const vb = predictedVehicleBox(vt, time);
              return wcx >= vb[0] && wcx <= vb[2] && wcy >= vb[1] && wcy <= vb[3];
            });
            if (lockedVehHit) continue;
            const ww = Math.max(72, 4.5 * wbw);
            const wh = Math.max(48, 5.0 * wbh);
            enqBox(wcx - 0.5 * ww, wcy - 0.5 * wh, wcx + 0.5 * ww, wcy + 0.5 * wh);
          }
        }
        if (zoomRois.length) {
          const zoomPlates = await runSmartZoomMosaic(zoomRois, pass1StrongBoxes);
          for (const vt of zoomedVts) {
            const vb = predictedVehicleBox(vt, time);
            const hit = zoomPlates.some(zp => {
              const zcx = 0.5 * (zp.box[0] + zp.box[2]), zcy = 0.5 * (zp.box[1] + zp.box[3]);
              return zcx >= vb[0] && zcx <= vb[2] && zcy >= vb[1] && zcy <= vb[3];
            });
            if (!hit) {
              vt.zoomMisses = (vt.zoomMisses ?? 0) + 1;
              vt.zoomSkip = Math.min(5, vt.zoomMisses * 2);
            } else {
              vt.zoomMisses = 0;
              vt.zoomSkip = 0;
            }
          }
          if (!zoomPlates.length && !zoomableVehs.length) zoomCooldown = 2;
          proposals.push(...zoomPlates);
        }
      }
    } else {
      fastFrames++;
    }

    const platedVehicleIds = new Set(
      tracks
        .filter(t =>
          (t.state === 'COLLECTING' && t.frames >= 2 && mean(t.memory.mem_P.data) >= 0.12 && sigmoid(t.odds) >= 0.60) ||
          (t.state === 'FINALIZED' && t.result?.status === 'ACCEPTED')
        )
        .map(t => t.vehicleId)
        .filter(v => v !== null && v !== undefined)
    );
    const allVtracks = vtracks.map(vt => {
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
        vt,
        box,
        velocity: [vt.vx, vt.vy, vt.vw, vt.vh],
        mask: vt.mask,
        score: vt.score,
        mountBox,
        mountProb: vt.mountProb,
      };
    });

    // Use allVtracks for ownerOf containment so stationary/red-light vehicles NEVER block their own plates!
    const ownerOf = box => {
      const cx = 0.5 * (box[0] + box[2]), cy = 0.5 * (box[1] + box[3]);
      let best = null, bestArea = Infinity;
      for (const v of allVtracks) {
        if (
          v.vt.hits >= 2 &&
          v.vt.maxDisp < Math.max(8, 0.015 * wOrig) &&
          v.vt.maxScore < 0.56 &&
          !platedVehicleIds.has(v.id)
        ) {
          continue;
        }
        const vw = v.box[2] - v.box[0], vh = v.box[3] - v.box[1];
        const minY = v.box[1] - 0.02 * vh;
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

    const predictedVehicles = allVtracks
      .filter(({ vt }) => {
        if (platedVehicleIds.has(vt.id)) return true;
        if (vt.hits < 3 && vt.maxScore < 0.78) return false;
        if (
          vt.hits >= 3 &&
          vt.maxDisp < Math.max(8, 0.015 * wOrig) &&
          vt.maxScore < 0.78
        ) {
          return false;
        }
        return true;
      })
      .map(({ vt, ...pub }) => pub);

    // Sub-pixel optical pre-alignment on current frame before Stage-2 search window generation.
    for (const track of [...active, ...finalizedActive, ...dormant]) {
      track.curOpt = null;
      if (!scout.sceneCut && track.optRef && time - track.last <= 1.20) {
        const dtOpt = clamp(time - track.last, 0.01, 0.5);
        const kinCorners = predictedCorners(track, time);
        const minZ = track.optRef.basis.width < 36 ? 0.50 : 0.58;
        const opt = trackPlateOnRgba(track.optRef, frame, wOrig, hOrig, kinCorners, {
          searchRadiusPx: clamp(22 + 240 * dtOpt, 18, 60),
          allowScaleSearch: dtOpt >= 0.04,
          minZncc: minZ,
        });
        if (opt.ok && validCorners(opt.corners)) track.curOpt = opt;
      }
    }

    // Smoothly predict active & dormant tracks and generate Stage-2 tracking search proposals.
    const proposalBoxFor = track => {
      const kin = predictedBox(track, time);
      const opt = track.curOpt;
      if (!opt) return kin;
      if (opt.zncc >= 0.70) return opt.box;
      return [
        0.5 * (kin[0] + opt.box[0]),
        0.5 * (kin[1] + opt.box[1]),
        0.5 * (kin[2] + opt.box[2]),
        0.5 * (kin[3] + opt.box[3]),
      ];
    };
    for (const track of active) {
      proposals.push({
        box: proposalBoxFor(track),
        score: full ? 0.45 : 1.0,
        track,
        provenance: 2,
      });
    }
    for (const track of finalizedActive) {
      const box = proposalBoxFor(track);
      if (!full || ownerOf(box) !== null || Boolean(track.curOpt && track.curOpt.zncc >= 0.70)) {
        proposals.push({
          box,
          score: full ? 0.01 : 0.9,
          track,
          provenance: 2,
        });
      }
    }
    for (const track of dormant) {
      const maxDormantSec = track.frames >= 2 ? 1.20 : 0.22;
      const allowDormantProbe = track.missed <= 2 || full || Boolean(track.curOpt && track.curOpt.zncc >= 0.56);
      if (time - track.dormantSince <= maxDormantSec && allowDormantProbe) {
        proposals.push({
          box: proposalBoxFor(track),
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
        if (addedMounts >= 2) break;
        if (vt.hits < 2 || vt.mountProb < 0.38 || !vt.mountRel) continue;
        if (vt.maxDisp < Math.max(8, 0.015 * wOrig) && vt.maxScore < 0.58) continue;
        const vb = predictedVehicleBox(vt, time);
        const vw = vb[2] - vb[0], vh = vb[3] - vb[1];
        if (vw < 28 || vh < 20) continue;
        const covered = proposals.some(p => {
          const pcx = 0.5 * (p.box[0] + p.box[2]), pcy = 0.5 * (p.box[1] + p.box[3]);
          return pcx >= vb[0] - 0.06 * vw && pcx <= vb[2] + 0.06 * vw && pcy >= vb[1] - 0.02 * vh && pcy <= vb[3] + 0.18 * vh;
        });
        if (covered) continue;
        const roi = [
          clamp(vb[0] + vt.mountRel[0] * vw, 0, wOrig),
          clamp(vb[1] + vt.mountRel[1] * vh, 0, hOrig),
          clamp(vb[0] + vt.mountRel[2] * vw, 0, wOrig),
          clamp(vb[1] + vt.mountRel[3] * vh, 0, hOrig),
        ];
        if (roi[2] - roi[0] >= 6 && roi[3] - roi[1] >= 3) {
          proposals.push({ box: roi, score: 0.15, provenance: 2, vehicleId: vt.id, speculative: true });
          vt.mountCooldown = 3;
          addedMounts++;
        }
      }
    }

    // Keep track-backed proposals, vehicle-contained proposals, or standalone plate candidates (score >= 0.36).
    const contained = proposals.filter(({ box, score, track, provenance }) =>
      track || (
        (box[2] - box[0] >= 5.5 && box[3] - box[1] >= 2.8) && (
          provenance === 2 ||
          ownerOf(box) !== null ||
          score >= (provenance === 1 ? 0.46 : 0.20)
        )
      )
    );

    // Link unbacked detector proposals to existing collectors before NMS so we never spawn duplicate tracks.
    for (const c of contained) {
      if (c.track) continue;
      let bestTrack = null, bestIou = 0.25;
      for (const t of tracks) {
        if (t.state !== 'COLLECTING') continue;
        const ov = Math.max(iou(predictedBox(t, time), c.box), t.curOpt ? iou(t.curOpt.box, c.box) : 0);
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
    for (const ft of finalizedActive) {
      const ow = ownerOf(ft.curOpt ? ft.curOpt.box : predictedBox(ft, time));
      if (ow) acceptedOwners.add(ow.id);
    }
    const remaining = [];
    for (const candidate of nms(contained, 20, 0.45)) {
      const cq = candidate.corners;
      const cBox = validCorners(cq) ? cornersToBox(cq) : candidate.box;
      const acceptedCover = tracks.find(t => {
        if (t.state !== 'FINALIZED' || t.result.status !== 'ACCEPTED') return false;
        const kinBox = predictedBox(t, time);
        const pBox = t.curOpt && iou(t.curOpt.box, cBox) > iou(kinBox, cBox) ? t.curOpt.box : kinBox;
        if (iou(pBox, cBox) < 0.20) return false;
        const ratio = Math.max(1, cBox[2] - cBox[0]) / Math.max(1, pBox[2] - pBox[0]);
        return ratio >= 0.45 && ratio <= 2.2;
      });
      if (acceptedCover) {
        if (!used.has(acceptedCover.id)) {
          const dt = Math.max(0.03, time - acceptedCover.last);
          const opt = acceptedCover.curOpt;
          const prevBaseBox = acceptedCover.measBox ?? acceptedCover.box;
          const prevBaseCorners = acceptedCover.measCorners ?? acceptedCover.corners;
          let nextCorners, nextBox;
          if (opt && opt.zncc >= 0.62) {
            nextCorners = validCorners(cq)
              ? opt.corners.map((oc, k) => [0.78 * oc[0] + 0.22 * cq[k][0], 0.78 * oc[1] + 0.22 * cq[k][1]])
              : opt.corners;
            nextBox = cornersToBox(nextCorners);
          } else if (validCorners(cq)) {
            nextCorners = cq;
            nextBox = cBox;
          } else {
            const dx0 = 0.5 * (cBox[0] + cBox[2] - prevBaseBox[0] - prevBaseBox[2]);
            const dy0 = 0.5 * (cBox[1] + cBox[3] - prevBaseBox[1] - prevBaseBox[3]);
            nextCorners = prevBaseCorners.map(([x, y]) => [x + dx0, y + dy0]);
            nextBox = cBox;
          }
          const dx = 0.5 * (nextBox[0] + nextBox[2] - prevBaseBox[0] - prevBaseBox[2]);
          const dy = 0.5 * (nextBox[1] + nextBox[3] - prevBaseBox[1] - prevBaseBox[3]);
          acceptedCover.velocity = [
            clamp(0.55 * acceptedCover.velocity[0] + 0.45 * (dx / dt), -480, 480),
            clamp(0.55 * acceptedCover.velocity[1] + 0.45 * (dy / dt), -480, 480),
          ];
          acceptedCover.measCorners = nextCorners;
          acceptedCover.measBox = nextBox;
          acceptedCover.corners = nextCorners;
          acceptedCover.box = nextBox;
          if (validCorners(cq) || (opt && opt.zncc >= 0.78)) {
            const patchU8 = extractPatchFromRgba(frame, wOrig, hOrig, nextCorners);
            const nextRef = createOpticalRef(patchU8, nextCorners);
            if (nextRef) {
              acceptedCover.optRef = nextRef;
              acceptedCover.optPatch = patchU8;
            }
          }
          acceptedCover.last = time;
          acceptedCover.lastFrame = frameIndex;
          acceptedCover.missed = 0;
          used.add(acceptedCover.id);
          const owner = ownerOf(acceptedCover.measBox);
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
      tracks
        .filter(
          t =>
            t.state === 'COLLECTING' &&
            t.frames >= 3 &&
            t.vehicleId !== null &&
            mean(t.memory.mem_P.data) >= 0.16 &&
            sigmoid(t.odds) >= 0.72
        )
        .map(t => t.vehicleId)
    );
    const vehCandCounts = new Map();
    const claimedTrackIds = new Set();
    const selected = [];
    for (const candidate of remaining) {
      if (candidate.track) {
        if (used.has(candidate.track.id) || claimedTrackIds.has(candidate.track.id)) continue;
      }
      if (
        tracks.some(t => {
          if (!used.has(t.id)) return false;
          const tb = t.measBox ?? t.box;
          return iou(tb, candidate.box) >= 0.18 || iomin(tb, candidate.box) >= 0.65;
        })
      ) {
        continue;
      }
      if (selected.some(s => iou(s.box, candidate.box) >= 0.18 || iomin(s.box, candidate.box) >= 0.72)) continue;
      const owner = ownerOf(candidate.box);
      const vid = candidate.track?.vehicleId ?? owner?.id;
      if (vid !== undefined && vid !== null) {
        if (acceptedOwners.has(vid)) continue;
        const maxPerVeh = !confirmedPlatedVehicles.has(vid) ? 3 : 2;
        const cnt = vehCandCounts.get(vid) ?? 0;
        if (cnt >= maxPerVeh) continue;
        vehCandCounts.set(vid, cnt + 1);
      }
      if (candidate.track) claimedTrackIds.add(candidate.track.id);
      selected.push(candidate);
      if (selected.length >= (singleThread ? (activeConfirmed ? 2 : 4) : 10)) break;
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
      const reqSupport = candidate.speculative ? 0.68 : minIsPlateProb;
      if (!loc.valid || loc.support < reqSupport) continue;
      const rawCorners = loc.corners;
      if (!validCorners(rawCorners)) continue;
      const rawBox = cornersToBox(rawCorners);
      const boxW = rawBox[2] - rawBox[0], boxH = rawBox[3] - rawBox[1];
      const aspect = boxW / Math.max(1e-3, boxH);
      if (boxW < 5.5 || boxH < 2.8 || aspect < 0.48 || aspect > 5.0) continue;
      if (
        tracks.some(t => {
          if (!used.has(t.id)) return false;
          const tb = t.measBox ?? t.box;
          return iou(tb, rawBox) >= 0.15 || iomin(tb, rawBox) >= 0.62;
        })
      ) {
        continue;
      }
      const postOwner = ownerOf(rawBox);
      if (postOwner && postStage2Owners.has(postOwner.id)) continue;

      // Associate with tracked collectors (active OR dormant) using IoU + cosine ReID + distance gate.
      let track = prior;
      if (!track || used.has(track.id) || track.state !== 'COLLECTING') {
        track = null;
        let bestAssocScore = -Infinity;
        const pcx = 0.5 * (rawBox[0] + rawBox[2]), pcy = 0.5 * (rawBox[1] + rawBox[3]);
        for (const t of tracks) {
          if (t.state !== 'COLLECTING' || used.has(t.id)) continue;
          const kinBox = predictedBox(t, time);
          const pBox = t.curOpt && iou(t.curOpt.box, rawBox) > iou(kinBox, rawBox) ? t.curOpt.box : kinBox;
          const ov = iou(pBox, rawBox);
          const tcx = 0.5 * (pBox[0] + pBox[2]), tcy = 0.5 * (pBox[1] + pBox[3]);
          const tBase = t.measBox ?? t.box;
          const lastCx = 0.5 * (tBase[0] + tBase[2]), lastCy = 0.5 * (tBase[1] + tBase[3]);
          const dtGap = Math.max(0.03, time - t.last);
          const maxJump = Math.max(2.4 * (tBase[2] - tBase[0]), 60) + 420 * dtGap;
          if (Math.hypot(pcx - lastCx, pcy - lastCy) > maxJump) continue;
          const dist = Math.hypot(pcx - tcx, pcy - tcy) / Math.max(12, pBox[2] - pBox[0]);
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

      const policyIdx = candidate.provenance ?? 0;
      const tauNc = getStage2TauNc(policyIdx);

      // CRITICAL ORDER-OF-OPERATIONS: stabilize instantaneous measCorners BEFORE projective() + Drizzle!
      let measCorners = rawCorners;
      let measBox = rawBox;
      let crop;
      let srActive = false;

      if (!track) {
        const hasVehOwner = ownerOf(rawBox) !== null;
        const minBirth = hasVehOwner ? 0.36 : (candidate.provenance === 1 ? 0.80 : 0.66);
        const maxBirthNc = hasVehOwner ? tauNc + 1.8 : tauNc + 0.55;
        if (loc.support < minBirth || loc.llr < (hasVehOwner ? -0.35 : 0.25) || loc.nc > maxBirthNc) continue;
        crop = projective(measCorners);
        if (isRedLampCrop(crop)) continue;
        if (!hasVehOwner) {
          const cd = crop.data;
          let lSum = 0, lSq = 0;
          for (let k = 0; k < 12800; k += 4) {
            const lv = 0.299 * cd[k] + 0.587 * cd[12800 + k] + 0.114 * cd[25600 + k];
            lSum += lv;
            lSq += lv * lv;
          }
          const lMean = lSum / 3200;
          if (Math.sqrt(Math.max(0, lSq / 3200 - lMean * lMean)) < 0.032) continue;
        }
        track = newTrack(measBox, measCorners, loc.cornerCov, time);
        const drz0 = stepTrackDrizzle(track, measCorners);
        measCorners = drz0.corners;
        measBox = cornersToBox(measCorners);
        crop = drz0.crop;
      } else {
        if (!postOwner && track.vehicleId === null && track.frames >= 3 && loc.llr < -0.35) continue;
        const dtMeas = Math.max(0.02, time - track.last);
        let qHad = stepHadamardFilter(track, rawCorners, loc.cornerCov, dtMeas);
        const plateW =
          0.5 *
          (Math.hypot(qHad[1][0] - qHad[0][0], qHad[1][1] - qHad[0][1]) +
            Math.hypot(qHad[2][0] - qHad[3][0], qHad[2][1] - qHad[3][1]));
        const opt = track.curOpt;
        if (opt && opt.zncc >= 0.64 && loc.llr < 2.2) {
          const maxDiff =
            Math.max(...qHad.map((c, k) => Math.hypot(c[0] - opt.corners[k][0], c[1] - opt.corners[k][1]))) /
            Math.max(10, plateW);
          if (maxDiff < 0.24) {
            const wOpt = clamp(0.45 * opt.zncc, 0.20, 0.36);
            const fused = qHad.map((c, k) => [
              wOpt * opt.corners[k][0] + (1 - wOpt) * c[0],
              wOpt * opt.corners[k][1] + (1 - wOpt) * c[1],
            ]);
            if (validCorners(fused)) {
              qHad = fused;
              track.hadModes = cornersToHadamard(fused);
            }
          }
        } else if (plateW < 36 && track.lowResRef) {
          const lr = trackLowResTranslation2x(track.lowResRef, qHad, 0.48);
          if (lr.ok) {
            const dxLock = 0.80 * lr.dx, dyLock = 0.80 * lr.dy;
            const shifted = qHad.map(([x, y]) => [x + dxLock, y + dyLock]);
            if (validCorners(shifted)) {
              qHad = shifted;
              track.hadModes[0] += dxLock;
              track.hadModes[1] += dyLock;
            }
          }
        }
        const drz = stepTrackDrizzle(track, qHad);
        measCorners = drz.corners;
        measBox = cornersToBox(measCorners);
        crop = drz.crop;
        srActive = drz.srActive;
      }

      used.add(track.id);
      track.dormant = false;
      track.dormantSince = Infinity;
      track.missed = 0;

      if (track.state !== 'COLLECTING' || track.readouts !== 0) {
        throw Error('Sealed collector cannot receive evidence');
      }

      const geom = geometry(measCorners, crop, track.frames ? track.measCorners : null);
      const encoded = await sessions.frame_encoder.run({
        crops: crop,
        plate_sides: integer(2),
        geom_meta: geom,
      });
      const ncExcess = Math.max(0, loc.nc - tauNc);
      const negLlr = Math.max(0, -loc.llr - 0.25);
      const damping = Math.exp(-Math.min(20, 0.55 * ncExcess + 0.75 * negLlr)) * (0.15 + 0.85 * loc.visibility);
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
      const wasConfirmed = track.maxLlr >= 1.1;
      track.maxLlr = Math.max(track.maxLlr, increment);
      if (!track.frames) {
        track.firstSupport = loc.support;
        const posBoost = increment > 1.5 ? 1 + 0.60 * Math.min(1.6, increment - 1.5) : 1;
        track.odds = clamp(0.85 * (0.25 + 0.75 * loc.visibility) * increment * posBoost, -30, 30);
        track.measCorners = measCorners;
        track.measBox = measBox;
        track.corners = measCorners;
        track.box = measBox;
      } else {
        const dt = Math.max(0.03, time - track.last);
        const wt = Math.min(0.85, Math.max(0.26, dt / 0.20)) * (0.20 + 0.80 * loc.visibility);
        const hasCharEvidence = mean(track.memory.mem_P.data) >= 0.18;
        const effInc = hasCharEvidence ? Math.max(-0.90, increment) : increment;
        track.odds = clamp(track.odds + wt * effInc, -30, 30);

        // Zero-lag instantaneous measurement state vs display-smoothed state.
        const oldMeasBox = track.measBox ?? track.box;
        const oldCx = 0.5 * (oldMeasBox[0] + oldMeasBox[2]);
        const oldCy = 0.5 * (oldMeasBox[1] + oldMeasBox[3]);
        const oldW = oldMeasBox[2] - oldMeasBox[0];
        const oldH = oldMeasBox[3] - oldMeasBox[1];
        const predCx = oldCx + track.velocity[0] * dt;
        const predCy = oldCy + track.velocity[1] * dt;
        const measCx = 0.5 * (measBox[0] + measBox[2]);
        const measCy = 0.5 * (measBox[1] + measBox[3]);
        const measW = measBox[2] - measBox[0];
        const measH = measBox[3] - measBox[1];
        const beta = track.frames < 2 ? 0.28 : 0.45;
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
        track.prevMeasCorners = track.measCorners;
        track.measCorners = measCorners;
        track.measBox = measBox;

        const prevDispCorners = predictedCorners(track, time, true);
        const smCorners = measCorners.map((c, k) => [
          0.22 * prevDispCorners[k][0] + 0.78 * c[0],
          0.22 * prevDispCorners[k][1] + 0.78 * c[1],
        ]);
        track.corners = validCorners(smCorners) ? smCorners : measCorners;
        track.box = cornersToBox(track.corners);
      }

      // Refresh optical & low-res 2x reference templates from zero-lag measCorners!
      if (!track.optRef || loc.support >= 0.56) {
        const patchU8 = extractPatchFromRgba(frame, wOrig, hOrig, track.measCorners);
        const nextRef = createOpticalRef(patchU8, track.measCorners);
        if (nextRef) {
          track.optRef = nextRef;
          track.optPatch = patchU8;
        }
      }
      const curMeasW = track.measBox[2] - track.measBox[0];
      if (curMeasW < 38 && (!track.lowResRef || loc.support >= 0.52)) {
        const nextLr = createLowResRef2x(track.measCorners);
        if (nextLr) track.lowResRef = nextLr;
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
        if (loc.llr >= 1.4 || mean(track.memory.mem_P.data) >= 0.12) {
          postStage2Owners.add(owner.id);
        }
      }
      if (geom.data[6] > track.sharpness || (srActive && track.srEffViews > (track.bestSrViews ?? 0) + 0.5)) {
        track.sharpness = Math.max(track.sharpness, geom.data[6]);
        if (srActive) track.bestSrViews = track.srEffViews;
        track.thumbnail = thumbnail(crop);
      }
      track.frames++;
      track.last = time;
      track.lastFrame = frameIndex;
      if (
        discardShortUnreadable &&
        track.frames >= 5 &&
        mean(track.memory.mem_P.data) < 0.045 &&
        (track.maxLlr < 2.2 || track.odds < -0.5)
      ) {
        track.state = 'DISCARDED';
      }
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
        if (
          exited &&
          discardShortUnreadable &&
          t.frames <= 4 &&
          (t.frames <= 2 ? t.maxLlr < 2.2 : (sigmoid(t.odds) < 0.78 && mean(t.memory.mem_P.data) < 0.40))
        ) {
          t.state = 'DISCARDED';
          continue;
        }
        let readyToSeal = false;
        if (requireFullFieldsForEarlySeal) {
          const pw = (t.measBox ?? t.box)[2] - (t.measBox ?? t.box)[0];
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
        if (exited || readyToSeal) {
          const sealed = await sealTrack(t);
          if (!t.suppressedUnreadable) newlyFinalized.push(sealed);
        }
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
        .filter(t => t.state === 'FINALIZED' && !t.suppressedUnreadable && used.has(t.id))
        .map(t => ({
          id: t.id,
          box: t.box,
          corners: t.corners,
          optCorners: t.measCorners ?? t.corners,
          velocity: t.velocity,
          sizeVelocity: t.sizeVelocity,
          optPatch: t.optPatch,
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
        .map(t => {
          const coastOpt = !used.has(t.id) && t.curOpt && t.curOpt.zncc >= 0.60 ? t.curOpt : null;
          return {
            id: t.id,
            box: used.has(t.id) ? t.box : coastOpt ? coastOpt.box : predictedBox(t, time, true),
            corners: used.has(t.id) ? t.corners : coastOpt ? coastOpt.corners : predictedCorners(t, time, true),
            optCorners: used.has(t.id) ? (t.measCorners ?? t.corners) : (coastOpt ? coastOpt.corners : predictedCorners(t, time)),
            velocity: t.velocity,
            sizeVelocity: t.sizeVelocity,
            optPatch: t.optPatch,
            vehicleId: t.vehicleId,
            coasted: !used.has(t.id),
            state: t.state,
            frames: t.frames,
            readouts: t.readouts,
          };
        }),
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
      const episodeSupport =
        track.frames === 1
          ? (track.maxLlr >= 2.2 && precision >= 0.45
              ? Math.max(track.firstSupport, sigmoid(1.35 * track.odds))
              : track.firstSupport)
          : sigmoid(track.odds);
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
        episodeSupport,
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
    if (
      discardShortUnreadable &&
      track.result.status !== 'ACCEPTED' &&
      (track.frames <= 5 || mean(track.memory.mem_P.data) < 0.10 || track.maxLlr < 1.6)
    ) {
      track.suppressedUnreadable = true;
    } else {
      emittedResults.push(track.result);
    }
    progress({ track_id: track.id, state: track.state, readouts: track.readouts });
    return track.result;
  }

  async function resegment(options = {}) {
    if (phase === 'IDLE' || phase === 'SEALED') throw Error('Collection is unavailable');
    const newlySealed = [];
    for (const t of tracks) {
      if (t.state === 'COLLECTING' && t.frames >= 2) {
        const sealed = await sealTrack(t);
        if (!t.suppressedUnreadable) newlySealed.push(sealed);
      }
    }
    tracks = [];
    vtracks = [];
    h3 = zero([1, 36, 56, 96]);
    h4 = zero([1, 36, 28, 48]);
    lastFull = -Infinity;
    lastFullTime = -Infinity;
    lastDetIntervalSec = 0.12;
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
      if (
        discardShortUnreadable &&
        t.frames <= 4 &&
        (t.frames <= 2 ? t.maxLlr < 2.2 : (sigmoid(t.odds) < 0.78 && mean(t.memory.mem_P.data) < 0.40))
      ) {
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
