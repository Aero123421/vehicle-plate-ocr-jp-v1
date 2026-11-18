// SPDX-License-Identifier: Apache-2.0
// Media stays local; only same-origin models, runtime and sample are fetched.
import { createOpticalRef, sampleGrayBilinear, trackPlateOpticalCore } from './optical_tracker.mjs';

const $ = id => document.getElementById(id);
const video = $('video');
const overlay = $('overlay');
const ctx = overlay.getContext('2d');
video.controls = true;

// Failure to isolate is harmless: the Worker automatically uses one WASM thread.
if (
  !window.crossOriginIsolated &&
  'serviceWorker' in navigator &&
  (location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1')
) {
  navigator.serviceWorker
    .register('./coi-sw.js')
    .then(() => navigator.serviceWorker.ready)
    .then(() => {
      const reload = () => {
        try {
          if (navigator.serviceWorker.controller && !sessionStorage.getItem('vpocr_coi_reloaded')) {
            sessionStorage.setItem('vpocr_coi_reloaded', '1');
            location.reload();
          }
        } catch {
          /* Storage disabled: retain the single-thread fallback. */
        }
      };
      navigator.serviceWorker.addEventListener('controllerchange', reload);
      reload();
    })
    .catch(() => {});
}

// Live browser defaults wait for either all-4-field clarity or exit so approaching cars aren't cut off early.
const options = {
  autoSealFrames: 12,
  autoSealMinSec: 0.9,
  autoSealMinPrecision: 0.7,
  autoExitSec: 1.10,
  requireFullFieldsForEarlySeal: true,
  discardShortUnreadable: true,
};
const canTransferBitmap =
  typeof OffscreenCanvas === 'function' && typeof createImageBitmap === 'function';
const captureCanvas = document.createElement('canvas');
const captureContext = captureCanvas.getContext('2d', { willReadFrequently: true });
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

const tell = text => {
  const el = $('status');
  if (el) el.textContent = text;
};
const setStat = (id, text) => {
  const el = $(id);
  if (el) el.textContent = text;
};
const report = error => {
  setStat('stat-phase', 'ERROR');
  tell(`処理できませんでした: ${error.message}`);
};

let worker, loading, sequence = 0, stream, objectURL, mode = 'video', ready = false, collecting = false;
let workerBusy = false, captureEpoch = 0, clipEpoch = 0, lastCapturedTime = -Infinity, lastPayload = null;
let framePending = Promise.resolve(), actions = Promise.resolve(), blockers = 0, seekingPending = false;
let internalSeek = null, thoroughPlaying = false, lastStepWall = -Infinity;
let latencyEma = 0, infFpsEma = 0, lastInfWall = 0;

// Smooth display state eases small measurement corrections over ~85 ms at 60 FPS.
const smoothVehicles = new Map();
const smoothPlates = new Map();
const waiting = new Map();
const results = new Map();

function resetTelemetry() {
  latencyEma = 0;
  infFpsEma = 0;
  lastInfWall = 0;
  setStat('stat-phase', 'IDLE');
  setStat('stat-latency', '-- ms');
  setStat('stat-fps', '--');
  setStat('stat-vehicles', '0');
  setStat('stat-tracks', '0');
  renderLiveTracks(null);
}

function rpc(type, data = {}, transfer = []) {
  return new Promise((resolve, reject) => {
    const id = ++sequence;
    waiting.set(id, { resolve, reject });
    worker.postMessage({ id, type, ...data }, transfer);
  });
}

async function init() {
  if (!loading) {
    loading = (async () => {
      setStat('stat-phase', 'LOADING');
      tell('ローカルモデルを準備しています…');
      worker = new Worker(new URL('./worker.mjs', import.meta.url), { type: 'module' });
      worker.onmessage = ({ data }) => {
        if (data.type === 'progress') {
          if (typeof data.message === 'string') tell(data.message);
          return;
        }
        const pending = waiting.get(data.id);
        if (!pending) return;
        waiting.delete(data.id);
        data.error ? pending.reject(Error(data.error)) : pending.resolve(data.result);
      };
      worker.onerror = event => {
        for (const pending of waiting.values()) pending.reject(Error(event.message || 'Worker error'));
        waiting.clear();
        worker.terminate();
        worker = null;
        loading = null;
        ready = false;
        collecting = false;
        controls();
      };
      const info = await rpc('init', { options });
      if (info && typeof info.threads === 'number') {
        setStat('stat-threads', `SIMD · ${info.threads}T${info.isolated ? ' COI' : ''}`);
      }
    })();
  }
  try {
    await loading;
  } catch (error) {
    worker?.terminate();
    worker = null;
    loading = null;
    throw error;
  }
}

function controls() {
  video.controls = true;
  if ($('stop')) $('stop').disabled = !collecting;
  for (const id of ['pick-file', 'file', 'sample', 'camera']) {
    if ($(id)) $(id).disabled = blockers > 0;
  }
}

function action(fn) {
  blockers++;
  controls();
  const task = actions.then(async () => {
    await framePending;
    return fn();
  });
  actions = task.catch(report).finally(() => {
    blockers--;
    controls();
  });
  return actions;
}

function clearOverlay() {
  lastPayload = null;
  smoothVehicles.clear();
  smoothPlates.clear();
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function shutdownCamera() {
  stream?.getTracks().forEach(t => t.stop());
  stream = null;
  video.srcObject = null;
}

function clearURL() {
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = null;
}

function mediaReady() {
  if (video.readyState >= 2) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener('loadeddata', done);
      video.removeEventListener('error', failed);
    };
    const done = () => {
      cleanup();
      resolve();
    };
    const failed = () => {
      cleanup();
      reject(Error('動画を読み込めません。'));
    };
    const timer = setTimeout(failed, 10000);
    video.addEventListener('loadeddata', done);
    video.addEventListener('error', failed);
  });
}

async function finish() {
  if (!collecting) return;
  setStat('stat-phase', 'SEALED');
  tell('SEALED · 各トラックを一度だけ読み出しています…');
  const terminal = await rpc('finalize');
  collecting = false;
  renderResults(terminal);
  renderLiveTracks(null);
  controls();
  setStat('stat-phase', 'FINALIZED');
  tell(`FINALIZED · ${results.size} 件 · ${Math.round(latencyEma || lastPayload?.latencyMs || 0)} ms/frame`);
}

async function resumeSegment() {
  const payload = await rpc('resegment', { options });
  renderResults(payload.finalizedBeforeSeek);
  collecting = true;
  lastCapturedTime = -Infinity;
  setStat('stat-phase', 'COLLECTING');
  controls();
}

function extrapolateCorners(corners, vel, sizeVel, dt) {
  const mx = 0.25 * (corners[0][0] + corners[1][0] + corners[2][0] + corners[3][0]);
  const my = 0.25 * (corners[0][1] + corners[1][1] + corners[2][1] + corners[3][1]);
  const xs = [corners[0][0], corners[1][0], corners[2][0], corners[3][0]];
  const ys = [corners[0][1], corners[1][1], corners[2][1], corners[3][1]];
  const bw = Math.max(8, Math.max(...xs) - Math.min(...xs));
  const bh = Math.max(4, Math.max(...ys) - Math.min(...ys));
  const sx = clamp(1 + ((sizeVel?.[0] ?? 0) / bw) * dt * 0.45, 0.82, 1.30);
  const sy = clamp(1 + ((sizeVel?.[1] ?? 0) / bh) * dt * 0.45, 0.82, 1.30);
  const dx = (vel?.[0] ?? 0) * dt;
  const dy = (vel?.[1] ?? 0) * dt;
  return corners.map(([x, y]) => [mx + (x - mx) * sx + dx, my + (y - my) * sy + dy]);
}

function updateSmoothingTargets(payload, wallNow) {
  const payloadTime = payload.timestamp;
  const activeV = new Set();
  for (const v of payload.vehicles) {
    activeV.add(v.id);
    const prev = smoothVehicles.get(v.id);
    let offset = [0, 0, 0, 0];
    if (prev) {
      const dtPrev = clamp(payloadTime - prev.time, 0, 0.4);
      const decay = Math.exp(-(wallNow - prev.wall) / 75);
      const predPrev = [
        prev.box[0] + prev.vel[0] * dtPrev - 0.5 * prev.vel[2] * dtPrev + prev.offset[0] * decay,
        prev.box[1] + prev.vel[1] * dtPrev - 0.5 * prev.vel[3] * dtPrev + prev.offset[1] * decay,
        prev.box[2] + prev.vel[0] * dtPrev + 0.5 * prev.vel[2] * dtPrev + prev.offset[2] * decay,
        prev.box[3] + prev.vel[1] * dtPrev + 0.5 * prev.vel[3] * dtPrev + prev.offset[3] * decay,
      ];
      offset = v.box.map((val, i) => clamp(predPrev[i] - val, -80, 80));
    }
    smoothVehicles.set(v.id, {
      box: v.box,
      vel: v.velocity ?? [0, 0, 0, 0],
      time: payloadTime,
      wall: wallNow,
      offset,
    });
  }
  for (const key of smoothVehicles.keys()) if (!activeV.has(key)) smoothVehicles.delete(key);

  const activeP = new Set();
  for (const t of [...payload.tracks, ...payload.finalizedTracks]) {
    activeP.add(t.id);
    const prev = smoothPlates.get(t.id);
    let cornersOffset = t.corners.map(() => [0, 0]);
    if (prev) {
      const dtPrev = clamp(payloadTime - prev.time, 0, 0.4);
      const decay = Math.exp(-(wallNow - prev.wall) / 70);
      const predC = extrapolateCorners(prev.corners, prev.vel, prev.sizeVel, dtPrev);
      cornersOffset = t.corners.map(([nx, ny], k) => [
        clamp(predC[k][0] + prev.cornersOffset[k][0] * decay - nx, -48, 48),
        clamp(predC[k][1] + prev.cornersOffset[k][1] * decay - ny, -48, 48),
      ]);
    }
    const optRef = t.optPatch ? createOpticalRef(t.optPatch, t.optCorners ?? t.corners) : prev?.optRef ?? null;
    smoothPlates.set(t.id, {
      corners: t.corners,
      box: t.box,
      vel: t.velocity ?? [0, 0],
      sizeVel: t.sizeVelocity ?? [0, 0],
      time: payloadTime,
      wall: wallNow,
      cornersOffset,
      optRef,
      dirty: true,
      optLocked: prev?.optLocked ?? false,
      liveCorners: prev?.liveCorners ?? null,
      liveVel: prev?.liveVel ?? (t.velocity ?? [0, 0]),
      lastLiveKey: prev?.lastLiveKey ?? -1,
      lastDt: prev?.lastDt ?? 0,
      lastLiveWall: prev?.lastLiveWall ?? wallNow,
      skewOffset: prev?.skewOffset ?? t.corners.map(() => [0, 0]),
      skewWall: prev?.skewWall ?? wallNow,
    });
  }
  for (const key of smoothPlates.keys()) if (!activeP.has(key)) smoothPlates.delete(key);
}

function renderLiveTracks(payload) {
  const container = $('live-tracks');
  if (!container) return;
  if (!payload || (!payload.tracks.length && !payload.vehicles.length)) {
    container.replaceChildren(
      element('p', 'muted empty-hint', collecting ? '車両・プレートを探索中…' : '追跡中の車両・プレートはありません。')
    );
    return;
  }
  const nodes = [];
  for (const v of payload.vehicles) {
    const label = `V#${v.id} · TRACKING`;
    nodes.push(element('span', 'pill', label));
  }
  for (const t of payload.tracks) {
    const ownerTag = t.vehicleId ? `V#${t.vehicleId} · ` : '';
    const stateTag = t.coasted ? 'COAST' : 'COLLECTING';
    nodes.push(element('span', 'pill', `#${t.id} · ${ownerTag}${t.frames}f · ${stateTag} (0 readouts)`));
  }
  container.replaceChildren(...nodes);
}

function capture(timestamp, epoch = captureEpoch) {
  if (workerBusy || !ready || !collecting || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
    return Promise.resolve();
  }
  workerBusy = true;
  lastCapturedTime = timestamp;
  const clip = clipEpoch;
  framePending = (async () => {
    const scale = Math.min(1, 1920 / Math.max(video.videoWidth, video.videoHeight));
    const wOrig = Math.max(1, Math.round(video.videoWidth * scale));
    const hOrig = Math.max(1, Math.round(video.videoHeight * scale));
    let payload;
    if (canTransferBitmap && mode === 'video') {
      try {
        const bitmap = await createImageBitmap(video, { resizeWidth: wOrig, resizeHeight: hOrig });
        payload = await rpc('frame', { bitmap, wOrig, hOrig, timestamp }, [bitmap]);
      } catch {
        if (captureCanvas.width !== wOrig || captureCanvas.height !== hOrig) {
          captureCanvas.width = wOrig;
          captureCanvas.height = hOrig;
        }
        captureContext.drawImage(video, 0, 0, wOrig, hOrig);
        const rgbaBuffer = captureContext.getImageData(0, 0, wOrig, hOrig).data;
        payload = await rpc('frame', { rgbaBuffer, wOrig, hOrig, timestamp }, [rgbaBuffer.buffer]);
      }
    } else {
      if (captureCanvas.width !== wOrig || captureCanvas.height !== hOrig) {
        captureCanvas.width = wOrig;
        captureCanvas.height = hOrig;
      }
      captureContext.drawImage(video, 0, 0, wOrig, hOrig);
      const rgbaBuffer = captureContext.getImageData(0, 0, wOrig, hOrig).data;
      payload = await rpc('frame', { rgbaBuffer, wOrig, hOrig, timestamp }, [rgbaBuffer.buffer]);
    }
    if (clip !== clipEpoch) return;
    renderResults(payload.newlyFinalized);
    if (epoch !== captureEpoch) return;
    const now = performance.now();
    lastPayload = payload;
    updateSmoothingTargets(payload, now);

    latencyEma = latencyEma ? 0.8 * latencyEma + 0.2 * payload.latencyMs : payload.latencyMs;
    if (lastInfWall > 0) {
      const instFps = 1000 / Math.max(8, now - lastInfWall);
      infFpsEma = infFpsEma ? 0.78 * infFpsEma + 0.22 * instFps : instFps;
    }
    lastInfWall = now;

    setStat('stat-phase', 'COLLECTING');
    setStat('stat-latency', `${Math.round(latencyEma)} ms`);
    setStat('stat-fps', `${infFpsEma ? infFpsEma.toFixed(1) : '--'} inf/s`);
    setStat('stat-vehicles', String(payload.vehicles.length));
    setStat('stat-tracks', `${payload.tracks.length} / ${results.size}`);
    renderLiveTracks(payload);
    tell(
      `COLLECTING · ${payload.frameIndex} フレーム · 車両 ${payload.vehicles.length} · 確定 ${results.size} 件 · ${Math.round(latencyEma)} ms`
    );
  })()
    .catch(report)
    .finally(() => {
      workerBusy = false;
    });
  return framePending;
}

async function choose(source, isFile = false, camera = false) {
  captureEpoch++;
  clipEpoch++;
  ready = false;
  seekingPending = false;
  thoroughPlaying = false;
  internalSeek = null;
  video.pause();
  await action(async () => {
    await finish();
    shutdownCamera();
    clearURL();
    clearOverlay();
    resetTelemetry();
    mode = camera ? 'camera' : 'video';
    results.clear();
    $('results-cards').replaceChildren(
      element('p', 'muted empty-hint', '安定・退出したトラックの確定結果を順次表示します。')
    );
    $('results-json').textContent = '[]';
    if (camera) {
      if (!navigator.mediaDevices?.getUserMedia) throw Error('カメラには HTTPS または localhost が必要です。');
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
      } catch (error) {
        if (!['OverconstrainedError', 'ConstraintNotSatisfiedError', 'NotFoundError'].includes(error.name)) {
          throw error;
        }
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }
      video.removeAttribute('src');
      video.srcObject = stream;
    } else {
      if (isFile) {
        objectURL = URL.createObjectURL(source);
        video.src = objectURL;
      } else {
        video.src = source;
      }
      video.load();
    }
    await Promise.all([init(), mediaReady()]);
    await rpc('reset', { options });
    collecting = true;
    ready = true;
    seekingPending = false;
    lastCapturedTime = -Infinity;
    video.playbackRate = 1;
    setStat('stat-phase', 'COLLECTING');
    controls();
    await capture(mode === 'camera' ? performance.now() / 1000 : video.currentTime);
    await video.play();
  });
}

$('sample').onclick = () => choose('samples/synthetic_gate_demo.mp4');
$('pick-file').onclick = () => $('file').click();
$('file').onchange = event => {
  const file = event.target.files[0];
  if (file) choose(file, true);
  event.target.value = '';
};
$('camera').onclick = () => choose(null, false, true);

for (const name of ['dragover', 'dragleave', 'drop']) {
  $('drop').addEventListener(name, event => {
    event.preventDefault();
    $('drop').classList.toggle('drag', name === 'dragover');
    if (name === 'drop' && !blockers) {
      const file = event.dataTransfer.files[0];
      if (file) choose(file, true);
    }
  });
}

$('stop').onclick = () => {
  captureEpoch++;
  thoroughPlaying = false;
  video.pause();
  action(async () => {
    await finish();
    if (mode === 'camera') {
      shutdownCamera();
      ready = false;
    }
  });
};

video.addEventListener('seeking', () => {
  if (internalSeek !== null && Math.abs(video.currentTime - internalSeek.target) < 0.025) {
    internalSeek.recognized = true;
    return;
  }
  internalSeek = null;
  captureEpoch++;
  seekingPending = true;
  clearOverlay();
});

video.addEventListener('seeked', () => {
  if (internalSeek?.recognized && internalSeek.epoch === captureEpoch) return;
  if (!ready || mode === 'camera') return;
  const epoch = captureEpoch;
  action(async () => {
    if (epoch !== captureEpoch || video.seeking) return;
    await resumeSegment();
    if (epoch !== captureEpoch) return;
    lastCapturedTime = video.currentTime;
    await capture(video.currentTime, epoch);
    if (epoch === captureEpoch) seekingPending = false;
  });
});

video.addEventListener('ended', () => {
  thoroughPlaying = false;
  const epoch = captureEpoch;
  action(async () => {
    if (epoch === captureEpoch && !seekingPending) await finish();
  });
});

video.addEventListener('play', () => {
  if (!ready) return;
  thoroughPlaying = true;
  if (mode === 'video') {
    action(async () => {
      if (!collecting && !seekingPending) await resumeSegment();
    });
  }
});

video.addEventListener('pause', () => {
  thoroughPlaying = false;
});

function syncOverlayRect() {
  const vw = video.videoWidth || 16;
  const vh = video.videoHeight || 9;
  const cw = video.clientWidth || vw;
  const ch = video.clientHeight || vh;
  const scale = Math.min(cw / vw, ch / vh);
  const rw = vw * scale;
  const rh = vh * scale;
  const ox = (cw - rw) / 2;
  const oy = (ch - rh) / 2;
  overlay.style.left = `${ox}px`;
  overlay.style.top = `${oy}px`;
  overlay.style.width = `${rw}px`;
  overlay.style.height = `${rh}px`;
}

const maskCanvas = document.createElement('canvas');
maskCanvas.width = 64;
maskCanvas.height = 64;
const maskContext = maskCanvas.getContext('2d');
const maskImage = maskContext.createImageData(64, 64);

// Dedicated tiny ROI canvas for 60 FPS sub-millisecond direct video optical lock-on.
const OPT_ROI_MAX_W = 192;
const OPT_ROI_MAX_H = 128;
const optCanvas = document.createElement('canvas');
optCanvas.width = OPT_ROI_MAX_W;
optCanvas.height = OPT_ROI_MAX_H;
const optCtx = optCanvas.getContext('2d', { willReadFrequently: true });
const optGrayBuf = new Float32Array(OPT_ROI_MAX_W * OPT_ROI_MAX_H);

function alignPlateToLiveVideo(sm, liveKey, dt, w, h, wallNow) {
  if (!sm?.optRef || video.readyState < 2 || !video.videoWidth || !video.videoHeight || video.seeking) {
    return null;
  }
  if (!sm.dirty && sm.lastLiveKey === liveKey && sm.optLocked && sm.liveCorners) {
    const skewDecay = Math.exp(-(wallNow - sm.skewWall) / 42);
    return sm.liveCorners.map(([x, y], k) => [
      clamp(x + sm.skewOffset[k][0] * skewDecay, 0, w),
      clamp(y + sm.skewOffset[k][1] * skewDecay, 0, h),
    ]);
  }

  const kinCorners = extrapolateCorners(sm.corners, sm.vel, sm.sizeVel, Math.max(0, dt));
  const kxs = [kinCorners[0][0], kinCorners[1][0], kinCorners[2][0], kinCorners[3][0]];
  const kys = [kinCorners[0][1], kinCorners[1][1], kinCorners[2][1], kinCorners[3][1]];
  const kw = Math.max(8, Math.max(...kxs) - Math.min(...kxs));
  const kh = Math.max(4, Math.max(...kys) - Math.min(...kys));
  const kinMx = 0.25 * (kinCorners[0][0] + kinCorners[1][0] + kinCorners[2][0] + kinCorners[3][0]);
  const kinMy = 0.25 * (kinCorners[0][1] + kinCorners[1][1] + kinCorners[2][1] + kinCorners[3][1]);

  let guessCorners;
  let searchRadiusPx;
  let allowScaleSearch;
  let reuseLive = false;
  if (sm.liveCorners && sm.optLocked && Math.abs(dt - sm.lastDt) < 0.24) {
    const liveMx = 0.25 * (sm.liveCorners[0][0] + sm.liveCorners[1][0] + sm.liveCorners[2][0] + sm.liveCorners[3][0]);
    const liveMy = 0.25 * (sm.liveCorners[0][1] + sm.liveCorners[1][1] + sm.liveCorners[2][1] + sm.liveCorners[3][1]);
    // Self-heal immediately if previous live lock ever strayed too far from worker kinematic anchor.
    if (Math.abs(liveMx - kinMx) <= Math.max(26, 1.45 * kw) && Math.abs(liveMy - kinMy) <= Math.max(16, 1.15 * kh)) {
      reuseLive = true;
    }
  }
  if (reuseLive) {
    const stepDt = clamp(dt - sm.lastDt, -0.04, 0.08);
    guessCorners = sm.liveCorners.map(([x, y]) => [x + sm.liveVel[0] * stepDt, y + sm.liveVel[1] * stepDt]);
    searchRadiusPx = sm.dirty ? 24 : 15;
    allowScaleSearch = sm.dirty;
  } else {
    guessCorners = kinCorners;
    searchRadiusPx = clamp(20 + 200 * Math.abs(dt), 16, 52);
    allowScaleSearch = true;
  }

  const gxs = [guessCorners[0][0], guessCorners[1][0], guessCorners[2][0], guessCorners[3][0]];
  const gys = [guessCorners[0][1], guessCorners[1][1], guessCorners[2][1], guessCorners[3][1]];
  const gwBox = Math.max(8, Math.max(...gxs) - Math.min(...gxs));
  const ghBox = Math.max(4, Math.max(...gys) - Math.min(...gys));
  const padX = Math.max(32, 0.95 * gwBox + searchRadiusPx);
  const padY = Math.max(24, 1.15 * ghBox + 0.85 * searchRadiusPx);
  const rx0 = clamp(Math.floor(Math.min(...gxs) - padX), 0, w - 16);
  const ry0 = clamp(Math.floor(Math.min(...gys) - padY), 0, h - 12);
  const rx1 = clamp(Math.ceil(Math.max(...gxs) + padX), rx0 + 16, w);
  const ry1 = clamp(Math.ceil(Math.max(...gys) + padY), ry0 + 12, h);
  const rw = rx1 - rx0;
  const rh = ry1 - ry0;

  const roiScale = Math.min(1, OPT_ROI_MAX_W / rw, OPT_ROI_MAX_H / rh);
  const bufW = clamp(Math.round(rw * roiScale), 24, OPT_ROI_MAX_W);
  const bufH = clamp(Math.round(rh * roiScale), 16, OPT_ROI_MAX_H);
  const vScaleX = video.videoWidth / w;
  const vScaleY = video.videoHeight / h;

  let rgba;
  try {
    optCtx.drawImage(video, rx0 * vScaleX, ry0 * vScaleY, rw * vScaleX, rh * vScaleY, 0, 0, bufW, bufH);
    rgba = optCtx.getImageData(0, 0, bufW, bufH).data;
  } catch {
    return null;
  }

  const pixels = bufW * bufH;
  for (let i = 0, p = 0; i < pixels; i++, p += 4) {
    optGrayBuf[i] = 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
  }

  const sxRoi = bufW / rw;
  const syRoi = bufH / rh;
  const sampleRoi = (nx, ny) => sampleGrayBilinear(optGrayBuf, bufW, bufH, (nx - rx0) * sxRoi, (ny - ry0) * syRoi);
  const res = trackPlateOpticalCore(sm.optRef, sampleRoi, w, h, guessCorners, {
    searchRadiusPx,
    allowScaleSearch,
    minZncc: sm.optRef.basis.width < 36 ? 0.50 : 0.60,
    maxIter: 6,
  });

  sm.lastLiveKey = liveKey;
  if (!res.ok) {
    sm.optLocked = false;
    sm.dirty = false;
    return null;
  }

  if (sm.liveCorners && sm.optLocked) {
    if (sm.dirty) {
      const prevDecay = Math.exp(-(wallNow - sm.skewWall) / 42);
      sm.skewOffset = res.corners.map(([nx, ny], k) => [
        clamp(sm.liveCorners[k][0] + sm.skewOffset[k][0] * prevDecay - nx, -10, 10),
        clamp(sm.liveCorners[k][1] + sm.skewOffset[k][1] * prevDecay - ny, -10, 10),
      ]);
      sm.skewWall = wallNow;
    } else {
      const stepSec = Math.max(0.012, (wallNow - sm.lastLiveWall) / 1000);
      const oldMx = 0.25 * (sm.liveCorners[0][0] + sm.liveCorners[1][0] + sm.liveCorners[2][0] + sm.liveCorners[3][0]);
      const oldMy = 0.25 * (sm.liveCorners[0][1] + sm.liveCorners[1][1] + sm.liveCorners[2][1] + sm.liveCorners[3][1]);
      const newMx = 0.25 * (res.corners[0][0] + res.corners[1][0] + res.corners[2][0] + res.corners[3][0]);
      const newMy = 0.25 * (res.corners[0][1] + res.corners[1][1] + res.corners[2][1] + res.corners[3][1]);
      sm.liveVel = [
        clamp(0.55 * sm.liveVel[0] + 0.45 * ((newMx - oldMx) / stepSec), -w, w),
        clamp(0.55 * sm.liveVel[1] + 0.45 * ((newMy - oldMy) / stepSec), -h, h),
      ];
    }
  } else {
    sm.skewOffset = res.corners.map(() => [0, 0]);
    sm.skewWall = wallNow;
  }

  sm.liveCorners = res.corners;
  sm.optLocked = true;
  sm.dirty = false;
  sm.lastDt = dt;
  sm.lastLiveWall = wallNow;

  const skewDecay = Math.exp(-(wallNow - sm.skewWall) / 42);
  return res.corners.map(([x, y], k) => [
    clamp(x + sm.skewOffset[k][0] * skewDecay, 0, w),
    clamp(y + sm.skewOffset[k][1] * skewDecay, 0, h),
  ]);
}

function paintSmooth(wallNow) {
  if (!lastPayload) return;
  const payload = lastPayload;
  const { wOrig: w, hOrig: h } = payload;
  if (overlay.width !== w || overlay.height !== h) {
    overlay.width = w;
    overlay.height = h;
    syncOverlayRect();
  }
  ctx.clearRect(0, 0, w, h);

  // Align overlay boxes to current playback timestamp via direct 60 FPS optical lock + kinematic fallback.
  let dt = 0;
  if (!video.seeking && collecting) {
    dt =
      mode === 'camera'
        ? clamp(wallNow / 1000 - payload.timestamp, 0, 0.25)
        : clamp(video.currentTime - payload.timestamp, -0.05, 0.42);
  }
  const liveKey = mode === 'camera' ? Math.floor(wallNow / 16) : Math.round(video.currentTime * 1000);

  const unit = Math.max(1, w / 768);
  const fontSize = Math.round(13 * unit);
  ctx.font = `600 ${fontSize}px ui-monospace, monospace`;

  // 1. Solve 60 FPS live plate corners first so owned vehicles can couple rigidly to live optical motion.
  const allPlates = [...payload.tracks, ...payload.finalizedTracks];
  const resolvedPlates = [];
  const vehicleLiveDelta = new Map();

  for (const track of allPlates) {
    const sm = smoothPlates.get(track.id);
    let corners = track.corners;
    const optCorners = sm ? alignPlateToLiveVideo(sm, liveKey, dt, w, h, wallNow) : null;
    if (optCorners) {
      corners = optCorners;
      if (track.vehicleId != null && sm?.box) {
        const anchorCx = 0.5 * (sm.box[0] + sm.box[2]);
        const anchorCy = 0.5 * (sm.box[1] + sm.box[3]);
        const anchorW = Math.max(8, sm.box[2] - sm.box[0]);
        const lxs = [corners[0][0], corners[1][0], corners[2][0], corners[3][0]];
        const lys = [corners[0][1], corners[1][1], corners[2][1], corners[3][1]];
        const liveCx = 0.5 * (Math.min(...lxs) + Math.max(...lxs));
        const liveCy = 0.5 * (Math.min(...lys) + Math.max(...lys));
        const liveW = Math.max(8, Math.max(...lxs) - Math.min(...lxs));
        const scaleRatio = clamp(liveW / anchorW, 0.80, 1.28);
        vehicleLiveDelta.set(track.vehicleId, {
          anchorCx,
          anchorCy,
          dx: liveCx - anchorCx,
          dy: liveCy - anchorCy,
          scale: scaleRatio,
        });
      }
    } else if (sm) {
      const decay = Math.exp(-(wallNow - sm.wall) / 70);
      const extrap = extrapolateCorners(sm.corners, sm.vel, sm.sizeVel, Math.max(0, dt));
      corners = extrap.map(([cx, cy], k) => [
        clamp(cx + sm.cornersOffset[k][0] * decay, 0, w),
        clamp(cy + sm.cornersOffset[k][1] * decay, 0, h),
      ]);
    }
    resolvedPlates.push({ track, corners });
  }

  // 2. Render vehicles (coupled to live optical plate delta when available).
  for (const v of payload.vehicles) {
    const sm = smoothVehicles.get(v.id);
    const coupled = vehicleLiveDelta.get(v.id);
    let box = v.box;
    if (coupled && sm) {
      const decay = Math.exp(-(wallNow - sm.wall) / 75);
      const { anchorCx, anchorCy, dx, dy, scale } = coupled;
      box = [
        clamp(anchorCx + (sm.box[0] - anchorCx) * scale + dx + sm.offset[0] * decay, 0, w),
        clamp(anchorCy + (sm.box[1] - anchorCy) * scale + dy + sm.offset[1] * decay, 0, h),
        clamp(anchorCx + (sm.box[2] - anchorCx) * scale + dx + sm.offset[2] * decay, 0, w),
        clamp(anchorCy + (sm.box[3] - anchorCy) * scale + dy + sm.offset[3] * decay, 0, h),
      ];
    } else if (sm) {
      const decay = Math.exp(-(wallNow - sm.wall) / 75);
      const [vx, vy, vw, vh] = sm.vel;
      const posDt = Math.max(0, dt);
      box = [
        clamp(sm.box[0] + vx * posDt - 0.5 * vw * posDt * 0.4 + sm.offset[0] * decay, 0, w),
        clamp(sm.box[1] + vy * posDt - 0.5 * vh * posDt * 0.4 + sm.offset[1] * decay, 0, h),
        clamp(sm.box[2] + vx * posDt + 0.5 * vw * posDt * 0.4 + sm.offset[2] * decay, 0, w),
        clamp(sm.box[3] + vy * posDt + 0.5 * vh * posDt * 0.4 + sm.offset[3] * decay, 0, h),
      ];
    }
    const [a, b, c, d] = box;
    const bw = c - a, bh = d - b;
    if (bw < 4 || bh < 4) continue;
    if (v.mask) {
      const data = maskImage.data;
      for (let i = 0; i < 4096; i++) {
        const k = i * 4;
        if (v.mask[i] > 0) {
          data[k] = 56;
          data[k + 1] = 224;
          data[k + 2] = 242;
          data[k + 3] = 52;
        } else {
          data[k + 3] = 0;
        }
      }
      maskContext.putImageData(maskImage, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(maskCanvas, a, b, bw, bh);
    }
    ctx.strokeStyle = 'rgba(65, 222, 242, 0.85)';
    ctx.lineWidth = 1.6 * unit;
    ctx.strokeRect(a, b, bw, bh);
    const vLabel = `V#${v.id}`;
    const vtw = ctx.measureText(vLabel).width;
    const lx = clamp(a, 0, w - vtw - 10 * unit);
    const ly = Math.max(fontSize + 8 * unit, b);
    ctx.fillStyle = 'rgba(7, 19, 30, 0.82)';
    ctx.fillRect(lx, ly - fontSize - 6 * unit, vtw + 10 * unit, fontSize + 6 * unit);
    ctx.fillStyle = '#5eead4';
    ctx.fillText(vLabel, lx + 5 * unit, ly - 4 * unit);
  }

  // 3. Render optically locked 4-corner plate quads on top.
  for (const { track, corners } of resolvedPlates) {
    const finalized = track.readouts === 1;
    const color = finalized ? (track.status === 'ACCEPTED' ? '#6ee7b7' : '#fbbf24') : '#a3e635';
    const minX = Math.min(corners[0][0], corners[1][0], corners[2][0], corners[3][0]);
    const minY = Math.min(corners[0][1], corners[1][1], corners[2][1], corners[3][1]);

    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 2.2 * unit;
    ctx.beginPath();
    corners.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.stroke();
    for (const [x, y] of corners) {
      ctx.beginPath();
      ctx.arc(x, y, 2.4 * unit, 0, Math.PI * 2);
      ctx.fill();
    }
    const label = finalized
      ? `#${track.id} · ${track.status === 'ACCEPTED' ? track.text : 'UNREADABLE'}`
      : `#${track.id} · ${track.frames}f · COLLECTING`;
    const tw = ctx.measureText(label).width;
    const tx = clamp(minX, 0, Math.max(0, w - tw - 12 * unit));
    const ty = Math.max(fontSize + 10 * unit, minY - 6 * unit);
    ctx.fillStyle = 'rgba(7, 19, 30, 0.9)';
    ctx.fillRect(tx, ty - fontSize - 6 * unit, tw + 12 * unit, fontSize + 8 * unit);
    ctx.fillStyle = color;
    ctx.fillText(label, tx + 6 * unit, ty - 3 * unit);
  }
}

// 60 FPS render & capture scheduling: playback never blocks on WASM inference.
function tick(wallNow) {
  requestAnimationFrame(tick);
  if (lastPayload && ready && !seekingPending) {
    paintSmooth(wallNow);
  }
  if (!ready || blockers || workerBusy || seekingPending || video.seeking || video.ended || !collecting) return;
  if (mode === 'video' && $('thorough').checked) {
    if (!video.paused && performance.now() - lastStepWall >= 1000 / 12) stepThorough();
    return;
  }
  if (video.paused) return;
  const time = mode === 'camera' ? performance.now() / 1000 : video.currentTime;
  if (Math.abs(time - lastCapturedTime) >= 0.04) capture(time);
}

// Decoded-frame callback reduces video sampling jitter when available.
function decodedFrame() {
  if (
    ready &&
    !blockers &&
    !workerBusy &&
    !seekingPending &&
    !video.seeking &&
    !video.paused &&
    !video.ended &&
    collecting &&
    !$('thorough').checked
  ) {
    const time = mode === 'camera' ? performance.now() / 1000 : video.currentTime;
    if (Math.abs(time - lastCapturedTime) >= 0.04) capture(time);
  }
  video.requestVideoFrameCallback(decodedFrame);
}

async function stepThorough() {
  if (blockers) return;
  const epoch = captureEpoch;
  thoroughPlaying = true;
  lastStepWall = performance.now();
  await action(async () => {
    if (epoch !== captureEpoch || video.paused) return;
    const next = lastCapturedTime + 1 / 12;
    if (next >= video.duration) {
      thoroughPlaying = false;
      await finish();
      return;
    }
    internalSeek = { target: next, epoch, recognized: false };
    try {
      await new Promise((resolve, reject) => {
        const done = () => {
          cleanup();
          resolve();
        };
        const failed = () => {
          cleanup();
          reject(Error('Seek timed out'));
        };
        const cleanup = () => {
          clearTimeout(timer);
          video.removeEventListener('seeked', done);
        };
        const timer = setTimeout(failed, 5000);
        video.addEventListener('seeked', done, { once: true });
        video.currentTime = next;
      });
      if (epoch === captureEpoch) await capture(next, epoch);
    } finally {
      internalSeek = null;
    }
  });
}

$('thorough').addEventListener('change', () => {
  if (!$('thorough').checked && thoroughPlaying) {
    thoroughPlaying = false;
    video.play().catch(report);
  }
});

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function prependCard(result) {
  const cards = $('results-cards');
  const placeholder = cards.querySelector('.empty-hint');
  if (placeholder) placeholder.remove();

  const card = element('article', `result-card${result.status === 'ACCEPTED' ? '' : ' unreadable'}`);
  const header = element('div', 'card-top');
  header.append(element('span', 'accepted', result.status), element('span', 'tier', result.confidence_tier));
  card.append(header, element('p', 'plate-text', result.text ?? 'UNREADABLE'));
  if (
    result.status === 'ACCEPTED' &&
    result.best_guess_text &&
    result.best_guess_text !== result.text
  ) {
    card.append(element('p', 'abstained', `全項目候補 (参考): ${result.best_guess_text}`));
  }

  const evidence = element('div', 'evidence-row');
  if (result.thumbnail?.length === 51200) {
    const preview = element('canvas', 'plate-preview');
    preview.width = 160;
    preview.height = 80;
    preview.setAttribute('role', 'img');
    preview.setAttribute('aria-label', '補正済みプレート画像');
    preview.getContext('2d').putImageData(new ImageData(result.thumbnail, 160, 80), 0, 0);
    evidence.append(preview);
  }
  if (result.status === 'ACCEPTED') {
    const pills = element('div', 'confidence-pills');
    for (const [key, label] of Object.entries({
      region: '地域',
      classification: '分類',
      hiragana: 'かな',
      serial: '一連番号',
    })) {
      pills.append(element('span', 'pill', `${label} ${Math.round((result.field_confidences?.[key] ?? 0) * 100)}%`));
    }
    evidence.append(pills);
  } else {
    evidence.append(element('p', 'abstained', '確信度不足のため終端ゲートで棄却されました。'));
  }
  if (evidence.childElementCount) card.append(evidence);

  card.append(
    element('p', 'frame-count', `#${result.track_id} · FINALIZED · ${result.frames} frames · 1 terminal readout`)
  );
  cards.prepend(card);
}

function renderResults(items = []) {
  for (const result of items) {
    if (!results.has(result.track_id)) {
      results.set(result.track_id, result);
      prependCard(result);
    }
  }
  setStat('stat-tracks', `${lastPayload?.tracks?.length ?? 0} / ${results.size}`);
  $('results-json').textContent = JSON.stringify(
    [...results.values()].map(({ thumbnail, ...r }) => r),
    null,
    2
  );
}

new ResizeObserver(() => {
  syncOverlayRect();
  if (lastPayload) paintSmooth(performance.now());
}).observe(video);

window.addEventListener('pagehide', () => {
  captureEpoch++;
  ready = false;
  shutdownCamera();
  clearURL();
  worker?.terminate();
});

resetTelemetry();
requestAnimationFrame(tick);
if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(decodedFrame);
controls();
