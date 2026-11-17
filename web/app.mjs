// SPDX-License-Identifier: Apache-2.0
// Media stays local; only same-origin models, runtime and sample are fetched.
const $=id=>document.getElementById(id);
const video=$('video'),overlay=$('overlay'),ctx=overlay.getContext('2d');
video.controls=true;
// Failure to isolate is harmless: the Worker automatically uses one WASM thread.
if(!window.crossOriginIsolated&&'serviceWorker' in navigator&&
  (location.protocol==='https:'||location.hostname==='localhost'||location.hostname==='127.0.0.1')) {
  navigator.serviceWorker.register('./coi-sw.js').then(()=>navigator.serviceWorker.ready).then(()=>{
    const reload=()=>{
      try {
        if(navigator.serviceWorker.controller&&!sessionStorage.getItem('vpocr_coi_reloaded')) {
          sessionStorage.setItem('vpocr_coi_reloaded','1');location.reload();
        }
      } catch { /* Storage disabled: retain the single-thread fallback. */ }
    };
    navigator.serviceWorker.addEventListener('controllerchange',reload);reload();
  }).catch(()=>{});
}
const options={autoSealFrames:6,autoSealMinSec:.45,autoExitSec:.50};
const captureCanvas=typeof OffscreenCanvas==='function'?new OffscreenCanvas(1,1):document.createElement('canvas');
const captureContext=captureCanvas.getContext('2d',{willReadFrequently:true});
const tell=text=>{$('status').textContent=text;};
const report=error=>tell(`処理できませんでした: ${error.message}`);
let worker,loading,sequence=0,stream,objectURL,mode='video',ready=false,collecting=false;
let workerBusy=false,captureEpoch=0,clipEpoch=0,lastCapturedTime=-Infinity,lastPayload;
let framePending=Promise.resolve(),actions=Promise.resolve(),blockers=0,seekingPending=false;
let internalSeek=null,thoroughPlaying=false,lastStepWall=-Infinity;
const waiting=new Map(),results=new Map();
function rpc(type,data={},transfer=[]) {
  return new Promise((resolve,reject)=>{
    const id=++sequence;waiting.set(id,{resolve,reject});worker.postMessage({id,type,...data},transfer);
  });
}
async function init() {
  if(!loading) loading=(async()=>{
    tell('ローカルモデルを準備しています…');
    worker=new Worker(new URL('./worker.mjs',import.meta.url),{type:'module'});
    worker.onmessage=({data})=>{
      if(data.type==='progress') {if(typeof data.message==='string') tell(data.message);return;}
      const pending=waiting.get(data.id);if(!pending) return;
      waiting.delete(data.id);data.error?pending.reject(Error(data.error)):pending.resolve(data.result);
    };
    worker.onerror=event=>{
      for(const pending of waiting.values()) pending.reject(Error(event.message||'Worker error'));
      waiting.clear();worker.terminate();worker=null;loading=null;ready=false;collecting=false;controls();
    };
    await rpc('init',{options});
  })();
  try {await loading;} catch(error) {worker?.terminate();worker=null;loading=null;throw error;}
}
function controls() {
  video.controls=true;$('stop').disabled=!collecting;
  for(const id of ['pick-file','file','sample','camera']) $(id).disabled=blockers>0;
}
function action(fn) {
  blockers++;controls();
  const task=actions.then(async()=>{await framePending;return fn();});
  actions=task.catch(report).finally(()=>{blockers--;controls();});return actions;
}
function clearOverlay() {lastPayload=null;ctx.clearRect(0,0,overlay.width,overlay.height);}
function shutdownCamera() {stream?.getTracks().forEach(t=>t.stop());stream=null;video.srcObject=null;}
function clearURL() {if(objectURL) URL.revokeObjectURL(objectURL);objectURL=null;}
function mediaReady() {
  if(video.readyState>=2) return Promise.resolve();
  return new Promise((resolve,reject)=>{
    const cleanup=()=>{clearTimeout(timer);video.removeEventListener('loadeddata',done);video.removeEventListener('error',failed);};
    const done=()=>{cleanup();resolve();},failed=()=>{cleanup();reject(Error('動画を読み込めません。'));};
    const timer=setTimeout(failed,10000);video.addEventListener('loadeddata',done);video.addEventListener('error',failed);
  });
}
async function finish() {
  if(!collecting) return;
  tell('SEALED · 各トラックを一度だけ読み出しています…');
  const terminal=await rpc('finalize');collecting=false;renderResults(terminal);controls();
  tell(`FINALIZED · ${results.size} 件 · ${Math.round(lastPayload?.latencyMs??0)} ms/frame`);
}
async function resumeSegment() {
  const payload=await rpc('resegment',{options});renderResults(payload.finalizedBeforeSeek);
  collecting=true;lastCapturedTime=-Infinity;controls();
}
function capture(timestamp,epoch=captureEpoch) {
  if(workerBusy||!ready||!collecting||video.readyState<2) return Promise.resolve();
  workerBusy=true;lastCapturedTime=timestamp;
  const clip=clipEpoch;
  framePending=(async()=>{
    const scale=Math.min(1,1280/Math.max(video.videoWidth,video.videoHeight));
    const wOrig=Math.max(1,Math.round(video.videoWidth*scale)),hOrig=Math.max(1,Math.round(video.videoHeight*scale));
    if(captureCanvas.width!==wOrig||captureCanvas.height!==hOrig) {captureCanvas.width=wOrig;captureCanvas.height=hOrig;}
    captureContext.drawImage(video,0,0,wOrig,hOrig);
    const rgbaBuffer=captureContext.getImageData(0,0,wOrig,hOrig).data;
    const payload=await rpc('frame',{rgbaBuffer,wOrig,hOrig,timestamp},[rgbaBuffer.buffer]);
    if(clip!==clipEpoch) return;
    renderResults(payload.newlyFinalized);
    if(epoch!==captureEpoch) return;
    paint(payload);
    tell(`COLLECTING · ${payload.frameIndex} フレーム · 確定 ${results.size} 件 · ${Math.round(payload.latencyMs)} ms/frame`);
  })().catch(report).finally(()=>{workerBusy=false;});
  return framePending;
}
async function choose(source,isFile=false,camera=false) {
  captureEpoch++;clipEpoch++;ready=false;seekingPending=false;thoroughPlaying=false;internalSeek=null;
  video.pause();
  await action(async()=>{
    await finish();shutdownCamera();clearURL();clearOverlay();mode=camera?'camera':'video';
    results.clear();$('results-cards').replaceChildren();$('results-json').textContent='[]';
    if(camera) {
      if(!navigator.mediaDevices?.getUserMedia) throw Error('カメラには HTTPS または localhost が必要です。');
      try {stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720}},audio:false});}
      catch(error) {
        if(!['OverconstrainedError','ConstraintNotSatisfiedError','NotFoundError'].includes(error.name)) throw error;
        stream=await navigator.mediaDevices.getUserMedia({video:true,audio:false});
      }
      video.removeAttribute('src');video.srcObject=stream;
    } else {
      if(isFile) {objectURL=URL.createObjectURL(source);video.src=objectURL;} else video.src=source;
      video.load();
    }
    await Promise.all([init(),mediaReady()]);
    await rpc('reset',{options});collecting=true;ready=true;seekingPending=false;lastCapturedTime=-Infinity;
    video.playbackRate=1;controls();
    await capture(mode==='camera'?performance.now()/1000:video.currentTime);
    await video.play();
  });
}
$('sample').onclick=()=>choose('samples/synthetic_gate_demo.mp4');
$('pick-file').onclick=()=>$('file').click();
$('file').onchange=event=>{const file=event.target.files[0];if(file) choose(file,true);event.target.value='';};
$('camera').onclick=()=>choose(null,false,true);
for(const name of ['dragover','dragleave','drop']) $('drop').addEventListener(name,event=>{
  event.preventDefault();$('drop').classList.toggle('drag',name==='dragover');
  if(name==='drop'&&!blockers) {const file=event.dataTransfer.files[0];if(file) choose(file,true);}
});
$('stop').onclick=()=>{
  captureEpoch++;thoroughPlaying=false;video.pause();
  action(async()=>{await finish();if(mode==='camera') {shutdownCamera();ready=false;}});
};
video.addEventListener('seeking',()=>{
  if(internalSeek!==null&&Math.abs(video.currentTime-internalSeek.target)<.025) {internalSeek.recognized=true;return;}
  internalSeek=null;captureEpoch++;seekingPending=true;clearOverlay();
});
video.addEventListener('seeked',()=>{
  if(internalSeek?.recognized&&internalSeek.epoch===captureEpoch) return;
  if(!ready||mode==='camera') return;
  const epoch=captureEpoch;
  action(async()=>{
    if(epoch!==captureEpoch||video.seeking) return;
    await resumeSegment();
    if(epoch!==captureEpoch) return;
    lastCapturedTime=video.currentTime;
    await capture(video.currentTime,epoch);
    if(epoch===captureEpoch) seekingPending=false;
  });
});
video.addEventListener('ended',()=>{
  thoroughPlaying=false;const epoch=captureEpoch;
  action(async()=>{if(epoch===captureEpoch&&!seekingPending) await finish();});
});
video.addEventListener('play',()=>{
  if(!ready) return;
  thoroughPlaying=true;
  if(mode==='video') action(async()=>{if(!collecting&&!seekingPending) await resumeSegment();});
});
video.addEventListener('pause',()=>{thoroughPlaying=false;});
// Normal mode never waits for inference to advance playback. Native controls
// remain available during every RPC, including paused seeks and EOF replay.
function tick() {
  requestAnimationFrame(tick);
  if(!ready||blockers||workerBusy||seekingPending||video.seeking||video.ended||!collecting) return;
  if(mode==='video'&&$('thorough').checked) {
    if(!video.paused&&performance.now()-lastStepWall>=1000/12) stepThorough();
    return;
  }
  if(video.paused) return;
  const time=mode==='camera'?performance.now()/1000:video.currentTime;
  if(Math.abs(time-lastCapturedTime)>=.05) capture(time);
}
// A second callback reduces capture jitter on browsers exposing decoded frames.
function decodedFrame() {
  if(ready&&!blockers&&!workerBusy&&!seekingPending&&!video.seeking&&!video.paused&&!video.ended&&collecting&&!$('thorough').checked) {
    const time=mode==='camera'?performance.now()/1000:video.currentTime;
    if(Math.abs(time-lastCapturedTime)>=.05) capture(time);
  }
  video.requestVideoFrameCallback(decodedFrame);
}
async function stepThorough() {
  if(blockers) return;
  const epoch=captureEpoch;
  // Step through the 12 FPS sampling timeline without disabling native controls.
  thoroughPlaying=true;lastStepWall=performance.now();
  await action(async()=>{
    if(epoch!==captureEpoch||video.paused) return;
    const next=lastCapturedTime+1/12;
    if(next>=video.duration) {thoroughPlaying=false;await finish();return;}
    internalSeek={target:next,epoch,recognized:false};
    try {
      await new Promise((resolve,reject)=>{
        const done=()=>{cleanup();resolve();},failed=()=>{cleanup();reject(Error('Seek timed out'));};
        const cleanup=()=>{clearTimeout(timer);video.removeEventListener('seeked',done);};
        const timer=setTimeout(failed,5000);video.addEventListener('seeked',done,{once:true});video.currentTime=next;
      });
      if(epoch===captureEpoch) await capture(next,epoch);
    } finally {internalSeek=null;}
  });
}
$('thorough').addEventListener('change',()=>{
  if(!$('thorough').checked&&thoroughPlaying) {thoroughPlaying=false;video.play().catch(report);}
});
const maskCanvas=document.createElement('canvas');maskCanvas.width=64;maskCanvas.height=64;
const maskContext=maskCanvas.getContext('2d');
function paint(payload) {
  lastPayload=payload;const {wOrig:w,hOrig:h}=payload;
  if(overlay.width!==w||overlay.height!==h) {overlay.width=w;overlay.height=h;}
  ctx.clearRect(0,0,w,h);
  const unit=Math.max(1,w/768),fontSize=14*unit;
  ctx.lineWidth=2*unit;ctx.font=`600 ${fontSize}px ui-monospace,monospace`;
  for(const {box:[a,b,c,d],mask} of payload.vehicles) {
    if(mask) {
      const image=maskContext.createImageData(64,64);
      for(let i=0;i<4096;i++) if(mask[i]>0) image.data.set([55,222,242,60],i*4);
      maskContext.putImageData(image,0,0);ctx.imageSmoothingEnabled=false;
      ctx.drawImage(maskCanvas,a,b,c-a,d-b);ctx.imageSmoothingEnabled=true;
    }
    ctx.strokeStyle='#41def2';ctx.strokeRect(a,b,c-a,d-b);
  }
  for(const track of [...payload.tracks,...payload.finalizedTracks]) {
    const finalized=track.readouts===1,color=finalized?(track.status==='ACCEPTED'?'#7fe2c0':'#f4ce70'):'#b6f773';
    ctx.strokeStyle=color;ctx.fillStyle=color;ctx.beginPath();
    track.corners.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.stroke();
    for(const [x,y] of track.corners) {ctx.beginPath();ctx.arc(x,y,2.5*unit,0,Math.PI*2);ctx.fill();}
    const label=finalized?`#${track.id} · ${track.status==='ACCEPTED'?track.text:'UNREADABLE'}`:`#${track.id} COLLECTING · 0 readouts`;
    const tw=ctx.measureText(label).width,x=Math.max(0,Math.min(w-tw-12*unit,track.box[0]));
    const y=Math.max(fontSize+12*unit,track.box[1]-9*unit);
    ctx.fillStyle='#07131ef0';ctx.fillRect(x,y-fontSize-5*unit,tw+12*unit,fontSize+10*unit);
    ctx.fillStyle=color;ctx.fillText(label,x+6*unit,y);
  }
}
function element(tag,className,text) {
  const node=document.createElement(tag);if(className) node.className=className;
  if(text!==undefined) node.textContent=text;return node;
}
function prependCard(result) {
  const card=element('article',`result-card${result.status==='ACCEPTED'?'':' unreadable'}`);
  const header=element('div','card-top');
  header.append(element('span','accepted',result.status),element('span','tier',result.confidence_tier));
  card.append(header,element('p','plate-text',result.text??'UNREADABLE'));
  if(result.thumbnail?.length===51200) {
    const preview=element('canvas','plate-preview');preview.width=160;preview.height=80;
    preview.setAttribute('role','img');preview.setAttribute('aria-label','補正済みプレート画像');
    preview.getContext('2d').putImageData(new ImageData(result.thumbnail,160,80),0,0);card.append(preview);
  }
  if(result.status==='ACCEPTED') {
    const pills=element('div','confidence-pills');
    for(const [key,label] of Object.entries({region:'地域',classification:'分類',hiragana:'かな',serial:'一連番号'}))
      pills.append(element('span','pill',`${label} ${Math.round((result.field_confidences?.[key]??0)*100)}%`));
    card.append(pills);
  }
  card.append(element('p','frame-count',`#${result.track_id} · FINALIZED · ${result.frames} frames · 1 terminal readout`));
  $('results-cards').prepend(card);
}
function renderResults(items=[]) {
  for(const result of items) if(!results.has(result.track_id)) {results.set(result.track_id,result);prependCard(result);}
  $('results-json').textContent=JSON.stringify([...results.values()].map(({thumbnail,...r})=>r),null,2);
}
new ResizeObserver(()=>{if(lastPayload) paint(lastPayload);}).observe(video);
window.addEventListener('pagehide',()=>{captureEpoch++;ready=false;shutdownCamera();clearURL();worker?.terminate();});
requestAnimationFrame(tick);
if(video.requestVideoFrameCallback) video.requestVideoFrameCallback(decodedFrame);
controls();
