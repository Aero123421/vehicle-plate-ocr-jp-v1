// Apache-2.0 · Main thread: media, native frame transfer and presentation only.
const $=id=>document.getElementById(id);
const video=$('video'),overlay=$('overlay'),ctx=overlay.getContext('2d');
const captureCanvas=typeof OffscreenCanvas==='function'?new OffscreenCanvas(1,1):document.createElement('canvas');
const captureContext=captureCanvas.getContext('2d',{willReadFrequently:true});
const tell=text=>{$('status').textContent=text;};
let worker,sequence=0,waiting=new Map(),loading,stream,objectURL;
let running=false,stop=false,videoLoaded=false,mode='video',lastPayload=null;
const maskCanvas=document.createElement('canvas');maskCanvas.width=64;maskCanvas.height=64;
const maskContext=maskCanvas.getContext('2d');
function rpc(type,data={},transfer=[]) {
  return new Promise((resolve,reject)=>{
    const id=++sequence;waiting.set(id,{resolve,reject});
    worker.postMessage({id,type,...data},transfer);
  });
}
async function load() {
  if(loading) return loading;
  loading=(async()=>{
    tell('ローカルモデルを準備しています…');
    worker=new Worker(new URL('./worker.mjs',import.meta.url),{type:'module'});
    worker.onmessage=({data})=>{
      if(data.type==='progress') {tell(data.message);return;}
      const pending=waiting.get(data.id);if(!pending) return;
      waiting.delete(data.id);data.error?pending.reject(Error(data.error)):pending.resolve(data.result);
    };
    worker.onerror=event=>{
      const error=Error(event.message||'Worker を起動できません。ページを再読み込みしてください。');
      for(const pending of waiting.values()) pending.reject(error);
      waiting.clear();worker.terminate();worker=null;loading=null;
    };
    await rpc('init');
  })();
  try {await loading;} catch(error) {worker?.terminate();worker=null;loading=null;throw error;}
}
function shutdownCamera() {
  stream?.getTracks().forEach(track=>track.stop());stream=null;video.srcObject=null;
}
function clearURL() {
  if(objectURL) URL.revokeObjectURL(objectURL);
  objectURL=null;
}
function controls() {
  for(const id of ['sample','file','camera','pick-file']) $(id).disabled=running;
  $('start').hidden=running||!videoLoaded||mode!=='video';
  $('stop').hidden=!running;$('stop').disabled=stop;
  $('drop').setAttribute('aria-busy',String(running));
}
function mediaReady() {
  if(video.readyState>=2) return Promise.resolve();
  return new Promise((resolve,reject)=>{
    const cleanup=()=>{clearTimeout(timer);video.removeEventListener('loadeddata',done);video.removeEventListener('error',failed);};
    const done=()=>{cleanup();resolve();};
    const failed=()=>{cleanup();reject(Error('動画を読み込めません。別の動画形式をお試しください。'));};
    const timer=setTimeout(failed,15000);
    video.addEventListener('loadeddata',done,{once:true});video.addEventListener('error',failed,{once:true});
  });
}
async function choose(source,isFile=false) {
  if(running) return;
  shutdownCamera();clearURL();mode='video';videoLoaded=false;
  if(isFile) {objectURL=URL.createObjectURL(source);video.src=objectURL;}
  else video.src=source;
  video.load();
  await start();
}
function report(error) {tell(`処理できませんでした: ${error.message}`);}
$('sample').onclick=()=>choose('samples/synthetic_gate_demo.mp4').catch(report);
$('pick-file').onclick=()=>$('file').click();
$('file').onchange=event=>{
  const file=event.target.files[0];if(file) choose(file,true).catch(report);
  event.target.value='';
};
$('camera').onclick=async()=>{
  if(running) return;
  running=true;stop=false;controls();
  try {
    if(!navigator.mediaDevices?.getUserMedia) throw Error('カメラには HTTPS または localhost が必要です。');
    shutdownCamera();clearURL();videoLoaded=false;mode='camera';
    const request={video:{facingMode:{ideal:'environment'},width:{ideal:1280},height:{ideal:720}},audio:false};
    try {stream=await navigator.mediaDevices.getUserMedia(request);}
    catch(error) {
      if(!['OverconstrainedError','ConstraintNotSatisfiedError','NotFoundError'].includes(error.name)) throw error;
      stream=await navigator.mediaDevices.getUserMedia({video:true,audio:false});
    }
    if(stop) {shutdownCamera();running=false;stop=false;controls();return;}
    video.removeAttribute('src');video.srcObject=stream;await video.play();
    running=false;await start();
  } catch(error) {shutdownCamera();running=false;controls();report(error);}
};
for(const name of ['dragover','dragleave','drop']) $('drop').addEventListener(name,event=>{
  event.preventDefault();$('drop').classList.toggle('drag',name==='dragover'&&!running);
  if(name==='drop'&&!running) {
    const file=event.dataTransfer.files[0];if(file) choose(file,true).catch(report);
  }
});
$('stop').onclick=()=>{
  stop=true;controls();tell('収集を終了しています。この後、一度だけ文字を読み出します…');
};
$('start').onclick=()=>start().catch(report);
window.addEventListener('pagehide',()=>{stop=true;shutdownCamera();clearURL();worker?.terminate();});

function capture(timestamp) {
  const scale=Math.min(1,1920/Math.max(video.videoWidth,video.videoHeight));
  const wOrig=Math.max(1,Math.round(video.videoWidth*scale)),hOrig=Math.max(1,Math.round(video.videoHeight*scale));
  if(captureCanvas.width!==wOrig||captureCanvas.height!==hOrig) {captureCanvas.width=wOrig;captureCanvas.height=hOrig;}
  captureContext.drawImage(video,0,0,wOrig,hOrig);
  const rgbaBuffer=captureContext.getImageData(0,0,wOrig,hOrig).data;
  return rpc('frame',{rgbaBuffer,wOrig,hOrig,timestamp},[rgbaBuffer.buffer]);
}
function paint(payload) {
  lastPayload=payload;
  const {wOrig:w,hOrig:h,vehicles,tracks}=payload;
  if(overlay.width!==w||overlay.height!==h) {overlay.width=w;overlay.height=h;}
  ctx.clearRect(0,0,w,h);
  const unit=Math.max(1,w/768),fontSize=14*unit;
  ctx.lineWidth=2*unit;ctx.font=`600 ${fontSize}px ui-monospace,monospace`;
  for(const vehicle of vehicles) {
    const [a,b,c,d]=vehicle.box;
    if(vehicle.mask) {
      const image=maskContext.createImageData(64,64);
      for(let i=0;i<4096;i++) if(vehicle.mask[i]>0) image.data.set([55,222,242,65],i*4);
      maskContext.putImageData(image,0,0);ctx.imageSmoothingEnabled=false;
      ctx.drawImage(maskCanvas,a,b,c-a,d-b);ctx.imageSmoothingEnabled=true;
    }
    ctx.strokeStyle='#41def2';ctx.strokeRect(a,b,c-a,d-b);
  }
  for(const track of tracks) {
    ctx.strokeStyle='#b6f773';ctx.fillStyle='#b6f773';ctx.beginPath();
    track.corners.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();ctx.stroke();
    for(const [x,y] of track.corners) {ctx.beginPath();ctx.arc(x,y,3*unit,0,Math.PI*2);ctx.fill();}
    const label=`#${track.id} COLLECTING (${track.frames} frames · 0 readouts)`;
    const tw=ctx.measureText(label).width,x=Math.max(0,Math.min(w-tw-12*unit,track.box[0]));
    const y=Math.max(fontSize+12*unit,track.box[1]-9*unit);
    ctx.fillStyle='#07131ee6';ctx.fillRect(x,y-fontSize-5*unit,tw+12*unit,fontSize+10*unit);
    ctx.fillStyle='#b6f773';ctx.fillText(label,x+6*unit,y);
  }
}
function element(tag,className,text) {
  const node=document.createElement(tag);if(className) node.className=className;
  if(text!==undefined) node.textContent=text;return node;
}
function resultsCards(results) {
  const host=$('results-cards');host.replaceChildren();
  const accepted=results.filter(r=>r.status==='ACCEPTED');
  for(const result of accepted) {
    const card=element('article','result-card');
    const header=element('div','card-top');
    header.append(element('span','accepted','ACCEPTED'),element('span','tier',result.confidence_tier));
    card.append(header,element('p','plate-text',result.text));
    const row=element('div','evidence-row');
    if(result.thumbnail) {
      const preview=element('canvas','plate-preview');preview.width=160;preview.height=80;
      preview.setAttribute('role','img');preview.setAttribute('aria-label','最も鮮明な観測の補正済みプレート画像');
      preview.getContext('2d').putImageData(new ImageData(result.thumbnail,160,80),0,0);row.append(preview);
    }
    const evidence=element('div','evidence');const pills=element('div','confidence-pills');
    for(const [key,label] of Object.entries({region:'地域',classification:'分類',hiragana:'かな',serial:'一連番号'})) {
      const probability=result.field_confidences?.[key]??0;
      pills.append(element('span','pill',`${label} ${Math.round(probability*100)}%`));
    }
    evidence.append(pills,element('p','frame-count',`#${result.track_id} · ${result.frames} frames · ${result.readouts} terminal readout`));
    row.append(evidence);card.append(row);host.append(card);
  }
  const abstained=results.filter(r=>r.status!=='ACCEPTED');
  if(abstained.length) {
    const counts=abstained.reduce((all,r)=>(all[r.status]=(all[r.status]??0)+1,all),{});
    host.append(element('p','abstained',`読み出しを保留: ${Object.entries(counts).map(([s,n])=>`${s} ${n} 件`).join(' · ')}`));
  }
  if(!results.length) host.append(element('p','muted','プレートを検出できませんでした。近くて鮮明な映像をお試しください。'));
  // JSON contains terminal decisions, not a bulky pixel array.
  $('results-json').textContent=JSON.stringify(results.map(({thumbnail,...result})=>result),null,2);
}
async function seek(time) {
  if(Math.abs(video.currentTime-time)<1e-5&&video.readyState>=2) return;
  await new Promise((resolve,reject)=>{
    const cleanup=()=>{video.removeEventListener('seeked',done);video.removeEventListener('error',failed);clearTimeout(timer);};
    const done=()=>{cleanup();resolve();},failed=()=>{cleanup();reject(Error('動画フレームを読み込めません。'));};
    const timer=setTimeout(failed,10000);video.addEventListener('seeked',done,{once:true});video.addEventListener('error',failed,{once:true});video.currentTime=time;
  });
}
async function start() {
  if(running) return;
  running=true;stop=false;controls();lastPayload=null;
  ctx.clearRect(0,0,overlay.width,overlay.height);
  $('results-cards').replaceChildren(element('p','muted','収集中の文字読み出しは 0 回です。「終了して読み出す」で結果を表示します。'));
  $('results-json').textContent='[]';$('raw-results').open=false;
  const isCamera=mode==='camera';let collecting=false;
  try {
    // Pause before asynchronous model loading so file playback never races seeking.
    if(!isCamera) {video.pause();video.controls=false;}
    await Promise.all([load(),mediaReady()]);
    videoLoaded=!isCamera;
    await rpc('reset');collecting=true;
    const t0=performance.now();let previous=-Infinity;
    const show=payload=>{
      paint(payload);
      tell(`COLLECTING · ${payload.frameIndex} フレーム · ${payload.tracks.length} トラック · 文字読み出し 0 回${isCamera?' · 終了ボタンで結果を表示':''}`);
    };
    if(isCamera) {
      while(!stop) {
        const timestamp=(performance.now()-t0)/1000;
        if(timestamp-previous<.04) {await new Promise(resolve=>setTimeout(resolve,10));continue;}
        previous=timestamp;show(await capture(timestamp));
      }
    } else {
      if(!Number.isFinite(video.duration)||video.duration<=0) throw Error('動画の長さを取得できません。');
      for(let t=0;t<video.duration-1e-5&&!stop;t+=1/12) {await seek(t);show(await capture(t));}
    }
    tell('SEALED · 終端の文字を一度だけ読み出しています…');
    const results=await rpc('finalize');collecting=false;resultsCards(results);
    ctx.clearRect(0,0,overlay.width,overlay.height);
    tell(`FINALIZED · ${lastPayload?.frameIndex??0} フレーム · ACCEPTED ${results.filter(r=>r.status==='ACCEPTED').length} 件 · 収集中の文字読み出し 0 回`);
  } catch(error) {
    // A processing failure closes remaining capabilities before another run.
    if(collecting&&worker) {try {await rpc('finalize');} catch {worker.terminate();worker=null;loading=null;}}
    report(error);
    $('results-cards').replaceChildren(element('p','muted','処理が中断されました。再実行するか、別の映像をお試しください。'));
  } finally {
    running=false;stop=false;video.controls=!isCamera;
    if(isCamera) shutdownCamera();
    controls();
  }
}
controls();
