// Apache-2.0 · vehicle-plate-ocr-jp-v1 v0.6.5.
// JP7029732B2 design-around: no character projection while COLLECTING.
import * as ort from './vendor/ort.wasm.min.mjs';
import { clamp, sigmoid, unpackGrammar, terminalDecision } from './terminal.mjs';

// The same pipeline is imported by the local Node parity check. Runtime and
// artifact loading are injectable; every image/track/terminal operation is shared.
export async function createPipeline({runtime=ort, readJSON, progress=()=>{}, autoSealFrames=0, autoSealMinSec=0, autoExitSec=0.55}={}) {
  const json=readJSON??(async path=>{
    const response=await fetch(new URL(path,import.meta.url));
    if(!response.ok) throw Error(`${path}: HTTP ${response.status}`);
    return response.json();
  });
  runtime.env.wasm.wasmPaths=new URL('./vendor/',import.meta.url).href;
  const hardware=(typeof navigator!=='undefined'&&navigator.hardwareConcurrency)||2;
  runtime.env.wasm.simd=true;
  runtime.env.wasm.numThreads=(typeof self!=='undefined'&&self.crossOriginIsolated&&typeof SharedArrayBuffer!=='undefined')
    ?Math.min(4,Math.max(2,Math.floor(hardware/2))):1;
  const float=(data,dims)=>new runtime.Tensor('float32',data instanceof Float32Array?data:new Float32Array(data),dims);
  const zero=dims=>float(new Float32Array(dims.reduce((a,b)=>a*b,1)),dims);
  const integer=value=>new runtime.Tensor('int64',BigInt64Array.of(BigInt(value)),[1]);
  const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
  const sessions={};
  const names=['temporal_det_nano','vehicle_cutout_nano','stage2','frame_encoder','memory_step','memory_readout'];
  let grammar,policy;
  try {
    async function createSessions() {
      for(const name of names) {
        progress(`INT8 グラフを準備中: ${name}`);
        sessions[name]=await runtime.InferenceSession.create(new URL(`models/${name}.int8.onnx`,import.meta.url).href,
          {executionProviders:['wasm'],graphOptimizationLevel:'all'});
      }
    }
    try {await createSessions();}
    catch(error) {
      if(runtime.env.wasm.numThreads<=1) throw error;
      await Promise.allSettled(Object.values(sessions).map(s=>s.release()));
      for(const name of names) delete sessions[name];
      runtime.env.wasm.numThreads=1;
      await createSessions();
    }
    grammar=await unpackGrammar(await json('models/grammar_tables.json'));
    policy=await json('models/gen6.json');
  } catch(error) {
    await Promise.all(Object.values(sessions).map(s=>s.release()));
    throw error;
  }
  const maxHeartbeatFrames=10, heartbeatInterval=maxHeartbeatFrames, plate_thresh=.22, min_is_plate_prob=.34;
  const plateLogitMin=Math.log(plate_thresh/(1-plate_thresh)),vehLogitMin=Math.log(.28/.72);
  const letterboxBuf=new Float32Array(3*768*448),foveaBuf=new Float32Array(3*96*160);
  const surroundBuf=new Float32Array(3*64*96),cropBuf=new Float32Array(3*80*160);
  const lumBuf=new Float32Array(12800),chromaBuf=new Float32Array(12800);
  const prevBorderBuf=new Float32Array(304);
  let emittedResults=[],nextId=1;
  let tracks=[],frameIndex=0,h3,h4,vehicles=[],frame,wOrig,hOrig;
  let lastFull=-Infinity,previousBorder,lastTime=-Infinity,phase='IDLE';
  let fullDetections=0,fastFrames=0;
  function configure(options={}) {
    if(options.autoSealFrames!==undefined) autoSealFrames=options.autoSealFrames;
    if(options.autoSealMinSec!==undefined) autoSealMinSec=options.autoSealMinSec;
    if(options.autoExitSec!==undefined) autoExitSec=options.autoExitSec;
  }
  function reset(options={}) {
    if(phase==='COLLECTING'||phase==='SEALED') throw Error('Finish the current collection first');
    configure(options);emittedResults=[];nextId=1;
    tracks=[];frameIndex=0;vehicles=[];lastFull=-Infinity;previousBorder=null;lastTime=-Infinity;
    fullDetections=0;fastFrames=0;
    h3=zero([1,36,56,96]);h4=zero([1,36,28,48]);phase='COLLECTING';
  }
function iou(a,b) {
  const intersection=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));
  return intersection/Math.max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection);
}
function nms(items,limit=16,threshold=.45) {
  const result=[];
  for(const item of items.sort((a,b)=>b.score-a.score)) {
    if(result.every(other=>iou(other.box,item.box)<threshold)) result.push(item);
    if(result.length>=limit) break;
  }
  return result;
}

  // All Stage-2/3 coordinates refer to the capped native sensor pixels.
  const neutral=[40/255,44/255,50/255];
  function patch(window,w,h,data) {
    const [x,y,rw,rh]=window;
    for(let j=0;j<h;j++) for(let i=0;i<w;i++) {
      const nx=x+(i+.5)*rw/w-.5,ny=y+(j+.5)*rh/h-.5,index=j*w+i,stride=w*h;
      if(nx<0||nx>wOrig-1||ny<0||ny>hOrig-1) {
        data[index]=neutral[0];data[index+stride]=neutral[1];data[index+2*stride]=neutral[2];
      } else {
        const x0=Math.floor(nx),y0=Math.floor(ny),x1=Math.min(wOrig-1,x0+1),y1=Math.min(hOrig-1,y0+1),dx=nx-x0,dy=ny-y0;
        const a=(y0*wOrig+x0)*4,b=(y0*wOrig+x1)*4,d=(y1*wOrig+x0)*4,e=(y1*wOrig+x1)*4;
        const wa=(1-dx)*(1-dy)/255,wb=dx*(1-dy)/255,wd=(1-dx)*dy/255,we=dx*dy/255;
        data[index]=frame[a]*wa+frame[b]*wb+frame[d]*wd+frame[e]*we;
        data[index+stride]=frame[a+1]*wa+frame[b+1]*wb+frame[d+1]*wd+frame[e+1]*we;
        data[index+2*stride]=frame[a+2]*wa+frame[b+2]*wb+frame[d+2]*wd+frame[e+2]*we;
      }
    }
    return float(data,[1,3,h,w]);
  }
  function letterbox() {
    const scale=Math.min(768/wOrig,448/hOrig),ox=(768-wOrig*scale)/2,oy=(448-hOrig*scale)/2;
    const data=letterboxBuf;
    if(wOrig===768&&hOrig===448) {
      const stride=768*448;
      for(let i=0;i<stride;i++) {const k=i*4;data[i]=frame[k]/255;data[i+stride]=frame[k+1]/255;data[i+2*stride]=frame[k+2]/255;}
      return {images:float(data,[1,3,448,768]),scale,ox,oy};
    }
    data.fill(114/255);
    for(let y=0;y<448;y++) for(let x=0;x<768;x++) {
      if(x+.5<ox||x+.5>=ox+wOrig*scale||y+.5<oy||y+.5>=oy+hOrig*scale) continue;
      const nx=clamp((x+.5-ox)/scale-.5,0,wOrig-1),ny=clamp((y+.5-oy)/scale-.5,0,hOrig-1);
      const index=y*768+x,stride=768*448;
      if(nx<0||nx>wOrig-1||ny<0||ny>hOrig-1) {
        data[index]=neutral[0];data[index+stride]=neutral[1];data[index+2*stride]=neutral[2];
      } else {
        const x0=Math.floor(nx),y0=Math.floor(ny),x1=Math.min(wOrig-1,x0+1),y1=Math.min(hOrig-1,y0+1),dx=nx-x0,dy=ny-y0;
        const a=(y0*wOrig+x0)*4,b=(y0*wOrig+x1)*4,d=(y1*wOrig+x0)*4,e=(y1*wOrig+x1)*4;
        const wa=(1-dx)*(1-dy)/255,wb=dx*(1-dy)/255,wd=(1-dx)*dy/255,we=dx*dy/255;
        data[index]=frame[a]*wa+frame[b]*wb+frame[d]*wd+frame[e]*we;
        data[index+stride]=frame[a+1]*wa+frame[b+1]*wb+frame[d+1]*wd+frame[e+1]*we;
        data[index+2*stride]=frame[a+2]*wa+frame[b+2]*wb+frame[d+2]*wd+frame[e+2]*we;
      }
    }
    return {images:float(data,[1,3,448,768]),scale,ox,oy};
  }
  function windows(box) {
    const [a,b,c,d]=box,cx=(a+c)/2,cy=(b+d)/2;
    const rw=Math.max(Math.max(5,c-a)*1.7,Math.max(3,d-b)*1.7*160/96,18),rh=rw*96/160;
    const sw=rw*4.5/1.7,sh=sw*64/96;
    return [[cx-rw/2,cy-rh/2,rw,rh],[cx-sw/2,cy-sh/2,sw,sh]];
  }
function projective(corners) {
  // Solve homography from unit square to the predicted sensor quadrilateral.
  const matrix=[],rhs=[];
  for(let i=0;i<4;i++) {
    const u=[0,1,1,0][i],v=[0,0,1,1][i],[x,y]=corners[i];
    matrix.push([u,v,1,0,0,0,-u*x,-v*x]);rhs.push(x);
    matrix.push([0,0,0,u,v,1,-u*y,-v*y]);rhs.push(y);
  }
  for(let k=0;k<8;k++) {
    let p=k;for(let i=k+1;i<8;i++) if(Math.abs(matrix[i][k])>Math.abs(matrix[p][k])) p=i;
    [matrix[k],matrix[p]]=[matrix[p],matrix[k]];[rhs[k],rhs[p]]=[rhs[p],rhs[k]];
    const d=matrix[k][k];if(Math.abs(d)<1e-10) throw Error('Degenerate plate homography');
    for(let j=k;j<8;j++) matrix[k][j]/=d;rhs[k]/=d;
    for(let i=0;i<8;i++) if(i!==k) {const f=matrix[i][k];for(let j=k;j<8;j++) matrix[i][j]-=f*matrix[k][j];rhs[i]-=f*rhs[k];}
  }
  const data=cropBuf;
  for(let y=0;y<80;y++) for(let x=0;x<160;x++) {
    const u=((x+.5)/160-.05)/.9,v=((y+.5)/80-.05)/.9;
    const z=rhs[6]*u+rhs[7]*v+1,xx=(rhs[0]*u+rhs[1]*v+rhs[2])/z,yy=(rhs[3]*u+rhs[4]*v+rhs[5])/z;
    const nx=xx,ny=yy,index=y*160+x,stride=12800;
    if(nx<0||nx>wOrig-1||ny<0||ny>hOrig-1) {
        data[index]=neutral[0];data[index+stride]=neutral[1];data[index+2*stride]=neutral[2];
      } else {
        const x0=Math.floor(nx),y0=Math.floor(ny),x1=Math.min(wOrig-1,x0+1),y1=Math.min(hOrig-1,y0+1),dx=nx-x0,dy=ny-y0;
        const a=(y0*wOrig+x0)*4,b=(y0*wOrig+x1)*4,d=(y1*wOrig+x0)*4,e=(y1*wOrig+x1)*4;
        const wa=(1-dx)*(1-dy)/255,wb=dx*(1-dy)/255,wd=(1-dx)*dy/255,we=dx*dy/255;
        data[index]=frame[a]*wa+frame[b]*wb+frame[d]*wd+frame[e]*we;
        data[index+stride]=frame[a+1]*wa+frame[b+1]*wb+frame[d+1]*wd+frame[e+1]*we;
        data[index+2*stride]=frame[a+2]*wa+frame[b+2]*wb+frame[d+2]*wd+frame[e+2]*we;
      }
  }
  return float(data,[1,3,80,160]);
}
function geometry(q,crop,previous) {
  const norm=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
  const wt=norm(q[0],q[1]),wr=norm(q[1],q[2]),wb=norm(q[2],q[3]),wl=norm(q[3],q[0]);
  const w=Math.max(2,(wt+wb)/2),h=Math.max(1,(wl+wr)/2),pixels=12800,lum=lumBuf,chroma=chromaBuf;
  let brightness=0,highlights=0,lapSum=0,lapSquare=0;
  for(let i=0;i<pixels;i++) {
    const r=crop.data[i],g=crop.data[pixels+i],b=crop.data[2*pixels+i],v=.299*r+.587*g+.114*b;
    lum[i]=v;brightness+=v;highlights+=Number(v>.96);chroma[i]=Math.max(r,g,b)-Math.min(r,g,b);
  }
  for(let y=0;y<80;y++) for(let x=0;x<160;x++) {
    const ym=(y===0?1:y-1)*160,yp=(y===79?78:y+1)*160,xm=x===0?1:x-1,xp=x===159?158:x+1;
    const v=2*(lum[ym+xm]+lum[ym+xp]+lum[yp+xm]+lum[yp+xp])-8*lum[y*160+x];
    lapSum+=v;lapSquare+=v*v;
  }
  const variance=Math.max(0,lapSquare/pixels-(lapSum/pixels)**2);
  chroma.sort();let chromaSum=0;for(let i=pixels-1920;i<pixels;i++) chromaSum+=chroma[i];
  let displacement=0;
  if(previous) for(let i=0;i<4;i++) for(let k=0;k<2;k++) displacement+=(q[i][k]-previous[i][k])**2;
  const delta=previous?clamp(Math.sqrt(displacement)/(2*w)):0;
  return float([clamp(.5*Math.log2(w/64),-1,1),clamp(.5*Math.log2(h/32),-1,1),clamp(w/(2*h),.25,2)-1,Math.tanh(2.5*Math.log((wl+.001)/(wr+.001))),Math.tanh(2.5*Math.log((wb+.001)/(wt+.001))),delta,clamp(.25*Math.log1p(100*variance)),brightness/pixels,chromaSum/1920,highlights/pixels],[1,10]);
}

  function thumbnail(crop) {
    const rgba=new Uint8ClampedArray(160*80*4);
    for(let i=0;i<12800;i++) {
      for(let c=0;c<3;c++) rgba[4*i+c]=Math.round(clamp(crop.data[c*12800+i])*255);
      rgba[4*i+3]=255;
    }
    return rgba;
  }
  function newTrack(box,time) {
    if(tracks.length>=256) throw Error('トラック上限です。終了して読み出してください。');
    const memory={mem_h:zero([1,12,112]),mem_P:zero([1,12,112]),mem_conflict:zero([1,12]),color_h:zero([1,32]),color_P:zero([1,32]),pol_h:zero([1]),pol_P:zero([1]),pose_ema:zero([1,4135]),last_timestamp_s:float([-1],[1])};
    const track={id:nextId++,box,firstTime:time,last:time,lastFrame:frameIndex,state:'COLLECTING',memory,
      frames:0,readouts:0,odds:0,sharpness:-Infinity,velocity:[0,0],reacquire:false};
    tracks.push(track);progress({track_id:track.id,state:track.state,readouts:0});return track;
  }
  function borderQuiet() {
    let count=0,difference=0;
    for(let y=0;y<20;y++) for(let x=0;x<32;x++) {
      if(x>3&&x<28&&y>2&&y<17) continue;
      const xx=Math.min(wOrig-1,Math.floor((x+.5)*wOrig/32)),yy=Math.min(hOrig-1,Math.floor((y+.5)*hOrig/20));
      const i=(yy*wOrig+xx)*4,value=(.299*frame[i]+.587*frame[i+1]+.114*frame[i+2])/255;
      difference+=Math.abs(value-prevBorderBuf[count]);prevBorderBuf[count++]=value;
    }
    const quiet=previousBorder!==null&&previousBorder!==undefined&&difference/count<.065;
    previousBorder=prevBorderBuf;return quiet;
  }

  function predictedBox(track,time) {
    const dt=Math.min(.5,time-track.last),[vx,vy]=track.velocity;
    return track.box.map((v,i)=>v+(i%2?vy:vx)*dt);
  }
  async function process({rgbaBuffer,wOrig:width,hOrig:height,timestamp:time}) {
    if(phase!=='COLLECTING') throw Error('Collection is closed');
    if(!Number.isInteger(width)||!Number.isInteger(height)||Math.min(width,height)<=0||Math.max(width,height)>1920)
      throw Error('Invalid native frame dimensions');
    frame=rgbaBuffer instanceof Uint8ClampedArray?rgbaBuffer:new Uint8ClampedArray(rgbaBuffer);
    if(frame.length!==width*height*4||!Number.isFinite(time)||time<=lastTime) throw Error('Invalid frame or non-monotonic timestamp');
    wOrig=width;hOrig=height;lastTime=time;
    const started=performance.now();
    tracks=tracks.filter(t=>t.state!=='FINALIZED'||time-t.last<1.5);
    const quiet=borderQuiet(),active=tracks.filter(t=>t.state!=='SEALED'&&frameIndex-t.lastFrame<heartbeatInterval&&time-t.last<1.5);
    const full=frameIndex-lastFull>=heartbeatInterval||!active.length||!quiet||active.every(t=>t.reacquire);
    const proposals=[];
    if(full) {
      const {images,scale,ox,oy}=letterbox();
      const native=box=>box.map((v,i)=>(v-(i%2?oy:ox))/scale);
      const output=await sessions.temporal_det_nano.run({images,h_p3_in:h3,h_p4_in:h4});
      h3=output.h_p3_out;h4=output.h_p4_out;lastFull=frameIndex;fullDetections++;
      const cls=output.cls_logits.data,boxes=output.pred_boxes.data;
      for(let i=0;i<cls.length/3;i++) {
        if(cls[i*3]>=plateLogitMin&&cls[i*3]>cls[i*3+1]&&cls[i*3]>cls[i*3+2]) {
          const box=native(Array.from(boxes.slice(i*4,i*4+4)));
          if(box[2]-box[0]>3&&box[3]-box[1]>2) proposals.push({box,score:sigmoid(cls[i*3]),corners:output.pred_corners?Array.from({length:4},(_,k)=>[(output.pred_corners.data[i*8+k*2]-ox)/scale,(output.pred_corners.data[i*8+k*2+1]-oy)/scale]):null});
        }
      }
      const candidates=[];
      for(let q=0;q<output.veh_log_odds.data.length;q++) {
        const logit=output.veh_log_odds.data[q];
        if(logit>=vehLogitMin) candidates.push({box:Array.from(output.raw_q_boxes_px.data.slice(q*4,q*4+4)),score:sigmoid(logit),q});
      }
      vehicles=nms(candidates,12,.52);
      if(vehicles.length) {
        const cutout=await sessions.vehicle_cutout_nano.run({p2_feat:output.p2_feat,aux_edge_3ch:output.aux_edge_3ch,
          veh_rois:float(vehicles.flatMap(v=>[0,...v.box]),[vehicles.length,5]),
          query_condition:float(vehicles.flatMap(v=>Array.from(output.query_condition.data.slice(v.q*64,(v.q+1)*64))),[vehicles.length,64])});
        vehicles.forEach((v,i)=>{v.mask=cutout.cutout_logits.data.slice(i*4096,(i+1)*4096);});
      }
      vehicles.forEach(v=>{v.box=native(v.box);});
    } else {
      fastFrames++;
      for(const track of active.filter(t=>!t.reacquire)) proposals.push({box:predictedBox(track,time),score:1,track});
    }
    if(full) for(const track of active) {
      if(track.state!=='FINALIZED'||frameIndex-track.lastFrame>1) continue;
      const box=predictedBox(track,time),cx=(box[0]+box[2])/2,cy=(box[1]+box[3])/2;
      // A vehicle-supported continuation covers occasional coarse plate misses.
      if(vehicles.some(v=>cx>=v.box[0]&&cx<=v.box[2]&&cy>=v.box[1]&&cy<=v.box[3]))
        proposals.push({box,score:.01,track});
    }
    const used=new Set();
    // Vehicle containment suppresses signboard/text proposals before expensive
    // localization. Track-predicted boxes remain eligible between full frames.
    const contained=proposals.filter(({box,track})=>track||!vehicles.length||vehicles.some(v=>{
      const cx=(box[0]+box[2])/2,cy=(box[1]+box[3])/2;
      return cx>=v.box[0]&&cx<=v.box[2]&&cy>=v.box[1]&&cy<=v.box[3];
    }));
    // Reuse terminal tracks before per-vehicle pruning: a covered lower-score
    // proposal must never become a fresh collector on the same vehicle.
    const remaining=[];
    for(const candidate of nms(contained,16)) {
      const cq=candidate.corners;
      const cornerBox=validCorners(cq)?[Math.min(...cq.map(p=>p[0])),Math.min(...cq.map(p=>p[1])),Math.max(...cq.map(p=>p[0])),Math.max(...cq.map(p=>p[1]))]:null;
      if(cornerBox&&tracks.some(t=>t.state==='FINALIZED'&&iou(predictedBox(t,time),cornerBox)>=.18)) candidate.box=cornerBox;
      const covered=tracks.filter(t=>t.state==='FINALIZED'&&iou(predictedBox(t,time),candidate.box)>=.18)
        .sort((a,b)=>iou(predictedBox(b,time),candidate.box)-iou(predictedBox(a,time),candidate.box))[0];
      if(!covered) {remaining.push(candidate);continue;}
      if(used.has(covered.id)) continue;
      const box=candidate.box,dx=(box[0]+box[2]-covered.box[0]-covered.box[2])/2,dy=(box[1]+box[3]-covered.box[1]-covered.box[3])/2;
      covered.corners=validCorners(candidate.corners)?candidate.corners:covered.corners.map(([x,y])=>[x+dx,y+dy]);
      covered.box=box;covered.last=time;covered.lastFrame=frameIndex;covered.reacquire=false;used.add(covered.id);
    }
    const ownerOf=box=>{
      const cx=(box[0]+box[2])/2,cy=(box[1]+box[3])/2;
      return vehicles.filter(v=>cx>=v.box[0]&&cx<=v.box[2]&&cy>=v.box[1]&&cy<=v.box[3])
        .sort((a,b)=>(a.box[2]-a.box[0])*(a.box[3]-a.box[1])-(b.box[2]-b.box[0])*(b.box[3]-b.box[1]))[0];
    };
    const owners=new Set(tracks.filter(t=>used.has(t.id)).map(t=>ownerOf(t.box)).filter(Boolean));
    const selected=remaining.filter(candidate=>{
      if(tracks.some(t=>used.has(t.id)&&iou(t.box,candidate.box)>=.18)) return false;
      const owner=ownerOf(candidate.box);
      if(owner&&owners.has(owner)) return false;
      if(owner) owners.add(owner);
      return true;
    }).slice(0,4);
    for(const candidate of selected) {
      if(tracks.some(t=>used.has(t.id)&&iou(t.box,candidate.box)>=.18)) continue;
      const [fw,sw]=windows(candidate.box),prior=candidate.track;
      const reuseGeometry=!full&&prior?.state==='COLLECTING'&&prior.localization&&frameIndex-prior.localizedFrame<3;
      const localized=reuseGeometry?prior.localization:await sessions.stage2.run({fovea_rois:patch(fw,160,96,foveaBuf),surround_rois:patch(sw,96,64,surroundBuf),sampling_policy:integer(prior?2:0)});
      const llr=localized.plate_null_llr.data[0],support=sigmoid(llr);
      if(!localized.geometry_valid.data[0]||support<min_is_plate_prob) {
        if(prior) prior.reacquire=true;
        continue;
      }
      // Quiet fast frames can propagate recently validated geometry, but their
      // crop, encoder and memory update always consume the current sensor pixels.
      const dx=prior?(candidate.box[0]+candidate.box[2]-prior.box[0]-prior.box[2])/2:0;
      const dy=prior?(candidate.box[1]+candidate.box[3]-prior.box[1]-prior.box[3])/2:0;
      const corners=reuseGeometry?prior.corners.map(([x,y])=>[x+dx,y+dy]):Array.from({length:4},(_,i)=>[localized.corners_norm.data[i*2]*fw[2]+fw[0],localized.corners_norm.data[i*2+1]*fw[3]+fw[1]]);
      if(!corners.flat().every(Number.isFinite)||corners.some(([x,y])=>x<0||x>wOrig-1||y<0||y>hOrig-1)) continue;
      let convex=true;
      for(let i=0;i<4;i++) {const a=corners[i],b=corners[(i+1)%4],c=corners[(i+2)%4];if((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])<=0) convex=false;}
      if(!convex) continue;
      const box=[Math.min(...corners.map(p=>p[0])),Math.min(...corners.map(p=>p[1])),Math.max(...corners.map(p=>p[0])),Math.max(...corners.map(p=>p[1]))];
      let track=candidate.track??tracks.filter(t=>t.state==='COLLECTING'&&!used.has(t.id)&&time-t.last<1.5&&iou(predictedBox(t,time),box)>.25)
        .sort((a,b)=>iou(predictedBox(b,time),box)-iou(predictedBox(a,time),box))[0];
      track??=newTrack(box,time);used.add(track.id);
      if(!reuseGeometry) {track.localization=localized;track.localizedFrame=frameIndex;}
      if(track.state!=='COLLECTING'||track.readouts!==0) throw Error('Sealed collector cannot receive evidence');
      const crop=projective(corners),geom=geometry(corners,crop,track.corners);
      const encoded=await sessions.frame_encoder.run({crops:crop,plate_sides:integer(0),geom_meta:geom});
      const visibility=mean(Array.from(localized.corner_vis_logits.data,sigmoid));
      const nc=Math.max(0,localized.support_nonconformity.data[0]);
      const damping=Math.exp(-Math.min(30,nc+Math.max(0,-llr)))*(.05+.95*visibility);
      for(const key of ['lambda_prec','lambda_color','pol_prec']) encoded[key]=float(Array.from(encoded[key].data,x=>x*damping),encoded[key].dims);
      const next=await sessions.memory_step.run({...track.memory,...encoded,geom_meta:geom,timestamp_s:float([time],[1])});
      for(const name of Object.keys(track.memory)) track.memory[name]=next[`new_${name}`];
      const qc=clamp(support,1e-6,1-1e-6),increment=Math.log(qc)-Math.log1p(-qc);
      if(!track.frames) {track.firstSupport=support;track.odds=increment;}
      else {
        const dt=Math.max(.04,time-track.last);
        track.odds=clamp(track.odds+Math.min(1,Math.max(.05,dt/.5))*(.05+.95*visibility)*increment,-30,30);
        track.velocity=[0,1].map(i=>.5*track.velocity[i]+.5*((box[i]+box[i+2]-track.box[i]-track.box[i+2])/2)/dt);
      }
      // Pixel thumbnail only. No character projection, logits or grammar here.
      if(geom.data[6]>track.sharpness) {track.sharpness=geom.data[6];track.thumbnail=thumbnail(crop);}
      track.frames++;track.last=time;track.lastFrame=frameIndex;track.box=box;track.corners=corners;track.reacquire=false;
    }
    for(const t of active) if(!used.has(t.id)) t.reacquire=true;
    // Zero auto-seal frames selects the strict EOF-only batch contract.
    const newlyFinalized=[];
    for(const t of tracks) if(t.state==='COLLECTING'&&autoSealFrames>0&&((autoExitSec>0&&time-t.last>=autoExitSec)||
      (autoSealFrames>0&&t.frames>=autoSealFrames&&time-t.firstTime>=autoSealMinSec&&mean(t.memory.mem_P.data)>=.35)))
      newlyFinalized.push(await sealTrack(t));
    frameIndex++;
    return {timestamp:time,wOrig,hOrig,frameIndex,fullDetections,fastFrames,latencyMs:performance.now()-started,newlyFinalized,
      finalizedTracks:tracks.filter(t=>t.state==='FINALIZED'&&used.has(t.id)).map(t=>({id:t.id,box:t.box,corners:t.corners,status:t.result.status,text:t.result.text,confidence_tier:t.result.confidence_tier,frames:t.result.frames,readouts:t.result.readouts})),proposalCount:contained.length,localizedCount:selected.length,
      vehicles:vehicles.map(({box,mask})=>({box,mask})),
      tracks:tracks.filter(t=>t.state==='COLLECTING'&&used.has(t.id)).map(t=>({id:t.id,box:t.box,corners:t.corners,state:t.state,frames:t.frames,readouts:t.readouts}))};
  }
  function validCorners(q) {
    if(!q||q.length!==4||!q.flat().every(Number.isFinite)) return false;
    for(let i=0;i<4;i++) {
      const a=q[i],b=q[(i+1)%4],c=q[(i+2)%4];
      if(a[0]<0||a[0]>=wOrig||a[1]<0||a[1]>=hOrig||(b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])<=0) return false;
    }
    return true;
  }
  async function sealTrack(track) {
    if(track.state!=='COLLECTING'||track.readouts!==0) throw Error('Terminal capability already consumed');
    track.state='SEALED';track.readouts=1; // Consume before any asynchronous projection.
    progress({track_id:track.id,state:track.state,readouts:track.readouts});
    let decision;
    try {
      const ceiling=policy.terminal.readout_precision_ceiling,precision=mean(track.memory.mem_P.data);
      if(ceiling>0&&precision>ceiling) for(const name of ['mem_h','mem_P']) {
        const data=track.memory[name].data;
        for(let i=0;i<data.length;i++) data[i]*=ceiling/precision;
      }
      const feed=Object.fromEntries(['mem_h','mem_P','mem_conflict','color_h','color_P','pol_h','pol_P'].map(k=>[k,track.memory[k]]));
      const readout=await sessions.memory_readout.run(feed);
      decision=terminalDecision(readout,track.memory,track.frames,track.frames===1?track.firstSupport:sigmoid(track.odds),grammar,policy);
    } catch(error) {decision={status:'UNREADABLE',confidence_tier:'NONE',text:null,error:error.message};}
    track.state='FINALIZED';
    track.result={track_id:track.id,...decision,frames:track.frames,readouts:1,state:'FINALIZED',thumbnail:track.thumbnail};
    emittedResults.push(track.result);progress({track_id:track.id,state:track.state,readouts:track.readouts});return track.result;
  }
  async function resegment(options={}) {
    if(phase==='IDLE'||phase==='SEALED') throw Error('Collection is unavailable');
    const newlySealed=[];
    for(const t of tracks) if(t.state==='COLLECTING'&&t.frames>=2) newlySealed.push(await sealTrack(t));
    tracks=[];vehicles=[];h3=zero([1,36,56,96]);h4=zero([1,36,28,48]);
    lastFull=-Infinity;previousBorder=null;lastTime=-Infinity;
    configure(options);phase='COLLECTING';
    return {finalizedBeforeSeek:newlySealed};
  }
  async function finalize() {
    if(phase!=='COLLECTING') throw Error('Terminal capability already consumed');
    phase='SEALED';
    for(const t of tracks) if(t.state==='COLLECTING') await sealTrack(t);
    phase='FINALIZED';return emittedResults.slice();
  }
  async function dispose() {await Promise.all(Object.values(sessions).map(s=>s.release()));}
  return {reset,process,finalize,resegment,dispose};
}

// RPC commands serialize frames, timeline changes and terminal operations.
if(typeof self!=='undefined'&&typeof self.postMessage==='function'&&typeof document==='undefined') {
  let pipeline,queue=Promise.resolve();
  self.onmessage=({data})=>{
    queue=queue.then(async()=>{
      try {
        let result;
        if(data.type==='init') {pipeline??=await createPipeline({...data.options,progress:message=>self.postMessage({type:'progress',message})});result=true;}
        else if(data.type==='reset') {pipeline.reset(data.options);result=true;}
        else if(data.type==='frame') result=await pipeline.process(data);
        else if(data.type==='resegment') result=await pipeline.resegment(data.options);
        else if(data.type==='finalize') result=await pipeline.finalize();
        else throw Error('Unknown worker command');
        // Keep stored thumbnails intact for later EOF/resegment snapshots.
        self.postMessage({id:data.id,type:data.type,result});
      } catch(error) {self.postMessage({id:data.id,error:error.message});}
    });
  };
}
