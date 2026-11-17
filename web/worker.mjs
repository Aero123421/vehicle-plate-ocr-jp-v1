// Apache-2.0 · vehicle-plate-ocr-jp-v1 v0.6.5.
// JP7029732B2 design-around: no character projection while COLLECTING.
import * as ort from './vendor/ort.wasm.min.mjs';
import { clamp, sigmoid, unpackGrammar, terminalDecision } from './terminal.mjs';

// The same pipeline is imported by the local Node parity check. Runtime and
// artifact loading are injectable; every image/track/terminal operation is shared.
export async function createPipeline({runtime=ort, readJSON, progress=()=>{}}={}) {
  const json=readJSON??(async path=>{
    const response=await fetch(new URL(path,import.meta.url));
    if(!response.ok) throw Error(`${path}: HTTP ${response.status}`);
    return response.json();
  });
  runtime.env.wasm.wasmPaths=new URL('./vendor/',import.meta.url).href;
  runtime.env.wasm.numThreads=1;
  const float=(data,dims)=>new runtime.Tensor('float32',data instanceof Float32Array?data:new Float32Array(data),dims);
  const zero=dims=>float(new Float32Array(dims.reduce((a,b)=>a*b,1)),dims);
  const integer=value=>new runtime.Tensor('int64',BigInt64Array.of(BigInt(value)),[1]);
  const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
  const sessions={};
  const names=['temporal_det_nano','vehicle_cutout_nano','stage2','frame_encoder','memory_step','memory_readout'];
  let grammar,policy;
  try {
    for(const name of names) {
      progress(`INT8 グラフを準備中: ${name}`);
      sessions[name]=await runtime.InferenceSession.create(new URL(`models/${name}.int8.onnx`,import.meta.url).href,
        {executionProviders:['wasm'],graphOptimizationLevel:'all'});
    }
    grammar=await unpackGrammar(await json('models/grammar_tables.json'));
    policy=await json('models/gen6.json');
  } catch(error) {
    await Promise.all(Object.values(sessions).map(s=>s.release()));
    throw error;
  }
  const heartbeatInterval=5, plate_thresh=.22, min_is_plate_prob=.34;
  let tracks=[],frameIndex=0,h3,h4,vehicles=[],frame,wOrig,hOrig;
  let lastFull=-Infinity,previousBorder,lastTime=-Infinity,phase='IDLE';
  let fullDetections=0,fastFrames=0;
  function reset() {
    if(phase==='COLLECTING'||phase==='SEALED') throw Error('Finish the current collection first');
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
  function sample(x,y,c) {
    if(x<0||x>wOrig-1||y<0||y>hOrig-1) return neutral[c];
    const x0=Math.floor(x),y0=Math.floor(y),x1=Math.min(wOrig-1,x0+1),y1=Math.min(hOrig-1,y0+1),dx=x-x0,dy=y-y0;
    const a=(y0*wOrig+x0)*4+c,b=(y0*wOrig+x1)*4+c,d=(y1*wOrig+x0)*4+c,e=(y1*wOrig+x1)*4+c;
    return ((frame[a]*(1-dx)+frame[b]*dx)*(1-dy)+(frame[d]*(1-dx)+frame[e]*dx)*dy)/255;
  }
  // Compute the bilinear weights once for RGB, without per-pixel closures.
  function sampleRGB(data,index,stride,x,y) {
    if(x<0||x>wOrig-1||y<0||y>hOrig-1) {
      data[index]=neutral[0];data[index+stride]=neutral[1];data[index+2*stride]=neutral[2];return;
    }
    const x0=Math.floor(x),y0=Math.floor(y),x1=Math.min(wOrig-1,x0+1),y1=Math.min(hOrig-1,y0+1),dx=x-x0,dy=y-y0;
    const a=(y0*wOrig+x0)*4,b=(y0*wOrig+x1)*4,d=(y1*wOrig+x0)*4,e=(y1*wOrig+x1)*4;
    const wa=(1-dx)*(1-dy)/255,wb=dx*(1-dy)/255,wd=(1-dx)*dy/255,we=dx*dy/255;
    for(let c=0;c<3;c++) data[index+c*stride]=frame[a+c]*wa+frame[b+c]*wb+frame[d+c]*wd+frame[e+c]*we;
  }
  function patch(window,w,h) {
    const [x,y,rw,rh]=window,data=new Float32Array(3*w*h);
    for(let j=0;j<h;j++) for(let i=0;i<w;i++)
      sampleRGB(data,j*w+i,w*h,x+(i+.5)*rw/w-.5,y+(j+.5)*rh/h-.5);
    return float(data,[1,3,h,w]);
  }
  function letterbox() {
    const scale=Math.min(768/wOrig,448/hOrig),ox=(768-wOrig*scale)/2,oy=(448-hOrig*scale)/2;
    const data=new Float32Array(3*768*448).fill(114/255);
    for(let y=0;y<448;y++) for(let x=0;x<768;x++) {
      if(x+.5<ox||x+.5>=ox+wOrig*scale||y+.5<oy||y+.5>=oy+hOrig*scale) continue;
      const nx=clamp((x+.5-ox)/scale-.5,0,wOrig-1),ny=clamp((y+.5-oy)/scale-.5,0,hOrig-1);
      sampleRGB(data,y*768+x,768*448,nx,ny);
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
  const data=new Float32Array(3*80*160);
  for(let y=0;y<80;y++) for(let x=0;x<160;x++) {
    const u=((x+.5)/160-.05)/.9,v=((y+.5)/80-.05)/.9;
    const z=rhs[6]*u+rhs[7]*v+1,xx=(rhs[0]*u+rhs[1]*v+rhs[2])/z,yy=(rhs[3]*u+rhs[4]*v+rhs[5])/z;
    sampleRGB(data,y*160+x,12800,xx,yy);
  }
  return float(data,[1,3,80,160]);
}
function geometry(q,crop,previous) {
  const norm=(a,b)=>Math.hypot(a[0]-b[0],a[1]-b[1]);
  const wt=norm(q[0],q[1]),wr=norm(q[1],q[2]),wb=norm(q[2],q[3]),wl=norm(q[3],q[0]);
  const w=Math.max(2,(wt+wb)/2),h=Math.max(1,(wl+wr)/2),pixels=12800,lum=new Float64Array(pixels),chroma=[];
  for(let i=0;i<pixels;i++) {const r=crop.data[i],g=crop.data[pixels+i],b=crop.data[2*pixels+i];lum[i]=.299*r+.587*g+.114*b;chroma.push(Math.max(r,g,b)-Math.min(r,g,b));}
  const lap=[];
  const reflect=(v,n)=>v<0?-v:v>=n?2*n-2-v:v;
  for(let y=0;y<80;y++) for(let x=0;x<160;x++) {
    const at=(dx,dy)=>lum[reflect(y+dy,80)*160+reflect(x+dx,160)];
    lap.push(2*(at(-1,-1)+at(1,-1)+at(-1,1)+at(1,1))-8*lum[y*160+x]);
  }
  const lm=mean(lap),variance=mean(lap.map(x=>(x-lm)**2));chroma.sort((a,b)=>b-a);
  const delta=previous?clamp(Math.sqrt(q.flatMap((p,i)=>p.map((v,k)=>(v-previous[i][k])**2)).reduce((a,b)=>a+b,0))/(2*w)):0;
  return float([clamp(.5*Math.log2(w/64),-1,1),clamp(.5*Math.log2(h/32),-1,1),clamp(w/(2*h),.25,2)-1,Math.tanh(2.5*Math.log((wl+.001)/(wr+.001))),Math.tanh(2.5*Math.log((wb+.001)/(wt+.001))),delta,clamp(.25*Math.log1p(100*variance)),mean(lum),mean(chroma.slice(0,1920)),mean(Array.from(lum,v=>v>.96?1:0))],[1,10]);
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
    const track={id:tracks.length+1,box,last:time,lastFrame:frameIndex,state:'COLLECTING',memory,
      frames:0,readouts:0,odds:0,sharpness:-Infinity,velocity:[0,0],reacquire:false};
    tracks.push(track);return track;
  }
  function borderQuiet() {
    const border=[];
    for(let y=0;y<20;y++) for(let x=0;x<32;x++) {
      if(x>3&&x<28&&y>2&&y<17) continue;
      const xx=Math.min(wOrig-1,Math.floor((x+.5)*wOrig/32)),yy=Math.min(hOrig-1,Math.floor((y+.5)*hOrig/20));
      border.push(.299*sample(xx,yy,0)+.587*sample(xx,yy,1)+.114*sample(xx,yy,2));
    }
    const quiet=previousBorder&&mean(border.map((v,i)=>Math.abs(v-previousBorder[i])))<.065;
    previousBorder=border;return Boolean(quiet);
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
    const quiet=borderQuiet(),active=tracks.filter(t=>t.state==='COLLECTING'&&frameIndex-t.lastFrame<heartbeatInterval);
    const full=frameIndex-lastFull>=heartbeatInterval||!active.length||!quiet||active.every(t=>t.reacquire);
    const proposals=[];
    if(full) {
      const {images,scale,ox,oy}=letterbox();
      const native=box=>box.map((v,i)=>(v-(i%2?oy:ox))/scale);
      const output=await sessions.temporal_det_nano.run({images,h_p3_in:h3,h_p4_in:h4});
      h3=output.h_p3_out;h4=output.h_p4_out;lastFull=frameIndex;fullDetections++;
      const cls=output.cls_logits.data,boxes=output.pred_boxes.data;
      for(let i=0;i<cls.length/3;i++) {
        const score=sigmoid(cls[i*3]);
        if(score>=plate_thresh&&cls[i*3]>cls[i*3+1]&&cls[i*3]>cls[i*3+2]) {
          const box=native(Array.from(boxes.slice(i*4,i*4+4)));
          if(box[2]-box[0]>3&&box[3]-box[1]>2) proposals.push({box,score});
        }
      }
      const candidates=[];
      for(let q=0;q<output.veh_log_odds.data.length;q++) {
        const score=sigmoid(output.veh_log_odds.data[q]);
        if(score>=.28) candidates.push({box:Array.from(output.raw_q_boxes_px.data.slice(q*4,q*4+4)),score,q});
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
    const used=new Set();
    // Vehicle containment suppresses signboard/text proposals before expensive
    // localization. Keep standalone plate support when no vehicle is visible.
    const contained=proposals.filter(({box})=>!vehicles.length||vehicles.some(v=>{
      const cx=(box[0]+box[2])/2,cy=(box[1]+box[3])/2;
      return cx>=v.box[0]&&cx<=v.box[2]&&cy>=v.box[1]&&cy<=v.box[3];
    }));
    const counts=new Map();
    const selected=nms(contained,8).filter(candidate=>{
      const cx=(candidate.box[0]+candidate.box[2])/2,cy=(candidate.box[1]+candidate.box[3])/2;
      const owner=vehicles.filter(v=>cx>=v.box[0]&&cx<=v.box[2]&&cy>=v.box[1]&&cy<=v.box[3])
        .sort((a,b)=>(a.box[2]-a.box[0])*(a.box[3]-a.box[1])-(b.box[2]-b.box[0])*(b.box[3]-b.box[1]))[0];
      const key=owner??'standalone',count=counts.get(key)??0;
      if(count>=2) return false;
      counts.set(key,count+1);return true;
    });
    for(const candidate of selected) {
      const [fw,sw]=windows(candidate.box);
      const localized=await sessions.stage2.run({fovea_rois:patch(fw,160,96),surround_rois:patch(sw,96,64),sampling_policy:integer(candidate.track?2:0)});
      const llr=localized.plate_null_llr.data[0],support=sigmoid(llr);
      if(!localized.geometry_valid.data[0]||support<min_is_plate_prob) {
        if(candidate.track) candidate.track.reacquire=true;
        continue;
      }
      const corners=Array.from({length:4},(_,i)=>[localized.corners_norm.data[i*2]*fw[2]+fw[0],localized.corners_norm.data[i*2+1]*fw[3]+fw[1]]);
      if(!corners.flat().every(Number.isFinite)||corners.some(([x,y])=>x<0||x>wOrig-1||y<0||y>hOrig-1)) continue;
      let convex=true;
      for(let i=0;i<4;i++) {const a=corners[i],b=corners[(i+1)%4],c=corners[(i+2)%4];if((b[0]-a[0])*(c[1]-b[1])-(b[1]-a[1])*(c[0]-b[0])<=0) convex=false;}
      if(!convex) continue;
      const box=[Math.min(...corners.map(p=>p[0])),Math.min(...corners.map(p=>p[1])),Math.max(...corners.map(p=>p[0])),Math.max(...corners.map(p=>p[1]))];
      let track=tracks.filter(t=>t.state==='COLLECTING'&&!used.has(t.id)&&time-t.last<1.5&&iou(predictedBox(t,time),box)>.25)
        .sort((a,b)=>iou(predictedBox(b,time),box)-iou(predictedBox(a,time),box))[0];
      track??=newTrack(box,time);used.add(track.id);
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
    frameIndex++;
    return {timestamp:time,wOrig,hOrig,frameIndex,fullDetections,fastFrames,proposalCount:contained.length,localizedCount:selected.length,
      vehicles:vehicles.map(({box,mask})=>({box,mask})),
      tracks:tracks.filter(t=>t.state==='COLLECTING'&&used.has(t.id)).map(t=>({id:t.id,box:t.box,corners:t.corners,state:t.state,frames:t.frames,readouts:t.readouts}))};
  }
  async function finalize() {
    if(phase!=='COLLECTING') throw Error('Terminal capability already consumed');
    phase='SEALED';
    // Seal every collector before any character projection can execute.
    for(const track of tracks) {
      if(track.state!=='COLLECTING'||track.readouts) throw Error('Terminal capability already consumed');
      track.state='SEALED';
    }
    const results=[];
    for(const track of tracks) {
      try {
        if(track.state!=='SEALED'||track.readouts!==0) throw Error('Readout requires an unused sealed capability');
        track.readouts=1; // Consumed before awaiting: exceptions cannot permit retry.
        const ceiling=policy.terminal.readout_precision_ceiling,precision=mean(track.memory.mem_P.data);
        if(ceiling>0&&precision>ceiling) for(const name of ['mem_h','mem_P'])
          track.memory[name]=float(Array.from(track.memory[name].data,x=>x*ceiling/precision),track.memory[name].dims);
        const feed=Object.fromEntries(['mem_h','mem_P','mem_conflict','color_h','color_P','pol_h','pol_P'].map(k=>[k,track.memory[k]]));
        const readout=await sessions.memory_readout.run(feed);
        const decision=terminalDecision(readout,track.memory,track.frames,track.frames===1?track.firstSupport:sigmoid(track.odds),grammar,policy);
        results.push({track_id:track.id,...decision,frames:track.frames,readouts:track.readouts,state:'FINALIZED',thumbnail:track.thumbnail});
      } catch(error) {
        results.push({track_id:track.id,status:'UNREADABLE',confidence_tier:'NONE',text:null,frames:track.frames,readouts:track.readouts,state:'FINALIZED',error:error.message});
      } finally {track.state='FINALIZED';}
    }
    phase='FINALIZED';return results;
  }
  return {reset,process,finalize};
}

// RPC commands are strictly serialized. Only a finalization command can enter
// SEALED and consume the sole terminal readout capability.
if(typeof self!=='undefined'&&typeof self.postMessage==='function'&&typeof document==='undefined') {
  let pipeline,queue=Promise.resolve();
  self.onmessage=({data})=>{
    queue=queue.then(async()=>{
      try {
        let result;
        if(data.type==='init') {pipeline??=await createPipeline({progress:message=>self.postMessage({type:'progress',message})});result=true;}
        else if(data.type==='reset') {pipeline.reset();result=true;}
        else if(data.type==='frame') result=await pipeline.process(data);
        else if(data.type==='finalize') result=await pipeline.finalize();
        else throw Error('Unknown worker command');
        const transfer=data.type==='finalize'?result.filter(r=>r.thumbnail).map(r=>r.thumbnail.buffer):[];
        self.postMessage({id:data.id,type:data.type,result},transfer);
      } catch(error) {self.postMessage({id:data.id,error:error.message});}
    });
  };
}
