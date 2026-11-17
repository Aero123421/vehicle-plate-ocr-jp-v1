// Apache-2.0. Local-only checks of the exact browser Worker pipeline.
// Usage: node tools/check_web_demo_parity.mjs
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {isAbsolute} from 'node:path';
import {spawn} from 'node:child_process';
import {performance} from 'node:perf_hooks';
import {createPipeline} from '../web/worker.mjs';

const started=performance.now();
const root=fileURLToPath(new URL('../',import.meta.url));
const ortModule=process.env.ORT_WEB_MODULE??'onnxruntime-web';
const ort=await import(isAbsolute(ortModule)?pathToFileURL(ortModule).href:ortModule);
ort.env.wasm.wasmBinary=await readFile(new URL('../web/vendor/ort-wasm-simd-threaded.wasm',import.meta.url));
function command(binary,args) {
  return new Promise((resolve,reject)=>{
    const child=spawn(binary,args,{cwd:root,env:{...process.env,PYTHONDONTWRITEBYTECODE:'1',PYTHONPATH:fileURLToPath(new URL('../src',import.meta.url)),OMP_NUM_THREADS:'1',MKL_NUM_THREADS:'1',OPENBLAS_NUM_THREADS:'1',NUMEXPR_NUM_THREADS:'1'}});
    const chunks=[],errors=[];
    child.stdout.on('data',chunk=>chunks.push(chunk));child.stderr.on('data',chunk=>errors.push(chunk));
    child.on('error',reject);child.on('close',code=>code===0?resolve(Buffer.concat(chunks)):reject(Error(`${binary}: ${Buffer.concat(errors)}`)));
  });
}
const python=command(process.env.PYTHON??'python3',['-B','-c',
  'import json,torch,cv2,vehicle_plate_ocr_jp_v1 as p; torch.set_num_threads(1); torch.set_num_interop_threads(1); cv2.setNumThreads(1); print(json.dumps(p.create_predictor().predict_clip("web/samples/synthetic_gate_demo.mp4")))']);
const decoding=command(process.env.FFMPEG??'ffmpeg',['-v','error','-threads','1','-i','web/samples/synthetic_gate_demo.mp4','-f','rawvideo','-pix_fmt','rgba','-threads','1','pipe:1']);
const readJSON=async path=>JSON.parse(await readFile(new URL(`../web/${path}`,import.meta.url),'utf8'));
const timings={},calls={},transitions=new Map(),sealed=new Set();
let strictBatch=true,collecting=false,readouts=0,vehicleFrames=0,plateFrames=0;
const runtime={Tensor:ort.Tensor,env:ort.env,InferenceSession:{
  async create(url,options) {
    const session=await ort.InferenceSession.create(await readFile(fileURLToPath(url)),options);
    const name=url.split('/').at(-1).split('.')[0];
    return {release:()=>session.release(),async run(feed) {
      calls[name]=(calls[name]??0)+1;
      if(name==='memory_readout') {
        if(strictBatch) assert.equal(collecting,false,'batch readout during COLLECTING');
        else assert.equal(sealed.size,1,'readout requires exactly one SEALED track');
        readouts++;
      }
      const begin=performance.now();const result=await session.run(feed);
      timings[name]=(timings[name]??0)+performance.now()-begin;return result;
    }};
  }
}};
function progress(event) {
  if(typeof event==='string') return;
  const history=transitions.get(event.track_id)??[];
  const expected=['COLLECTING','SEALED','FINALIZED'][history.length];
  assert.equal(event.state,expected,`invalid transition for track ${event.track_id}`);
  assert.equal(event.readouts,event.state==='COLLECTING'?0:1);
  history.push(event.state);transitions.set(event.track_id,history);
  if(event.state==='SEALED') sealed.add(event.track_id);
  if(event.state==='FINALIZED') sealed.delete(event.track_id);
}
const pipeline=await createPipeline({runtime,readJSON,progress});
pipeline.reset();collecting=true;
const raw=await decoding,frameBytes=768*448*4;
assert.equal(raw.length/frameBytes,28,'strict parity must analyze all 28 frames');
const frame=(index,timestamp=index/12)=>({rgbaBuffer:new Uint8ClampedArray(raw.buffer,raw.byteOffset+index*frameBytes,frameBytes),wOrig:768,hOrig:448,timestamp});
let last;
for(let index=0;index<28;index++) {
  last=await pipeline.process(frame(index));
  assert.equal(readouts,0);assert.equal(last.newlyFinalized.length,0);
  assert(last.tracks.every(t=>t.state==='COLLECTING'&&t.readouts===0));
  vehicleFrames+=Number(last.vehicles.length>0);plateFrames+=Number(last.tracks.length>0);
}
collecting=false;
const results=await pipeline.finalize();
assert(vehicleFrames>0,'sample vehicle was not detected');assert(plateFrames>0,'sample plate was not detected');
assert(results.every(r=>r.state==='FINALIZED'&&r.readouts===1));assert.equal(readouts,results.length);
const accepted=results.filter(r=>r.status==='ACCEPTED');
assert(accepted.length>0,`No accepted JS result: ${JSON.stringify(results.map(({thumbnail,...r})=>r))}`);
const expected=JSON.parse(String(await python));
assert(expected.length>0,'Python returned no accepted predictions');
assert.deepEqual([...new Set(accepted.map(r=>r.serial))].sort(),['3532']);
assert.deepEqual([...new Set(accepted.map(r=>r.serial))].sort(),[...new Set(expected.map(r=>r.serial))].sort());
assert(accepted.every(r=>r.thumbnail?.length===160*80*4));
await assert.rejects(pipeline.finalize(),/consumed/);await assert.rejects(pipeline.process({}),/closed/);
const batchSeconds=(performance.now()-started)/1000;

const batchReadouts=readouts,batchStage2=calls.stage2;
// Reuse loaded models, but enable exactly the browser's live configuration.
strictBatch=false;transitions.clear();sealed.clear();
pipeline.reset({autoSealFrames:6,autoSealMinSec:.45,autoExitSec:.50});collecting=true;
let firstLiveFrame=-1,reusedFast=0,reusedFull=0;
const liveResults=new Map();
for(let index=0;index<22;index++) {
  const before={...calls},previous=last;
  last=await pipeline.process(frame(index));
  assert(last.tracks.every(t=>t.state==='COLLECTING'&&t.readouts===0));
  for(const r of last.newlyFinalized) {
    assert(!liveResults.has(r.track_id),'duplicate terminal emission');
    assert.deepEqual(transitions.get(r.track_id),['COLLECTING','SEALED','FINALIZED']);
    assert.equal(r.readouts,1);assert.equal(r.state,'FINALIZED');
    liveResults.set(r.track_id,r);if(firstLiveFrame<0) firstLiveFrame=index;
  }
  assert(last.finalizedTracks.every(t=>t.readouts===1&&liveResults.has(t.id)));
  if(last.finalizedTracks.length&&!last.newlyFinalized.length) {
    if(!last.localizedCount) for(const name of ['stage2','frame_encoder','memory_step','memory_readout'])
      assert.equal(calls[name]??0,before[name]??0,`${name} ran for a covered finalized track`);
    if(last.fullDetections>previous.fullDetections) reusedFull++;else reusedFast++;
  }
}
assert(firstLiveFrame>=0&&firstLiveFrame<27,'no mid-clip auto-seal');
assert(reusedFast>0,'finalized track never reused on a fast frame');
assert(reusedFull>0,`finalized track never reused on a full frame: first=${firstLiveFrame}, fast=${reusedFast}`);
assert.equal(readouts-batchReadouts,liveResults.size,'a track received a second readout');
const liveStage2=calls.stage2-batchStage2;
assert(liveStage2<batchStage2,'auto-seal failed to reduce Stage-2 work');
// Backward seek, then seek with an open two-frame collector, then discard a
// one-frame collector. Each path must reset temporal tensors and timestamps.
await pipeline.resegment({autoSealFrames:0,autoExitSec:0});
let seekPayload=await pipeline.process(frame(0,.1)),seekIndex=0;
while(!seekPayload.tracks.some(t=>t.frames>=2)&&seekIndex<6) {
  seekIndex++;seekPayload=await pipeline.process(frame(seekIndex,.1+seekIndex/12));
}
assert(seekPayload.tracks.some(t=>t.frames>=2));
const seekResult=await pipeline.resegment();
assert(seekResult.finalizedBeforeSeek.length>0);
assert(seekResult.finalizedBeforeSeek.every(t=>t.readouts===1));
let discardPayload=await pipeline.process(frame(0,.1)),discardIndex=0;
while(!discardPayload.tracks.length&&discardIndex<6) {
  discardIndex++;discardPayload=await pipeline.process(frame(discardIndex,.1+discardIndex/12));
}
assert(discardPayload.tracks.every(t=>t.frames===1));
const beforeDiscard=readouts;
assert.deepEqual((await pipeline.resegment()).finalizedBeforeSeek,[]);
assert.equal(readouts,beforeDiscard,'one-frame seek collector should be discarded');
await pipeline.process(frame(0,.1));
const all=await pipeline.finalize();collecting=false;
assert(all.every(r=>r.readouts===1&&r.state==='FINALIZED'));
assert.equal(new Set(all.map(r=>r.track_id)).size,all.length);
assert.equal(all.length,readouts-batchReadouts,'lost or repeated terminal results');
await assert.rejects(pipeline.finalize(),/consumed/);
await pipeline.resegment();await pipeline.process(frame(0,.1)); // EOF replay.
await pipeline.finalize();await pipeline.dispose();
// Simulate an isolation-capable browser with a failed multi-thread session.
// No actual extra Worker or network requests are needed for this fallback test.
const previousSelf=globalThis.self;globalThis.self={crossOriginIsolated:true};
let released=0,attempts=0;
const fallbackEnv={wasm:{}};
try {
  const fallback=await createPipeline({readJSON,runtime:{Tensor:ort.Tensor,env:fallbackEnv,InferenceSession:{async create() {
    attempts++;
    if(fallbackEnv.wasm.numThreads>1&&attempts===2) throw Error('thread startup failed');
    return {async release(){released++;}};
  }}}});
  assert.equal(fallbackEnv.wasm.numThreads,1);assert.equal(attempts,8);assert.equal(released,1);
  await fallback.dispose();assert.equal(released,7);
} finally {if(previousSelf===undefined) delete globalThis.self;else globalThis.self=previousSelf;}
console.log(JSON.stringify({python:expected,javascript:accepted.map(({thumbnail,...r})=>r),vehicleFrames,plateFrames,
  batchStreamingReadouts:0,batchTerminalReadouts:batchReadouts,batchSeconds:Number(batchSeconds.toFixed(3)),
  realtime:{firstLiveFrame,reusedFast,reusedFull,batchStage2,liveStage2,seekVerified:true},timings,
  seconds:Number(((performance.now()-started)/1000).toFixed(3))},null,2));

assert(batchSeconds<12.0,`Batch parity including Python/ffmpeg exceeded 12.0s: ${batchSeconds.toFixed(3)}s`);
