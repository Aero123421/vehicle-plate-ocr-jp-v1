// Apache-2.0. Local-only release smoke check of the exact browser Worker pipeline.
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
    const child=spawn(binary,args,{cwd:root,env:{...process.env,PYTHONDONTWRITEBYTECODE:'1',OMP_NUM_THREADS:'2',MKL_NUM_THREADS:'2'}});
    const chunks=[],errors=[];
    child.stdout.on('data',chunk=>chunks.push(chunk));child.stderr.on('data',chunk=>errors.push(chunk));
    child.on('error',reject);child.on('close',code=>code===0?resolve(Buffer.concat(chunks)):reject(Error(`${binary}: ${Buffer.concat(errors)}`)));
  });
}
const python=command(process.env.PYTHON??'python3',['-B','-c',
  'import json,torch,vehicle_plate_ocr_jp_v1 as p; torch.set_num_threads(2); print(json.dumps(p.create_predictor().predict_clip("web/samples/synthetic_gate_demo.mp4")))']);
const ffmpeg=process.env.FFMPEG??'ffmpeg';
const decoding=command(ffmpeg,['-v','error','-threads','1','-i','web/samples/synthetic_gate_demo.mp4','-f','rawvideo','-pix_fmt','rgba','-threads','1','pipe:1']);
let collecting=false,readouts=0,vehicleFrames=0,plateFrames=0;
const timings={};
const runtime={Tensor:ort.Tensor,env:ort.env,InferenceSession:{
  async create(url,options) {
    const session=await ort.InferenceSession.create(await readFile(fileURLToPath(url)),options);
    return {release:()=>session.release(),async run(feed) {
      if(url.includes('/memory_readout.')) {assert.equal(collecting,false,'readout during COLLECTING');readouts++;}
      const begin=performance.now();const result=await session.run(feed);
      const key=url.split('/').at(-1);timings[key]=(timings[key]??0)+performance.now()-begin;return result;
    }};
  }
}};
const pipeline=await createPipeline({runtime,readJSON:async path=>JSON.parse(await readFile(new URL(`../web/${path}`,import.meta.url),'utf8'))});
pipeline.reset();collecting=true;
const raw=await decoding,frameBytes=768*448*4;
assert.equal(raw.length%frameBytes,0);
let last;
for(let index=0;index<raw.length/frameBytes;index++) {
  const rgbaBuffer=new Uint8ClampedArray(raw.buffer,raw.byteOffset+index*frameBytes,frameBytes);
  last=await pipeline.process({rgbaBuffer,wOrig:768,hOrig:448,timestamp:index/12});
  if(process.env.PARITY_DEBUG) console.error(index,JSON.stringify({full:last.fullDetections,fast:last.fastFrames,vehicles:last.vehicles.length,proposals:last.proposalCount,localized:last.localizedCount,tracks:last.tracks.map(t=>[t.id,t.frames])}));
  assert.equal(readouts,0);
  assert(last.tracks.every(t=>t.state==='COLLECTING'&&t.readouts===0));
  vehicleFrames+=Number(last.vehicles.length>0);plateFrames+=Number(last.tracks.length>0);
}
collecting=false;
const results=await pipeline.finalize();
assert(vehicleFrames>0,'sample vehicle was not detected');
assert(plateFrames>0,'sample plate was not detected');
assert(results.every(r=>r.state==='FINALIZED'&&r.readouts===1));
assert.equal(readouts,results.length);
const accepted=results.filter(r=>r.status==='ACCEPTED');
assert(accepted.length>0,`No accepted JS result: ${JSON.stringify(results.map(({thumbnail,...r})=>r))}`);
const expected=JSON.parse(String(await python));
assert(expected.length>0,'Python returned no accepted predictions');
assert.deepEqual([...new Set(accepted.map(r=>r.serial))].sort(),[...new Set(expected.map(r=>r.serial))].sort());
assert(accepted.every(r=>r.thumbnail?.length===160*80*4));
await assert.rejects(pipeline.finalize(),/consumed/);
await assert.rejects(pipeline.process({}),/closed/);
const seconds=(performance.now()-started)/1000;
console.log(JSON.stringify({python:expected,javascript:accepted.map(({thumbnail,...r})=>r),vehicleFrames,plateFrames,
  fullDetections:last.fullDetections,fastFrames:last.fastFrames,streamingReadouts:0,terminalReadouts:readouts,timings,seconds:Number(seconds.toFixed(3))},null,2));
assert(seconds<10,`Fast parity check exceeded 10 seconds: ${seconds.toFixed(3)}s`);
