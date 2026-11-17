// Apache-2.0. Finite-factor terminal MBR port of session_v2.py.
// All arithmetic after ONNX is Float64; no heuristic grammar bonuses.
export const clamp = (x, lo=0, hi=1) => Math.max(lo, Math.min(hi, x));
export const sigmoid = x => 1 / (1 + Math.exp(-x));
const clean = x => Number.isNaN(x) ? 0 : x === Infinity ? 50 : x === -Infinity ? -50 : x;
const sum = a => a.reduce((x,y) => x+y, 0);
const mean = a => sum(a)/a.length;
const lse2 = (a,b) => {
  const m = Math.max(a,b);
  return m === -Infinity ? m : m + Math.log(Math.exp(a-m)+Math.exp(b-m));
};
const lse = a => a.reduce(lse2, -Infinity);
const argmax = a => a.reduce((best,v,i) => v > a[best] ? i : best, 0);

export async function unpackGrammar(table) {
  if (table.valid_encoding !== 'zlib-packbits-little-c-order') throw Error('Unknown grammar format');
  const compressed = Uint8Array.from(atob(table.valid_base64), c => c.charCodeAt(0));
  const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('deflate'));
  table.validBits = new Uint8Array(await new Response(stream).arrayBuffer());
  table.serials = Array.from({length:9999}, (_,i) => {
    const digits = String(i+1).split('').map(c => table.tokens.indexOf(c));
    return [...Array(4-digits.length).fill(table.dot_id), ...digits];
  });
  return table;
}

// Receives one sealed readout, the precision state and calibrated episode support.
export function terminalDecision(readout, memory, frames, support, table, policy) {
  const cal = policy.terminal, cfg = policy.session;
  const precision = Array.from({length:12}, (_,s) => mean(Array.from(memory.mem_P.data.slice(s*112,(s+1)*112), x => Number.isFinite(x) ? Math.max(0,x) : 0)));
  const lp = Array.from({length:12}, (_,s) => {
    const row = Array.from(readout.slot_logits.data.slice(s*254,(s+1)*254), x => precision[s] > 0 ? clean(x)/cal.temperature : 0);
    const z = lse(row); return row.map(x => x-z);
  });
  const reg = table.regions.map(row => sum(row.map((id,s) => lp[s][id])));
  const cls = table.classes.map(row => sum(row.map((id,s) => lp[s+4][id])));
  const ser = table.serials.map(row => sum(row.map((id,s) => lp[s+8][id])));
  const colorMean = mean(memory.color_P.data);
  const typeScores = Array.from(readout.plate_type_logits.data, x => colorMean > 0 ? clean(x)/cal.temperature : 0);
  const tz = lse(typeScores), typeLp = typeScores.map(x => x-tz);
  const nc=cls.length, nk=table.kana_ids.length;
  const joint = new Float64Array(5*nc*nk).fill(-Infinity);
  const ck = new Float64Array(nc*nk).fill(-Infinity);
  for (let t=0;t<5;t++) for (let c=0;c<nc;c++) for (let k=0;k<nk;k++) {
    const idx=(t*nc+c)*nk+k;
    if ((table.validBits[idx>>3] >> (idx&7)) & 1) {
      joint[idx] = typeLp[t] + cls[c] + lp[7][table.kana_ids[k]];
      ck[c*nk+k] = lse2(ck[c*nk+k],joint[idx]);
    }
  }
  const pad=table.pad_id;
  const zr=lse2(lse(reg),sum(lp.slice(0,4).map(row=>row[pad])));
  const zs=lse2(lse(ser),sum(lp.slice(8,12).map(row=>row[pad])));
  const zck=lse2(lse(ck),sum(lp.slice(4,8).map(row=>row[pad])));
  const ir=argmax(reg), iz=argmax(ser), ij=argmax(ck), ic=Math.floor(ij/nk), ik=ij%nk;
  const status=Array.from(readout.status_logits.data,clean);
  const conflict=Math.max(0,...Array.from(memory.mem_conflict.data, x => Number.isFinite(x) ? x : 0));
  status[2]=cal.mixed_logit_scale*status[2]+cal.mixed_bias+cal.conflict_weight*conflict;
  const mixed=Math.exp(status[2]-lse(status));
  const visibility=mean(Array.from(readout.slot_readability_pred.data,x=>clamp(Number.isNaN(x)?0:x)));
  const nullOdds=(status[0]-status[1]+zs+cal.null_bias+cal.readability_weight*visibility+cal.support_weight*clamp(support))/cal.null_temperature;
  let event=clamp(support)*(1-mixed)*sigmoid(nullOdds)*cal.identity_probability;
  if (!frames || !precision.some(x=>x>0)) event=0;
  const classMass=lse(ck.slice(ic*nk,(ic+1)*nk));
  const kanaMass=lse(Array.from({length:nc},(_,c)=>ck[c*nk+ik]));
  const probabilities={region:event*Math.exp(reg[ir]-zr),classification:event*Math.exp(classMass-zck),hiragana:event*Math.exp(kanaMass-zck),serial:event*Math.exp(ser[iz]-zs)};
  const ranges={region:[0,4],classification:[4,7],hiragana:[7,8],serial:[8,12]};
  for (const [name,[start,end]] of Object.entries(ranges)) if (!precision.slice(start,end).every(x=>x>0)) probabilities[name]=0;
  let jointP=event*Math.exp(reg[ir]+ser[iz]+ck[ij]-zr-zs-zck);
  if (!precision.every(x=>x>0)) jointP=0;
  const fields=Object.keys(probabilities).filter(f=>(cfg.field_wrong_costs?.[f]??cfg.C_wrong)*(1-probabilities[f])<(cfg.field_abstain_costs?.[f]??cfg.C_abstain));
  const full=cfg.C_wrong_full*(1-jointP)<cfg.C_abstain_full;
  const accepted=fields.includes('serial')||full;
  const serial=String(iz+1).padStart(4,'0');
  const shownSerial=iz+1>=1000?`${serial.slice(0,2)}-${serial.slice(2)}`:'・'.repeat(4-String(iz+1).length)+String(iz+1);
  const values={region:table.region_names[ir],classification:table.class_strings[ic],hiragana:table.kana_chars[ik],serial:shownSerial};
  const masks={region:'??',classification:'???',hiragana:'?',serial:'??-??'};
  return {
    status:accepted?'ACCEPTED':mixed>1-mixed-event?'TRACK_MIXED':'UNREADABLE',
    confidence_tier:accepted?(full?'FULL_4FIELD':'CORE_VERIFIED'):'NONE',
    verified_fields:accepted?fields:[],
    text:accepted?Object.keys(values).map(f=>full||fields.includes(f)?values[f]:masks[f]).join(' '):null,
    ...(accepted?{serial}:{}), field_confidences:probabilities,
    joint_correctness_probability:jointP, readable_probability:event,
  };
}
