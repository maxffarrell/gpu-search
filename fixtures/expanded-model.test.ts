import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {loadModel} from '../packages/model/runtime.js';
const root=new URL('../packages/model/candidate/',import.meta.url);
const manifest=JSON.parse(readFileSync(new URL('manifest.json',root),'utf8'));
const fixtures: {text:string; vectors:number[]}[]=JSON.parse(readFileSync(new URL('numerical-fixtures.json',root),'utf8'));
test('expanded int8 CPU embeddings and scores match independent exported NumPy fixtures',async()=>{
 const model=await loadModel(manifest,Uint8Array.from(readFileSync(new URL('weights.bin',root))).buffer);
 for(const f of fixtures){
  const v=model.encode(f.text);assert.ok(v.every((x,i)=>Math.abs(x-f.vectors[i])<=1e-4),f.text);
  for(const score of model.score(f.text,fixtures.map((c,i)=>({id:String(i),label:c.text})))){
   const expected=f.vectors.reduce((sum,x,i)=>sum+x*fixtures[Number(score.id)].vectors[i],0);
   assert.ok(Math.abs(score.score-expected)<=2e-4);
  }
 }
});

test('trained UI sanity cases retrieve Members and Profile without aliases',async()=>{
 const model=await loadModel(manifest,Uint8Array.from(readFileSync(new URL('weights.bin',root))).buffer);
 const candidates=['Profile','Profiles','Members','Invoices','API Keys','Plan','Billing','Notifications','Create Account','Delete Account'].map(label=>({id:label,label}));
 // These queries were in original training; this protects basic behavior, not generalization.
 assert.equal(model.score('coworkers',candidates)[0]?.label,'Members');
 assert.equal(model.score('my information',candidates)[0]?.label,'Profile');
});
