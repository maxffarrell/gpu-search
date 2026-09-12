import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createIndex,type Candidate} from '../packages/core/src/index.js';
const fixtures=JSON.parse(readFileSync(new URL('../eval/baseline-fixtures.json',import.meta.url),'utf8'));
const aliases:Record<string,string[]>=JSON.parse(readFileSync(new URL('../data/aliases.json',import.meta.url),'utf8'));
test('Python evaluation baselines rank the same as the shipped TypeScript engine',async()=>{
 for(const row of fixtures){
 for(const [enrich,key] of [[false,'lexicalIds'],[true,'aliasIds']] as const){
 const candidates=row.candidates.map((c:Candidate)=>({...c,aliases:[...(c.aliases??[]),...(enrich?aliases[c.label]??[]:[])]}));
 const index=await createIndex(candidates,{semantic:false});
 assert.deepEqual((await index.search(row.query,{limit:50000})).results.map(x=>x.id),row[key],`${row.id} ${key}`);index.dispose();
 }
 }
});
