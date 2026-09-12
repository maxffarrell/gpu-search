import { chromium, webkit } from 'playwright';
import { createServer } from 'vite';
import { mkdirSync,writeFileSync } from 'node:fs';
import { cpus,platform,release } from 'node:os';
const server=await createServer({root:'.',server:{host:'127.0.0.1',port:4179,strictPort:true}});await server.listen();
const reports=[];
try {
 for(const [name,engine] of [['chromium',chromium],['webkit',webkit]] as const){
 const browser=await engine.launch({headless:true, ...(name === 'chromium' && process.env.CHROME_CHANNEL ? {channel: process.env.CHROME_CHANNEL} : {}), ...(name === 'webkit' && process.env.WEBKIT_EXECUTABLE ? {executablePath: process.env.WEBKIT_EXECUTABLE} : {})});
 try {
 const page=await browser.newPage();await page.goto('http://127.0.0.1:4179/bench/harness.html');
 await page.evaluate('globalThis.__name = (target) => target');
 const requests:string[]=[];page.on('request',r=>requests.push(r.url()));
 const timings=await page.evaluate(async()=>{
 // This is a local benchmark harness, never included in the public demo.
 const {createIndex}=await import('/packages/core/src/index.ts' as string);
 let seed=20260912;const random=()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed/4294967296};
 const queries=['profile','profle','billing','api','team','xqzv','settings','member','delete account','subscription'];
 const rows=[];
 for(const count of [10,50,250,1000,5000,50000]){
 const candidates=Array.from({length:count},(_,i)=>({id:String(i),label:['Profile','Billing','Team Members','API Keys','Delete Account','Subscription','Notifications','Project Settings','Audit Log','Invoices'][i%10]+(i<10?'':` ${i}`),aliases:[`Destination ${i}`],context:i%2?'personal settings':'workspace access'}));
 const start=performance.now();const index=await createIndex(candidates,{semantic:false});const indexingMs=performance.now()-start;
 for(let i=0;i<30;i++)await index.search(queries[Math.floor(random()*queries.length)]);
 const samples=[];for(let i=0;i<200;i++){const q=queries[Math.floor(random()*queries.length)],t=performance.now();await index.search(q);samples.push(performance.now()-t)}
 const sorted=[...samples].sort((a,b)=>a-b);rows.push({count,indexingMs,p50:sorted[99],p95:sorted[189],samples});index.dispose();
 }
 return {rows,userAgent:navigator.userAgent,gpuAvailable:'gpu' in navigator,memory:(performance as unknown as {memory?:{usedJSHeapSize:number}}).memory?.usedJSHeapSize??null};
 });
 reports.push({browser:name,version:browser.version(),...timings,requestsDuringBenchmark:requests});
 }finally{await browser.close()}
 }
}finally{await server.close()}
const result={generatedAt:new Date().toISOString(),browserOverrides:{chromiumChannel:process.env.CHROME_CHANNEL??null,webkitExecutable:process.env.WEBKIT_EXECUTABLE??null},hardware:cpus()[0]?.model,os:`${platform()} ${release()}`,powerState:'Not controlled; local interactive development session',method:'Full lexical search; 30 warmups, 200 deterministic randomized queries per size; synthetic labels with aliases/context. No GPU/model shipped, no GPU timing claim. Heap unavailable on some browsers; timer resolution can round short calls to zero.',reports};
mkdirSync('bench/reports',{recursive:true});writeFileSync('bench/reports/browser.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({...result,reports:reports.map(r=>({...r,rows:r.rows.map(({samples,...rest})=>rest)}))},null,2));
