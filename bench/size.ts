import { build } from 'esbuild';
import { brotliCompressSync, gzipSync, constants } from 'node:zlib';
import { mkdirSync, writeFileSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
const measure = (bytes: Uint8Array) => ({ raw: bytes.length, gzip: gzipSync(bytes, {level:9}).length, brotli: brotliCompressSync(bytes,{params:{[constants.BROTLI_PARAM_QUALITY]:11,[constants.BROTLI_PARAM_LGWIN]:22}}).length });
const entries = [];
for (const entry of ['index','lexical']) {
 const result=await build({entryPoints:[`packages/core/src/${entry}.ts`],bundle:true,minify:true,format:'esm',platform:'browser',target:'es2022',write:false});
 entries.push({resource:`core/${entry}.js`,...measure(result.outputFiles[0].contents)});
}
const walk=(dir:string):string[]=>readdirSync(dir,{withFileTypes:true}).flatMap(d=>d.isDirectory()?walk(join(dir,d.name)):[join(dir,d.name)]);
const demo=walk('apps/demo/dist').map(path=>({resource:path,...measure(readFileSync(path))}));
const report={generatedAt:new Date().toISOString(),node:process.version,esbuild:'0.27.2',compression:'Node zlib; gzip level9; Brotli quality11 window22; each served resource separately',library:entries,demo,demoTotals:demo.reduce((a,r)=>({raw:a.raw+r.raw,gzip:a.gzip+r.gzip,brotli:a.brotli+r.brotli}),{raw:0,gzip:0,brotli:0}),semantic:'Demo ships experimental pooled16 CPU inference, manifest and int8 weights; core library remains lexical by default. No validated cutoff.',webgpu:'Not shipped: semantic feasibility precedes GPU optimization',sourceMaps:'Not built or served'};
mkdirSync('bench/reports',{recursive:true});writeFileSync('bench/reports/size.json',JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
if(entries.find(x=>x.resource==='core/lexical.js')!.brotli>8192)throw new Error('Lexical transfer budget exceeded');
// Count the entire served CPU demo, including model, metadata, UI and favicon.
if(report.demoTotals.brotli>50*1024)throw new Error('Complete CPU demo exceeds the 50 KiB Brotli budget');
