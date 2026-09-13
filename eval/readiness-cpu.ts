import { readFileSync, writeFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { cpus, release } from 'node:os';
import { loadModel } from '../packages/model/runtime.js';

const manifest = JSON.parse(readFileSync('packages/model/candidate/manifest.json', 'utf8'));
const bytes = readFileSync('packages/model/candidate/weights.bin');
const started = performance.now();
const model = await loadModel(manifest, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
const loadMs = performance.now() - started;
const roots = ['Project settings', 'Team members', 'Billing details', 'API keys', 'Personal profile', 'Security settings', 'Notification preferences', 'Account access'];
const queries = ['coworkers', 'change my password', 'personal information', 'download invoices', 'developer credentials'];
const rows = [];
for (const count of [10, 100, 1000]) {
  const candidates = Array.from({ length: count }, (_, i) => ({ id: String(i), label: `${roots[i % roots.length]} ${i}`, aliases: [`${roots[(i + 1) % roots.length]} ${i}`], context: i % 2 ? 'Personal settings' : 'Organization settings' }));
  for (let i = 0; i < 30; i++) model.score(queries[i % queries.length]!, candidates);
  const samples = [];
  for (let i = 0; i < 200; i++) {
    const start = performance.now(); model.score(queries[(i * 7) % queries.length]!, candidates); samples.push(performance.now() - start);
  }
  samples.sort((a, b) => a - b);
  rows.push({ count, p50Ms: samples[100], p95Ms: samples[190], minMs: samples[0], maxMs: samples[199] });
}
const report = { scope: 'Local Node CPU diagnostic of existing raw semantic model.score; not browser proof or full hybrid search benchmark', hardware: cpus()[0]?.model, os: release(), node: process.version, arch: process.arch, powerState: 'uncontrolled', modelId: model.id, modelSha256: model.hash, loadMs, warmups: 30, samplesPerSize: 200, candidates: 'Synthetic varied labels with one alias and context each; same immutable list reused', rows, observation: 'Current model.score re-encodes each candidate on each query; no embedding index cache exists. No timing target is inferred for other devices.' };
writeFileSync('eval/readiness-cpu.json', JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report));
