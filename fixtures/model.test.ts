import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { loadModel, ModelAssetError } from '../packages/model/runtime.js';

const location = new URL('../packages/model/experimental/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('manifest.json', location), 'utf8'));
const raw = readFileSync(new URL('weights.bin', location));
const payload = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
const fixtures = JSON.parse(readFileSync(new URL('numerical-fixtures.json', location), 'utf8')) as { text: string; vectors: number[] }[];
async function rehash(bytes: ArrayBuffer) {
  const copy = structuredClone(manifest);
  copy.payloadSha256 = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', bytes)), b => b.toString(16).padStart(2, '0')).join('');
  return copy;
}
const dot = (a: ArrayLike<number>, b: ArrayLike<number>) => Array.from(a).reduce((sum, value, i) => sum + value * b[i]!, 0);
const unit = (v: number[]) => { const n = Math.sqrt(dot(v, v)); return v.map(value => value / Math.max(n, 1e-8)); };

test('actual exported int8 model matches quantized Python embeddings and menu cosines', async () => {
  const model = await loadModel(manifest, payload);
  assert.equal(model.id, manifest.modelId); assert.equal(model.hash, manifest.payloadSha256);
  for (const fixture of fixtures) {
    const actual = model.encode(fixture.text);
    assert.equal(actual.length, 16);
    assert.ok(actual.every((value, i) => Math.abs(value - fixture.vectors[i]!) <= 1e-4), fixture.text);
    const candidates = fixtures.map((f, i) => ({ id: String(i), label: f.text }));
    const scored = model.score(fixture.text, candidates);
    if (!fixture.text) { assert.deepEqual(scored, []); continue; }
    assert.equal(scored.length, fixtures.filter(f => f.text).length);
    for (const result of scored) assert.ok(Math.abs(result.score - dot(fixture.vectors, fixtures[Number(result.id)]!.vectors)) <= 2e-4);
  }
});

test('candidate aliases and context use the documented trained-vector composition', async () => {
  const model = await loadModel(manifest, payload);
  const a = fixtures.find(f => f.text === 'Profile')!.vectors;
  const b = fixtures.find(f => f.text === 'coworkers')!.vectors;
  const c = fixtures.find(f => f.text === 'C++')!.vectors;
  const composed = unit(a.map((v, i) => (v + b[i]!) / 2 + c[i]! * 0.25));
  const scored = model.score('Profile', [{ id: 'composed', label: 'Profile', aliases: ['coworkers', ''], context: 'C++' }]);
  assert.ok(Math.abs(scored[0]!.score - dot(a, composed)) <= 2e-4);
  const tied = model.score('Profile', [{ id: 'z', label: 'Profile' }, { id: 'a', label: 'Profile' }]);
  assert.deepEqual(tied.map(r => r.id), ['z', 'a']);
});

test('embeddings and scores depend on weight bytes; zero weights produce no semantic results', async () => {
  const model = await loadModel(manifest, payload);
  const zeros = new ArrayBuffer(payload.byteLength);
  const zeroModel = await loadModel(await rehash(zeros), zeros);
  assert.deepEqual(Array.from(zeroModel.encode('Profile')), Array(16).fill(0));
  assert.deepEqual(zeroModel.score('Profile', [{ id: 'a', label: 'Profile' }]), []);
  const altered = payload.slice(0), codes = new Int8Array(altered);
  // Change the learned Profile word row while preserving format and valid quantization.
  for (let i = 17 * 16; i < 18 * 16; i++) codes[i] = i % 2 ? 127 : -127;
  const changed = await loadModel(await rehash(altered), altered);
  assert.ok(changed.encode('Profile').some((v, i) => Math.abs(v - model.encode('Profile')[i]!) > 1e-3));
  const candidates = [{ id: 'a', label: 'coworkers' }, { id: 'b', label: 'C++' }];
  const before = new Map(model.score('Profile', candidates).map(r => [r.id, r.score]));
  assert.ok(changed.score('Profile', candidates).some(r => Math.abs(r.score - before.get(r.id)!) > 1e-3));
});

test('invalid or incompatible model assets are rejected before exposure', async () => {
  const mutations: ((m: typeof manifest) => void)[] = [
    m => m.formatVersion = 'unknown', m => m.featureVersion = 'other', m => m.normalizationVersion = 'other',
    m => m.candidateCompositionVersion = 'other', m => m.architecture = 'projected', m => m.dimension = 24,
    m => m.featureFamily = 'word', m => m.byteOrder = 'big-endian', m => m.modelId = '',
    m => m.validatedSemanticCutoff = 0.5, m => m.payloadSha256 = 'a'.repeat(64), m => m.payloadBytes++,
    m => m.tensors.reverse(), m => m.tensors.pop(), m => m.tensors[0].shape = [512, 32],
    m => m.tensors[0].elementCount--, m => m.tensors[0].bits = 6, m => m.tensors[1].byteOffset = 0,
    m => m.tensors[0].byteLength--, m => m.tensors[0].scale = NaN, m => m.tensors[0].scale = -1,
    m => m.tensors[0].scaleF32LE = '0000803f',
  ];
  for (const mutate of mutations) { const copy = structuredClone(manifest); mutate(copy); await assert.rejects(loadModel(copy, payload), ModelAssetError); }
  await assert.rejects(loadModel(null, payload), ModelAssetError);
  await assert.rejects(loadModel(manifest, payload.slice(1)), ModelAssetError);
  const reserved = payload.slice(0); new Int8Array(reserved)[0] = -128;
  await assert.rejects(loadModel(await rehash(reserved), reserved), /Reserved int8 code/);
  const corrupt = payload.slice(0); new Int8Array(corrupt)[0] = 0;
  await assert.rejects(loadModel(manifest, corrupt), /SHA-256 mismatch/);
});

test('loading snapshots caller data and returned embeddings cannot mutate the model', async () => {
  const bytes = payload.slice(0), metadata = structuredClone(manifest);
  const pending = loadModel(metadata, bytes);
  new Uint8Array(bytes).fill(0); metadata.modelId = 'mutated';
  const model = await pending;
  assert.equal(model.id, manifest.modelId);
  const before = Array.from(model.encode('Profile'));
  model.encode('Profile').fill(0);
  assert.deepEqual(Array.from(model.encode('Profile')), before);
});
