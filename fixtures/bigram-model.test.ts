import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { bigramFeatures } from '../packages/model/features-bigram.js';
import { loadBigramModel as loadModel } from '../packages/model/runtime-bigram.js';

const featureFixtures = JSON.parse(readFileSync(new URL('features-bigram.json', import.meta.url), 'utf8'));
const root = new URL('../packages/model/candidate-v2/', import.meta.url);

test('ordered bigram hashing matches Python for Unicode, boundaries, repeats, and truncation', () => {
  for (const { text, ...expected } of featureFixtures) assert.deepEqual(bigramFeatures(text), expected, text);
  assert.notDeepEqual(bigramFeatures('import data').bigramIds, bigramFeatures('data import').bigramIds);
  assert.equal(bigramFeatures('single').bigramCount, 0);
  assert.equal(bigramFeatures('a '.repeat(40)).bigramCount, 31);
});

test('actual bigram int8 encoder and scores match independent NumPy', async () => {
  const manifest = JSON.parse(readFileSync(new URL('manifest.json', root), 'utf8'));
  const payload = Uint8Array.from(readFileSync(new URL('weights.bin', root))).buffer;
  const fixtures: { text: string; vectors: number[] }[] = JSON.parse(readFileSync(new URL('numerical-fixtures.json', root), 'utf8'));
  const model = await loadModel(manifest, payload);
  for (const fixture of fixtures) {
    const actual = model.encode(fixture.text);
    assert.ok(actual.every((value, index) => Math.abs(value - fixture.vectors[index]) <= 1e-4), fixture.text);
    for (const score of model.score(fixture.text, fixtures.map((row, index) => ({ id: String(index), label: row.text })))) {
      const expected = fixture.vectors.reduce((sum, value, index) => sum + value * fixtures[Number(score.id)].vectors[index], 0);
      assert.ok(Math.abs(score.score - expected) <= 2e-4, fixture.text);
    }
  }
  assert.deepEqual(model.score('', [{ id: 'x', label: 'Import data' }]), []);
  await assert.rejects(loadModel({ ...manifest, featureVersion: 'unknown-word-order-contract' }, payload), /featureVersion/);
  // Loading legacy weights under v1 remains separately covered by expanded-model.test.ts.
});
