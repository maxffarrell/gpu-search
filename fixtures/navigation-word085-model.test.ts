import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { loadModel } from '../packages/model/runtime.js';

const root = new URL('../packages/model/experiments/navigation-align0p5-seed29-word085/', import.meta.url);
const manifest = JSON.parse(readFileSync(new URL('manifest.json', root), 'utf8'));
const fixture = JSON.parse(readFileSync(new URL('navigation-word085-model.json', import.meta.url), 'utf8')) as {
  modelSha256: string; manifestSha256: string;
  candidates: { id: string; label: string; aliases?: string[]; context?: string }[];
  cases: { text: string; vector: number[]; scores: number[] }[];
};
test('selected word085 navigation int8 model matches independent NumPy embeddings and composed menu scores', async () => {
  const bytes = Uint8Array.from(readFileSync(new URL('weights.bin', root))).buffer;
  const model = await loadModel(manifest, bytes);
  assert.equal(model.hash, fixture.modelSha256);
  assert.equal(createHash('sha256').update(readFileSync(new URL('manifest.json', root))).digest('hex'), fixture.manifestSha256, 'bind effective tensor scales, not only shared weight payload');
  assert.equal(bytes.byteLength, 32768);
  const index = model.prepare(fixture.candidates);
  for (const item of fixture.cases) {
    const vector = model.encode(item.text);
    assert.equal(vector.length, item.vector.length);
    assert.ok(vector.every((value, i) => Math.abs(value - item.vector[i]!) <= 1e-4), item.text);
    const results = index.score(item.text);
    assert.deepEqual(results, model.score(item.text, fixture.candidates));
    if (!item.text) { assert.deepEqual(results, []); continue; }
    assert.equal(results.length, fixture.candidates.length);
    for (const row of results) {
      const expected = item.scores[fixture.candidates.findIndex(c => c.id === row.id)]!;
      assert.ok(Math.abs(row.score - expected) <= 2e-4, `${item.text}: ${row.id}`);
    }
  }
  index.dispose();
});
