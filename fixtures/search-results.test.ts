import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createIndex } from '../packages/core/src/index.js';
import { loadModel } from '../packages/model/runtime.js';
import { searchResults } from '../apps/demo/src/ranking.js';

test('real model cannot displace exact, typo, prefix, or alias matches', async () => {
  const root = new URL('../packages/model/experiments/navigation-align0p5-seed29/', import.meta.url);
  const model = await loadModel(JSON.parse(readFileSync(new URL('manifest.json', root), 'utf8')), Uint8Array.from(readFileSync(new URL('weights.bin', root))).buffer);
  const candidates = [{ id: 'profile', label: 'Profile' }, { id: 'invoices', label: 'Invoices' }, { id: 'keys', label: 'API Keys', aliases: ['credentials'] }];
  const index = await createIndex(candidates);
  for (const [query, expected] of [['profle', 'profile'], ['profile', 'profile'], ['api k', 'keys'], ['credentials', 'keys']]) {
    const text = await index.search(query!);
    const rows = searchResults(text.results, model.score(query!, candidates));
    assert.equal(rows[0]?.id, expected);
    assert.ok(rows.every(row => row.detail !== 'model suggestion'));
  }
  assert.notEqual(model.score('profle', candidates)[0]?.id, 'profile', 'fixture exercises the actual model failure');
  index.dispose();
});

test('unmatched queries expose real suggestions explicitly and empty vectors return nothing', () => {
  const suggestions = [{ id: 'arbitrary', label: 'User supplied destination', score: .8 }];
  assert.deepEqual(searchResults([], suggestions), [{ id: 'arbitrary', label: 'User supplied destination', detail: 'model suggestion' }]);
  assert.deepEqual(searchResults([], []), []);
  assert.deepEqual(searchResults([], suggestions, 0), []);
});
