import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createIndex, normalizeKey, tokenize, osaDistance, SearchInputError, IndexDisposedError, type Candidate } from '../src/index.js';
import { createIndex as createLexicalIndex } from '../src/lexical.js';

const menu = (labels: string[]): Candidate[] => labels.map((label, i) => ({ id: String(i), label }));
test('strict normalization retains punctuation, digits, diacritics and non-ASCII case', () => {
  assert.equal(normalizeKey('  ＰＲＯＦＩＬＥ\u0085 Settings \u2003'), 'profile settings');
  assert.equal(normalizeKey('C++ C# C'), 'c++ c# c');
  assert.equal(normalizeKey('É E\u0301 １２'), 'É É 12');
  assert.notEqual(normalizeKey('É'), normalizeKey('é'));
  assert.deepEqual(tokenize('personalSettings API_keys c++ C# user@home'), ['personal', 'settings', 'api', 'keys', 'c++', 'c#', 'user@home']);
});
test('label equality outranks alias equality and prefix regardless of insertion', async () => {
  const index = await createIndex([{ id: 'a', label: 'Account', aliases: ['profile'] }, { id: 'b', label: 'Profiles' }, { id: 'c', label: 'Profile' }]);
  const response = await index.search('PROFILE');
  assert.deepEqual(response.results.map(x => [x.id, x.reason]), [['c', 'label-equality'], ['a', 'alias-equality'], ['b', 'prefix']]);
  assert.equal(response.backend, 'lexical'); assert.equal(response.degraded, true);
  assert.equal((await createLexicalIndex([])).search instanceof Function, true);
});
test('all lexical subclasses, precedence and label field tie preference', async () => {
  const index = await createIndex([
    { id: 'typo', label: 'proflie' }, { id: 'token', label: 'Your Profile' },
    { id: 'prefix', label: 'Profiles and settings', aliases: ['Profiles and settings'] },
    { id: 'alias', label: 'Account', aliases: ['profile'] },
  ], { semantic: false });
  const { results } = await index.search('profile');
  assert.deepEqual(results.map(x => x.reason), ['alias-equality', 'prefix', 'token-prefix', 'typo']);
  assert.equal(results[1]!.matchedField, 'label');
  assert.equal((await (await createIndex(menu(['Application Programming Interface']))).search('api')).results[0]!.reason, 'acronym');
});
test('OSA transpositions, bounds and short-query safeguards', async () => {
  assert.equal(osaDistance('ab', 'ba'), 1);
  assert.equal(osaDistance('CA', 'ABC'), 3); // Unrestricted Damerau-Levenshtein would be 2.
  assert.equal(osaDistance('😀ab', '😀ba'), 1);
  const index = await createIndex(menu(['Profile', 'Cats', 'API Keys', 'Account', 'Export data']));
  assert.equal((await index.search('profle')).results[0]!.reason, 'typo');
  assert.deepEqual((await index.search('cat')).results.map(x => x.label), ['Cats']);
  assert.deepEqual((await index.search('c')).results, []);
  assert.deepEqual((await index.search('ats')).results, []);
  assert.deepEqual((await index.search('keys api')).results, []);
  assert.deepEqual((await index.search('xqzv')).results, []);
  assert.deepEqual((await index.search('ex dt')).results, []);
});
test('exact symbol labels stay distinct; split-token score remains bounded', async () => {
  const index = await createIndex(menu(['C', 'C#', 'C++', 'fooBar']));
  for (const label of ['C', 'C#', 'C++']) assert.equal((await index.search(label)).results[0]!.label, label);
  const result = (await index.search('foo-bar')).results[0]!;
  assert.equal(result.reason, 'token-prefix'); assert.ok(result.lexicalScore! <= 1);
});
test('duplicates use insertion ordering and context is never a matching field', async () => {
  const index = await createIndex([{ id: 'z', label: 'Profile', context: 'billing' }, { id: 'a', label: 'Profile' }]);
  assert.deepEqual((await index.search('profile')).results.map(x => x.id), ['z', 'a']);
  assert.deepEqual((await index.search('billing')).results, []);
});
test('snapshot, top-k, zero, empty and whitespace inputs', async () => {
  const source = [{ id: 'a', label: 'Account', aliases: ['Members'] }];
  const index = await createIndex(source, { semantic: false });
  source[0]!.label = 'Other'; source[0]!.aliases[0] = 'Something'; source.push({ id: 'b', label: 'Members', aliases: [] });
  assert.equal((await index.search('members')).results[0]!.label, 'Account');
  assert.equal((await index.search('members', { limit: 0 })).results.length, 0);
  assert.equal((await index.search(' \n ')).results.length, 0);
  assert.equal((await (await createIndex([])).search('profile')).results.length, 0);
  const ranked = await createIndex(menu(['Profiles', 'Profile', 'Profile detail']));
  assert.equal((await ranked.search('profile', { limit: 1 })).results[0]!.label, 'Profile');
});
test('invalid candidates and overflow throw typed errors', async () => {
  const invalid: unknown[] = [null, {}, [null], new Array(1), [{ id: 'a', label: '' }], [{ id: 'a', label: ' ' }], [{ id: 'a', label: 'x'.repeat(257) }], [{ id: 'a', label: '\ud800' }], [{ id: 'a', label: 'A', aliases: [3] }], [{ id: 'a', label: 'A', aliases: Array(9).fill('x') }], [{ id: 'a', label: 'A', context: 3 }], [{ id: 'a', label: 'A' }, { id: 'a', label: 'B' }]];
  for (const input of invalid) await assert.rejects(createIndex(input as Candidate[]), SearchInputError);
  await assert.rejects(createIndex(menu(Array(50_001).fill('A'))), SearchInputError);
  await createIndex([{ id: 'emoji', label: '😀'.repeat(256) }]);
  const index = await createIndex([]);
  for (const limit of [-1, 0.5, NaN, Infinity]) await assert.rejects(index.search('a', { limit }), SearchInputError);
  await assert.rejects(index.search('😀'.repeat(257)), SearchInputError);
});
test('concurrent calls isolate results and abort/dispose prevent stale delivery', async () => {
  const index = await createIndex(menu(['Profile', 'Billing']));
  const [a, b] = await Promise.all([index.search('profile'), index.search('billing')]);
  assert.equal(a.results[0]!.label, 'Profile'); assert.equal(b.results[0]!.label, 'Billing');
  const controller = new AbortController();
  const pending = index.search('profile', { signal: controller.signal }); controller.abort();
  await assert.rejects(pending, { name: 'AbortError' });
  await assert.rejects(index.search('billing', { signal: controller.signal }), { name: 'AbortError' });
  const stale = index.search('profile'); index.dispose(); index.dispose();
  await assert.rejects(stale, IndexDisposedError); await assert.rejects(index.search('profile'), IndexDisposedError);
});
test('backend and missing model reporting is explicit with no GPU/network requirement', async () => {
  for (const backend of ['cpu', 'auto', 'webgpu'] as const) {
    const response = await (await createIndex(menu(['Profile']), { backend })).search('profile');
    assert.equal(response.backend, 'lexical'); assert.equal(response.degraded, true);
  }
  assert.equal((await (await createLexicalIndex(menu(['Profile']))).search('profile')).degraded, false);
});

test('seeded adversarial menus preserve hard tiers, stable ties and top-k prefixes', async () => {
  let seed = 0x20260912;
  const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
  for (let iteration = 0; iteration < 200; iteration++) {
    const query = ['profile', 'billing', 'members', 'settings'][Math.floor(random() * 4)]!;
    const source: Candidate[] = [
      { id: 'exact-a', label: query.toUpperCase() }, { id: 'exact-b', label: ` ${query} ` },
      { id: 'alias-a', label: 'Destination one', aliases: [query] }, { id: 'alias-b', label: 'Destination two', aliases: [query] },
      { id: 'prefix', label: `${query}s` }, { id: 'token', label: `Your ${query}` },
      { id: 'typo', label: query.slice(0, -1) + 'z' }, { id: 'unrelated', label: 'Xqzv' },
    ];
    for (let i = source.length - 1; i > 0; i--) { const j = Math.floor(random() * (i + 1)); [source[i], source[j]] = [source[j]!, source[i]!]; }
    const index = await createIndex(source, { semantic: false });
    const all = await index.search(query, { limit: 100 });
    assert.deepEqual(all.results.slice(0, 2).map(r => r.id), source.filter(c => c.id.startsWith('exact-')).map(c => c.id));
    assert.deepEqual(all.results.slice(2, 4).map(r => r.id), source.filter(c => c.id.startsWith('alias-')).map(c => c.id));
    assert.deepEqual(all.results.slice(4).map(r => r.reason), ['prefix', 'token-prefix', 'typo']);
    assert.equal(new Set(all.results.map(r => r.id)).size, all.results.length);
    assert.ok(all.results.every(r => r.lexicalScore === undefined || Number.isFinite(r.lexicalScore) && r.lexicalScore >= 0 && r.lexicalScore <= 1));
    const limit = Math.floor(random() * 9);
    assert.deepEqual((await index.search(query, { limit })).results, all.results.slice(0, limit));
    index.dispose();
  }
});

test('one candidate cannot reuse a token; matching fields compete by subclass before score', async () => {
  const index = await createIndex([
    { id: 'once', label: 'Profile' }, { id: 'twice', label: 'Profile Profile' },
    { id: 'field', label: 'Your Profile', aliases: ['Profiles and configuration'] },
    { id: 'camel', label: 'Personal Settings' },
  ]);
  assert.deepEqual((await index.search('pro pro')).results.map(r => r.id), ['twice']);
  const multiField = (await index.search('profile')).results.find(r => r.id === 'field')!;
  assert.equal(multiField.reason, 'prefix'); assert.equal(multiField.matchedField, 'alias');
  assert.equal((await index.search('personalSettings')).results.find(r => r.id === 'camel')!.reason, 'token-prefix');
});

test('core normalization and tokenization match the Python-generated shared fixtures', () => {
  const fixtures = JSON.parse(readFileSync(new URL('../../../fixtures/features.json', import.meta.url), 'utf8')) as { text: string; normalized: string; tokens: string[] }[];
  for (const fixture of fixtures) {
    assert.equal(normalizeKey(fixture.text), fixture.normalized, JSON.stringify(fixture.text));
    assert.deepEqual(tokenize(fixture.text), fixture.tokens, JSON.stringify(fixture.text));
  }
});
