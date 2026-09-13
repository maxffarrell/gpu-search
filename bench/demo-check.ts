import { chromium } from 'playwright';
import { preview } from 'vite';
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { loadModel } from '../packages/model/runtime.js';

const manifest = JSON.parse(readFileSync('packages/model/candidate/manifest.json', 'utf8'));
const bytes = readFileSync('packages/model/candidate/weights.bin');
const model = await loadModel(manifest, Uint8Array.from(bytes).buffer);
const server = await preview({ root: 'apps/demo', preview: { host: '127.0.0.1', port: 4180, strictPort: true } });
const browser = await chromium.launch({ headless: true, ...(process.env.CHROME_CHANNEL ? { channel: process.env.CHROME_CHANNEL } : {}) });
const url = 'http://127.0.0.1:4180';
try {
  const page = await browser.newPage();
  const errors: string[] = [], requests: string[] = [], assets: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('response', r => { if (/\.(bin|json)(?:\?|$)/.test(r.url())) assets.push(r.url()); });
  await page.goto(url);
  await page.locator('#model-results [data-score]').first().waitFor();
  assert.ok(assets.some(x => x.endsWith('.bin')), 'actual weights were fetched');
  assert.ok(assets.some(x => x.endsWith('.json')), 'actual manifest was fetched');
  page.on('request', r => requests.push(r.url()));
  const candidates = JSON.parse(await page.locator('#candidate-json').inputValue());
  // Seen-training sanity regressions, not held-out generalization evidence.
  assert.equal(model.score('coworkers', candidates)[0]?.id, 'members');
  assert.equal(model.score('my information', candidates)[0]?.id, 'profile');
  assert.ok(candidates.every((c: { aliases?: unknown; context?: unknown }) => !c.aliases && !c.context), 'initial candidates contain no hidden aliases/context');
  const samples = [];
  for (const query of ['profile', 'profle', 'coworkers', 'my information', 'xqzv', 'unseen destination 47']) {
    await page.locator('#query').fill(query);
    const expected = model.score(query, candidates).slice(0, 5);
    await page.waitForFunction(({ expected }) => {
      const rows = Array.from(document.querySelectorAll<HTMLElement>('#model-results [data-score]'));
      return rows.length === expected.length && rows.every((r, i) => r.dataset.id === expected[i].id && Math.abs(Number(r.dataset.score) - expected[i].score) < 2e-4);
    }, { expected });
    samples.push({ query, expected });
  }
  await page.locator('#query').fill('');
  await page.waitForFunction(() => document.querySelectorAll('#model-results [data-score]').length === 0);
  await page.locator('#query').fill('profile');
  await page.waitForFunction(() => document.querySelector('#lexical-results')?.textContent?.includes('label-equality'));

  // The explicit preference persists; Auto reacts to an OS change without reloading.
  await page.locator('button[data-theme="dark"]').click();
  assert.equal(await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme), 'dark');
  await page.reload();
  await page.locator('#model-results [data-score]').first().waitFor();
  assert.equal(await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme), 'dark');
  await page.locator('button[data-theme="light"]').click();
  assert.equal(await page.evaluate(() => getComputedStyle(document.documentElement).colorScheme), 'light');
  await page.locator('button[data-theme="auto"]').click();
  await page.emulateMedia({ colorScheme: 'dark' });
  await page.waitForFunction(() => getComputedStyle(document.documentElement).colorScheme === 'dark');
  await page.emulateMedia({ colorScheme: 'light' });
  await page.waitForFunction(() => getComputedStyle(document.documentElement).colorScheme === 'light');
  // Reload is expected to fetch assets; query/edit interactions below must not.
  requests.length = 0;
  await page.locator('.editor summary').click();
  await page.locator('#candidate-json').fill('invalid json');
  await page.locator('#apply').click();
  assert.ok((await page.locator('#editor-status').innerText()).length > 0);
  const custom = [{ id: 'new', label: 'A completely new destination' }, { id: 'safe', label: '<img src=x onerror=alert(1)>' }];
  await page.locator('#candidate-json').fill(JSON.stringify(custom));
  await page.locator('#apply').click();
  await page.locator('#query').fill('new destination');
  const customExpected = model.score('new destination', custom);
  await page.waitForFunction(({ expected }) => {
    const rows = Array.from(document.querySelectorAll<HTMLElement>('#model-results [data-score]'));
    return rows.length === expected.length && rows.every((r, i) => r.dataset.id === expected[i].id && Math.abs(Number(r.dataset.score) - expected[i].score) < 2e-4);
  }, { expected: customExpected });
  assert.equal(await page.locator('#model-results img, #lexical-results img').count(), 0);
  await page.locator('#reset').click();
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
  mkdirSync('bench/screenshots', { recursive: true });
  await page.screenshot({ path: 'bench/screenshots/mobile-light.png', fullPage: true });
  await page.locator('button[data-theme="dark"]').click();
  await page.screenshot({ path: 'bench/screenshots/mobile-dark.png', fullPage: true });
  assert.deepEqual(requests, [], 'queries and candidate edits make no network requests');
  assert.deepEqual(errors, []);

  const missing = await browser.newPage();
  await missing.route('**/*.bin', r => r.abort());
  await missing.goto(url);
  await missing.getByText(/Model unavailable/).first().waitFor();
  assert.equal(await missing.locator('#model-results [data-score]').count(), 0, 'missing weights never display substituted scores');
  await missing.close();

  // A causal check: valid all-zero weights eliminate every model result.
  const zero = new Uint8Array(bytes.length);
  const zeroManifest = { ...manifest, payloadSha256: createHash('sha256').update(zero).digest('hex') };
  const zeroPage = await browser.newPage();
  await zeroPage.route('**/*.bin', r => r.fulfill({ body: Buffer.from(zero), contentType: 'application/octet-stream' }));
  await zeroPage.route('**/*.json', r => r.fulfill({ json: zeroManifest }));
  await zeroPage.goto(url);
  await zeroPage.locator('#query').fill('profle');
  await zeroPage.waitForFunction(() => document.querySelector('#run-meta')?.textContent?.startsWith('Model: CPU'));
  assert.equal(await zeroPage.locator('#model-results [data-score]').count(), 0, 'zero weights cause zero model results');
  assert.ok((await zeroPage.locator('#lexical-results').innerText()).includes('Profile'), 'lexical baseline still operates independently');
  await zeroPage.close();

  mkdirSync('bench/reports', { recursive: true });
  writeFileSync('bench/reports/demo.json', JSON.stringify({ generatedAt: new Date().toISOString(), browser: browser.version(), modelId: model.id, modelSha256: model.hash, assets, checks: ['model asset fetch', 'CPU scores equal exported-weight inference for six queries', 'arbitrary edited candidate inference', 'missing weights show no model results', 'zero weights eliminate model results', 'Auto/Light/Dark and persistence', 'invalid JSON', 'safe label text', '390px containment', 'zero query/edit requests'], samples, customExpected, requests, errors }, null, 2) + '\n');
  console.log('Real model inference, theme, privacy and failure checks passed');
} finally { await browser.close(); await new Promise<void>(resolve => server.httpServer.close(() => resolve())); }
