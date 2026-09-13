import { chromium, webkit } from 'playwright';
import { createServer } from 'vite';
import { cpus, platform, release } from 'node:os';
import { mkdirSync, writeFileSync } from 'node:fs';

const externalUrl = process.env.MODEL_BENCH_URL;
const server = externalUrl ? undefined : await createServer({ root: '.', server: { host: '127.0.0.1', port: 4187, strictPort: true } });
if (server) await server.listen();
const baseUrl = externalUrl ?? 'http://127.0.0.1:4187';
const reports = [];
try {
  for (const [name, engine] of [['chromium', chromium], ['webkit', webkit]] as const) {
    const browser = await engine.launch({ headless: true, ...(name === 'chromium' && process.env.CHROME_CHANNEL ? { channel: process.env.CHROME_CHANNEL } : {}), ...(name === 'webkit' && process.env.WEBKIT_EXECUTABLE ? { executablePath: process.env.WEBKIT_EXECUTABLE } : {}) });
    try {
      const page = await browser.newPage();
      const errors: string[] = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
      page.on('response', response => { if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`); });
      // The standalone text harness has no favicon; serve that optional browser request explicitly.
      await page.route('**/favicon.ico', route => route.fulfill({ status: 204, body: '' }));
      await page.goto(`${baseUrl}/bench/harness.html`);
      await page.evaluate('globalThis.__name = (target) => target');
      const data = await page.evaluate(async () => {
        const { loadModel } = await import('/packages/model/runtime.ts' as string);
        const start = performance.now();
        const [manifestResponse, weightsResponse] = await Promise.all([fetch('/packages/model/experiments/navigation-align0p5-seed29/manifest.json'), fetch('/packages/model/experiments/navigation-align0p5-seed29/weights.bin')]);
        if (!manifestResponse.ok || !weightsResponse.ok) throw new Error('Model assets failed to load');
        const [manifest, bytes] = await Promise.all([manifestResponse.json(), weightsResponse.arrayBuffer()]);
        const assetLoadMs = performance.now() - start;
        const loading = performance.now();
        const model = await loadModel(manifest, bytes);
        const verifyAndDecodeMs = performance.now() - loading;
        const queries = ['coworkers', 'change my password', 'personal information', 'download invoices', 'developer credentials'];
        const labels = ['Project Settings', 'Team Members', 'Billing Details', 'API Keys', 'Personal Profile', 'Security Settings', 'Notifications', 'Account Access'];
        const rows = [];
        let seed = 20260913;
        const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
        for (const count of [100, 1000, 5000]) {
          const candidates = Array.from({ length: count }, (_, i) => ({ id: String(i), label: `${labels[i % labels.length]} ${i}`, aliases: [`${labels[(i + 1) % labels.length]} ${i}`], context: i % 2 ? 'Personal settings' : 'Organization settings' }));
          const prep = performance.now();
          const index = model.prepare(candidates);
          const preparationMs = performance.now() - prep;
          const first = performance.now();
          const firstResult = index.score(queries[0]);
          const firstQueryMs = performance.now() - first;
          const reference = model.score(queries[0], candidates);
          if (JSON.stringify(firstResult) !== JSON.stringify(reference)) throw new Error(`Cached/reference score mismatch at ${count}`);
          for (let i = 0; i < 30; i++) index.score(queries[Math.floor(random() * queries.length)]);
          const resourcesBefore = performance.getEntriesByType('resource').length;
          const samples = [];
          for (let i = 0; i < 200; i++) {
            const query = queries[Math.floor(random() * queries.length)];
            const started = performance.now(); index.score(query); samples.push(performance.now() - started);
          }
          const requestsDuringQueries = performance.getEntriesByType('resource').length - resourcesBefore;
          const sorted = [...samples].sort((a, b) => a - b);
          let uncachedReference = null;
          if (count === 1000) {
            for (let i = 0; i < 30; i++) model.score(queries[Math.floor(random() * queries.length)], candidates);
            const uncachedSamples = [];
            const beforeUncached = performance.getEntriesByType('resource').length;
            for (let i = 0; i < 200; i++) {
              const query = queries[Math.floor(random() * queries.length)];
              const began = performance.now(); model.score(query, candidates); uncachedSamples.push(performance.now() - began);
            }
            const ordered = [...uncachedSamples].sort((a, b) => a - b);
            uncachedReference = { warmups: 30, sampleCount: 200, p50Ms: ordered[99], p95Ms: ordered[189], samples: uncachedSamples, requestsDuringQueries: performance.getEntriesByType('resource').length - beforeUncached };
          }
          const targetP95Ms = count === 1000 ? 8 : count === 5000 ? 16 : null;
          rows.push({ count, preparationMs, firstQueryMs, p50Ms: sorted[99], p95Ms: sorted[189], samples, uncachedReference, requestsDuringQueries, cachedVectorBytes: count * 16 * 4, targetP95Ms, meetsWarmTarget: targetP95Ms === null || sorted[189]! <= targetP95Ms, referenceParity: true });
          index.dispose();
          let rejected = false; try { index.score('profile'); } catch { rejected = true; }
          if (!rejected) throw new Error('Disposed prepared index still usable');
        }
        return { modelId: model.id, modelSha256: model.hash, dimension: manifest.dimension, backend: 'cpu', weightBytes: bytes.byteLength, assetLoadMs, verifyAndDecodeMs, rows, userAgent: navigator.userAgent, heapBytes: (performance as unknown as { memory?: { usedJSHeapSize: number } }).memory?.usedJSHeapSize ?? null };
      });
      reports.push({ browser: name, browserVersion: browser.version(), errors, ...data });
    } finally { await browser.close(); }
  }
} finally { await server?.close(); }
const report = { generatedAt: new Date().toISOString(), hardware: cpus()[0]?.model, os: `${platform()} ${release()}`, powerState: 'Uncontrolled interactive session; other training processes may be active', method: 'Real trained pooled16 int8 CPU model. Prepared vectors once per immutable menu; query encoding, all candidate dots, full sort, result materialization included. No lexical scoring/cutoff/GPU; not a full hybrid search benchmark. 30 warmups and 200 seeded randomized measured queries per arm; nearest-rank percentiles. At1000 candidates, uncached model.score is measured sequentially in the SAME browser after prepared scoring, also30warmups/200samples. Synthetic varied labels with one alias and context each. Browser timer resolution can round short calls to zero.', browserOverrides: { chromiumChannel: process.env.CHROME_CHANNEL ?? null, webkitExecutable: process.env.WEBKIT_EXECUTABLE ?? null }, reports };
mkdirSync('bench/reports', { recursive: true });
writeFileSync('bench/reports/model-browser.json', JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ ...report, reports: reports.map(r => ({ ...r, rows: r.rows.map(({ samples, ...row }) => row) })) }, null, 2));
if (reports.some(r => r.errors.length || r.rows.some(row => !row.meetsWarmTarget || row.requestsDuringQueries > 0 || (row.uncachedReference?.requestsDuringQueries ?? 0) > 0))) process.exitCode = 1;
