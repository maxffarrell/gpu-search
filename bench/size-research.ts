/** Measure a complete research demo without changing the production import graph. */
import { build } from 'vite';
import { mkdtempSync, readdirSync, readFileSync, writeFileSync, rmSync } from 'node:fs';
import { join, relative } from 'node:path';
import { tmpdir } from 'node:os';
import { brotliCompressSync, constants } from 'node:zlib';
import { createHash } from 'node:crypto';

const directory = mkdtempSync(join(tmpdir(), 'gpu-search-research-size-'));
try {
  await build({
    root: 'apps/demo', configFile: 'apps/demo/vite.config.ts',
    build: { outDir: directory, emptyOutDir: true },
    plugins: [{
      name: 'isolated-bigram-research-build', enforce: 'pre',
      transform(code, id) {
        if (!id.endsWith('/apps/demo/src/main.ts')) return;
        return code.replace("import { loadModel } from '../../../packages/model/runtime';", `import { loadBigramModel } from '../../../packages/model/runtime-bigram';
async function loadModel(...args) {
  const model = await loadBigramModel(...args);
  return { ...model, prepare(candidates) {
    const snapshot = structuredClone(candidates);
    return { score: query => model.score(query, snapshot), dispose() {} };
  } };
}`)
          .replaceAll('packages/model/experiments/navigation-align0p5-seed29-word085/', 'packages/model/candidate-v2/');
      },
    }],
  });
  const walk = (path: string): string[] => readdirSync(path, { withFileTypes: true }).flatMap(item => item.isDirectory() ? walk(join(path, item.name)) : [join(path, item.name)]);
  const resources = walk(directory).map(path => {
    const bytes = readFileSync(path);
    return { resource: relative(directory, path), raw: bytes.length,
      brotli: brotliCompressSync(bytes, { params: { [constants.BROTLI_PARAM_QUALITY]: 11, [constants.BROTLI_PARAM_LGWIN]: 22 } }).length,
      sha256: createHash('sha256').update(bytes).digest('hex') };
  });
  const total = resources.reduce((sum, resource) => sum + resource.brotli, 0);
  const report = { scope: 'Local research build only; production demo and weights unchanged', candidate: 'packages/model/candidate-v2',
    compression: 'Brotli quality11 window22; each complete demo resource separately', resources,
    totalBrotliBytes: total, cpuLimitBytes: 50 * 1024, passesCpuBudget: total <= 50 * 1024, webgpuShipped: false };
  writeFileSync('eval/optimized-size.json', JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 2));
  const manifest = JSON.parse(readFileSync('packages/model/candidate-v2/manifest.json', 'utf8'));
  if (!resources.some(resource => resource.sha256 === manifest.payloadSha256)) throw new Error('Research build omitted the actual candidate weights');
  if (!report.passesCpuBudget) throw new Error('Research demo exceeds complete 50 KiB CPU budget');
} finally {
  rmSync(directory, { recursive: true, force: true });
}
