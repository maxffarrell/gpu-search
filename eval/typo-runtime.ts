/** Serving-runtime parity; reserved rows require the existing exact frozen evaluation receipt. */
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { loadModel } from '../packages/model/runtime.js';
const held = process.argv.includes('--holdout');
const artifacts = process.argv.slice(2).filter(arg => arg !== '--holdout');
if (held) {
  const receipt = JSON.parse(readFileSync('eval/typo-holdout-access.json', 'utf8'));
  if (artifacts.length !== 2 || artifacts[0] !== 'packages/model/experiments/navigation-align0p5-seed29' || artifacts[1] !== receipt.candidate.directory) throw new Error('Only the frozen candidate and baseline may be verified');
  const hash = createHash('sha256').update(readFileSync(`${artifacts[1]}/manifest.json`)).digest('hex');
  if (hash !== receipt.candidate.manifestSha256) throw new Error('Frozen manifest mismatch');
}
const rows = readFileSync(`data/typo/${held ? 'holdout' : 'dev'}.jsonl`, 'utf8').trim().split('\n').map(line => JSON.parse(line));
const output = [];
for (const artifact of artifacts) {
  const model = await loadModel(JSON.parse(readFileSync(`${artifact}/manifest.json`, 'utf8')), Uint8Array.from(readFileSync(`${artifact}/weights.bin`)).buffer);
  let hits = 0, shortHits = 0, shortCount = 0;
  const perQuery = [];
  for (const row of rows) {
    const result = model.score(row.query, row.candidates)[0];
    const hit = result && row.relevance[result.id] >= 2 ? 1 : 0;
    hits += hit;
    perQuery.push({id:row.id,cleanLabel:row.cleanLabel,query:row.query,hit,top1:result?.id});
    if (row.cleanLabel.split(' ').length <= 2) { shortCount++; shortHits += hit; }
  }
  output.push({ artifact, model: model.id, top1: hits / rows.length, shortTop1: shortHits / shortCount, selectionScore: .5 * (hits / rows.length + shortHits / shortCount), queries: rows.length, shortCount, perQuery });
}
writeFileSync(`eval/typo-runtime-${held ? 'holdout' : 'dev'}.json`, JSON.stringify({ scope: held ? 'Serving-runtime verification of the same frozen evaluation event; not another independent test.' : 'Actual TypeScript runtime on synthetic development rows; no lexical matching or holdout access.', results: output }, null, 2) + '\n');
console.log(output.map(({perQuery,...summary}) => summary));
