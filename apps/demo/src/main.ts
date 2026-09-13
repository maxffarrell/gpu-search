import { createIndex } from '../../../packages/core/src/index';
import { loadModel } from '../../../packages/model/runtime';
import manifestUrl from '../../../packages/model/candidate/manifest.json?url';
import weightsUrl from '../../../packages/model/candidate/weights.bin?url';
import './style.css';

type Candidate = { id: string; label: string; aliases?: readonly string[]; context?: string };
type Index = Awaited<ReturnType<typeof createIndex>>;
type Model = Awaited<ReturnType<typeof loadModel>>;
const initial: Candidate[] = [
  { id: 'profile', label: 'Profile' },
  { id: 'profiles', label: 'Profiles' },
  { id: 'members', label: 'Members' },
  { id: 'invoices', label: 'Invoices' },
  { id: 'api-keys', label: 'API Keys' },
  { id: 'plan', label: 'Plan' },
  { id: 'billing', label: 'Billing' },
  { id: 'notifications', label: 'Notifications' },
  { id: 'create', label: 'Create Account' },
  { id: 'delete', label: 'Delete Account' },
];

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div class="page">
    <header><a class="title" href="/">gpu-search</a><nav aria-label="Project and appearance"><a id="repo-link" href="https://github.com/maxffarrell/gpu-search" target="_blank" rel="noreferrer">GitHub ↗</a><div class="theme-control" role="group" aria-label="Color theme"><button data-theme="auto">Auto</button><button data-theme="light">Light</button><button data-theme="dark">Dark</button></div></nav></header>
    <main>
      <p class="intro">Small, local search for software interfaces.</p>
      <label class="sr-only" for="query">Search candidates</label><div class="search"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="10.8" cy="10.8" r="6.8"/><path d="m16 16 4.5 4.5"/></svg><input id="query" type="search" value="coworkers" placeholder="Search…" autocomplete="off" spellcheck="false" /><kbd aria-hidden="true">/</kbd></div>
      <div class="examples"><span>Try</span><button data-query="profle">profle</button><button data-query="coworkers">coworkers</button><button data-query="my information">my information</button></div>
      <div class="comparison"><section aria-labelledby="model-title"><h2 id="model-title">Model scores <span>CPU · cosine</span></h2><div id="model-results" class="results"></div></section><section aria-labelledby="lexical-title"><h2 id="lexical-title">Lexical <span>deterministic</span></h2><div id="lexical-results" class="results"></div></section></div>
      <p id="run-meta" class="run-meta" role="status" aria-live="polite">Loading model…</p>
      <p class="note">Experimental model, trained on public intent datasets and software settings. Generalization is still limited. Raw cosine scores are not confidence; no relevance cutoff is applied.</p>
      <details class="editor"><summary>Candidate menu <span id="candidate-count"></span></summary><label for="candidate-json">Edit candidate records as JSON. Both comparisons use the same records.</label><textarea id="candidate-json" spellcheck="false" rows="13"></textarea><div class="actions"><button id="apply">Apply menu</button><button id="reset">Reset</button></div><p id="editor-status" role="status" aria-live="polite"></p></details>
      <details class="model-details"><summary>Model details</summary><p id="model-details">Loading model assets…</p><p>Inference runs locally in your browser. The initial menu contains labels only, without aliases or context. Training sources: CLINC150 (CC BY 3.0), BANKING77 (CC BY 4.0), and VS Code settings (MIT). See GitHub for attribution and evaluation. The model column shows the five highest raw scores; the lexical column applies exact and fuzzy matching rules.</p></details>
    </main>
    <footer><span>Inspired by</span><a href="https://gpu-lexer.vercel.app" target="_blank" rel="noreferrer">gpu-lexer ↗</a><a href="https://gpu-time.arikko.dev" target="_blank" rel="noreferrer">gpu-time ↗</a><a href="https://gpu-cron.vercel.app" target="_blank" rel="noreferrer">gpu-cron ↗</a></footer>
  </div>`;

const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
let theme: 'auto' | 'light' | 'dark' = 'auto';
try { const saved = localStorage.getItem('gpu-search-theme'); if (saved === 'light' || saved === 'dark') theme = saved; } catch { /* Storage may be disabled. */ }
function applyTheme() {
  document.documentElement.dataset.theme = theme === 'auto' ? (systemTheme.matches ? 'dark' : 'light') : theme;
  document.querySelectorAll<HTMLButtonElement>('button[data-theme]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.theme === theme)));
}
systemTheme.addEventListener('change', applyTheme);
document.querySelectorAll<HTMLButtonElement>('button[data-theme]').forEach(button => button.addEventListener('click', () => { theme = button.dataset.theme as typeof theme; try { localStorage.setItem('gpu-search-theme', theme); } catch { /* Storage may be disabled. */ } applyTheme(); }));
applyTheme();

const query = document.querySelector<HTMLInputElement>('#query')!;
const editor = document.querySelector<HTMLTextAreaElement>('#candidate-json')!;
const editorStatus = document.querySelector<HTMLParagraphElement>('#editor-status')!;
const runMeta = document.querySelector<HTMLParagraphElement>('#run-meta')!;
let index: Index | undefined;
let model: Model | undefined;
let modelState: 'loading' | 'ready' | 'error' = 'loading';
let candidates: Candidate[] = [];
let generation = 0;
let menuGeneration = 0;

function empty(target: string, message: string) {
  const root = document.querySelector(target)!;
  const text = document.createElement('p');
  text.className = 'empty';
  text.textContent = message;
  root.replaceChildren(text);
}
function draw(target: string, rows: { id: string; label: string; detail: string; score?: number }[]) {
  if (!rows.length) { empty(target, query.value.trim() ? 'No matches' : 'Enter a query'); return; }
  const list = document.createElement('ol');
  for (const row of rows) {
    const item = document.createElement('li');
    item.dataset.id = row.id;
    if (row.score !== undefined) item.dataset.score = String(row.score);
    const label = document.createElement('span');
    label.className = 'result-label';
    label.textContent = row.label;
    const detail = document.createElement('span');
    detail.className = 'result-detail';
    detail.textContent = row.detail;
    item.append(label, detail);
    list.append(item);
  }
  document.querySelector(target)!.replaceChildren(list);
}
async function search() {
  if (!index) return;
  const current = ++generation;
  const text = query.value;
  try {
    // The lexical index validates the query before either result column is rendered.
    const response = await index.search(text, { limit: 5 });
    if (current !== generation) return;
    draw('#lexical-results', response.results.map(result => ({ id: result.id, label: result.label, detail: result.reason })));
    if (!text.trim()) empty('#model-results', 'Enter a query');
    else if (model) {
      const scores = model.score(text, candidates).slice(0, 5);
      draw('#model-results', scores.map(result => ({ id: result.id, label: result.label, detail: result.score.toFixed(3), score: result.score })));
    } else empty('#model-results', modelState === 'error' ? 'Model unavailable' : 'Loading model…');
    runMeta.textContent = modelState === 'ready' ? 'Model: CPU · Lexical: CPU · Queries stay on this device' : modelState === 'error' ? 'Model unavailable · Lexical: CPU' : 'Loading model · Lexical: CPU';
    document.querySelectorAll<HTMLButtonElement>('[data-query]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.query === text)));
  } catch (error) {
    if (current !== generation) return;
    empty('#model-results', 'Unable to search');
    empty('#lexical-results', 'Unable to search');
    runMeta.textContent = error instanceof Error ? error.message : 'Search failed.';
  }
}
async function setCandidates(records: Candidate[]) {
  const current = ++menuGeneration;
  const next = await createIndex(records, { semantic: false });
  if (current !== menuGeneration) { next.dispose(); return; }
  // Construction validates every record. Retain a separate snapshot for model inference.
  candidates = records.map(record => ({ ...record, ...(record.aliases ? { aliases: [...record.aliases] } : {}) }));
  generation++;
  index?.dispose();
  index = next;
  document.querySelector('#candidate-count')!.textContent = `${candidates.length} labels`;
  await search();
}
editor.value = JSON.stringify(initial, null, 2);
query.addEventListener('input', () => void search());
document.querySelectorAll<HTMLButtonElement>('[data-query]').forEach(button => button.addEventListener('click', () => { query.value = button.dataset.query!; void search(); }));
document.addEventListener('keydown', event => { if (event.key === '/' && !event.metaKey && !event.ctrlKey && !event.altKey && !(event.target instanceof HTMLInputElement) && !(event.target instanceof HTMLTextAreaElement)) { event.preventDefault(); query.focus(); } });
document.querySelector('#apply')!.addEventListener('click', async () => {
  try {
    const records: unknown = JSON.parse(editor.value);
    if (!Array.isArray(records)) throw new Error('Provide a JSON array of candidate records.');
    await setCandidates(records as Candidate[]);
    editorStatus.textContent = 'Menu updated.';
    editorStatus.className = '';
  } catch (error) { editorStatus.textContent = error instanceof Error ? error.message : 'Invalid menu.'; editorStatus.className = 'error'; }
});
document.querySelector('#reset')!.addEventListener('click', async () => { editor.value = JSON.stringify(initial, null, 2); await setCandidates(initial); editorStatus.textContent = 'Example restored.'; editorStatus.className = ''; });
void setCandidates(initial).catch(error => { runMeta.textContent = String(error); });
async function fetchModel() {
  const [manifestResponse, weightsResponse] = await Promise.all([fetch(manifestUrl), fetch(weightsUrl)]);
  if (!manifestResponse.ok || !weightsResponse.ok) throw new Error('Could not load model assets.');
  const [manifest, weights] = await Promise.all([manifestResponse.json(), weightsResponse.arrayBuffer()]);
  return loadModel(manifest, weights);
}
void fetchModel().then(loaded => {
  model = loaded;
  modelState = 'ready';
  document.querySelector('#model-details')!.textContent = `Model: ${loaded.id} · 32 KiB weights · CPU inference · SHA-256: ${loaded.hash}`;
  void search();
}).catch(error => {
  modelState = 'error';
  document.querySelector('#model-details')!.textContent = `Model unavailable: ${error instanceof Error ? error.message : String(error)}`;
  void search();
});
window.addEventListener('pagehide', event => { if (!event.persisted) index?.dispose(); });
