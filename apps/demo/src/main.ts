import { createIndex } from '../../../packages/core/src/index';
import './style.css';

type Candidate = { id: string; label: string; aliases?: readonly string[]; context?: string };
type Index = Awaited<ReturnType<typeof createIndex>>;
const initial: Candidate[] = [
  { id: 'profile', label: 'Profile', aliases: ['my information', 'personal details'], context: 'Personal settings' },
  { id: 'profiles', label: 'Profiles', context: 'Manage saved profiles' },
  { id: 'members', label: 'Members', aliases: ['coworkers', 'teammates'], context: 'Workspace settings' },
  { id: 'invoices', label: 'Invoices', aliases: ['receipts'], context: 'Billing' },
  { id: 'api-keys', label: 'API Keys', aliases: ['developer tokens'], context: 'Developer settings' },
  { id: 'plan', label: 'Plan', aliases: ['subscription'], context: 'Billing' },
  { id: 'billing', label: 'Billing', aliases: ['payment method'], context: 'Workspace settings' },
  { id: 'notifications', label: 'Notifications', aliases: ['alerts'], context: 'Personal settings' },
  { id: 'create', label: 'Create Account', context: 'Account management' },
  { id: 'delete', label: 'Delete Account', context: 'Account management' },
];

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div class="page">
    <header class="header">
      <a href="/" class="wordmark" aria-label="gpu-search home"><span class="logo" aria-hidden="true">⌕</span> gpu-search</a>
      <div class="header-links"><span class="version">EXPERIMENT / 01</span><a href="https://github.com/maxffarrell/gpu-search" id="repo-link">GitHub <span aria-hidden="true">↗</span></a></div>
    </header>
    <main>
      <section class="intro" aria-labelledby="title">
        <div class="eyebrow"><span class="status-dot"></span> LOCAL-FIRST INTERFACE SEARCH</div>
        <h1 id="title">Find the right<br />thing. <span>Locally.</span></h1>
        <p class="description">Predictable search for command palettes, settings, and menus. Exact matches first. Typos understood. Your queries stay in your browser.</p>
      </section>
      <section class="playground" aria-label="Interactive search playground">
        <div class="section-bar"><h2>Try it out</h2><span class="micro">NO SERVER. NO API KEY.</span></div>
        <div class="search-shell"><svg viewBox="0 0 24 24" fill="none" aria-hidden="true"><circle cx="10.8" cy="10.8" r="6.8"/><path d="m16 16 4.5 4.5"/></svg><label class="sr-only" for="query">Search the candidate menu</label><input id="query" type="search" placeholder="Search your interface…" autocomplete="off" spellcheck="false" maxlength="512" value="profle" /><kbd aria-hidden="true">/</kbd></div>
        <div class="examples"><span>Try a query</span><button data-query="profile">profile <small>exact</small></button><button data-query="profle">profle <small>typo</small></button><button data-query="coworkers">coworkers <small>alias</small></button><button data-query="xqzv">xqzv <small>no match</small></button></div>
        <div class="comparison">
          <section class="result-panel" aria-labelledby="alias-title"><div class="panel-header"><h3 id="alias-title">With your aliases</h3><span class="badge">LEXICAL + ALIASES</span></div><div id="alias-results" class="results"></div></section>
          <section class="result-panel" aria-labelledby="lexical-title"><div class="panel-header"><h3 id="lexical-title">Labels only</h3><span class="badge">LEXICAL</span></div><div id="lexical-results" class="results"></div></section>
        </div>
        <div class="run-meta" id="run-meta" role="status" aria-live="polite">Building local indexes…</div>
        <p class="baseline-note"><span aria-hidden="true">↳</span> This is the deterministic baseline. “Coworkers” finds “Members” through an explicit alias, not a learned model. Semantic search and WebGPU are not enabled in this demo.</p>
      </section>
      <section class="details-grid">
        <div class="menu-section"><div class="section-bar"><h2>Your candidate menu</h2><span id="candidate-count" class="micro"></span></div><div id="candidate-list" class="candidate-list"></div><details class="editor"><summary>Edit candidates <span aria-hidden="true">+</span></summary><label for="candidate-json">Candidate records (JSON)</label><textarea id="candidate-json" spellcheck="false" rows="12"></textarea><div class="editor-actions"><button id="apply" class="primary-button">Apply menu</button><button id="reset" class="text-button">Reset example</button></div><p id="editor-status" role="status" aria-live="polite"></p></details></div>
        <div class="code-section" id="api"><div class="section-bar"><h2>A small API. A clear result.</h2><span class="micro">TYPESCRIPT</span></div><pre><code><span class="code-keyword">const</span> index = <span class="code-keyword">await</span> createIndex([
  { id: <span class="code-string">'members'</span>, label: <span class="code-string">'Members'</span>,
    aliases: [<span class="code-string">'coworkers'</span>] },
], { semantic: <span class="code-keyword">false</span> });

<span class="code-keyword">await</span> index.search(<span class="code-string">'coworkers'</span>);
<span class="code-comment">// match: 'lexical'
// reason: 'alias-equality'
// matchedField: 'alias'</span></code></pre><p class="code-caption">Reusable immutable indexes. Explicit match reasons. No model download in this baseline.</p></div>
      </section>
    </main>
    <footer><div class="footer-top"><span class="wordmark">gpu-search</span><span>Built for the things you're looking for.</span></div><div class="credits"><span>Inspired by</span><a href="https://gpu-lexer.vercel.app" target="_blank" rel="noreferrer">gpu-lexer ↗</a><a href="https://gpu-time.arikko.dev" target="_blank" rel="noreferrer">gpu-time ↗</a><a href="https://gpu-cron.vercel.app" target="_blank" rel="noreferrer">gpu-cron ↗</a></div></footer>
  </div>`;

const query = document.querySelector<HTMLInputElement>('#query')!;
const editor = document.querySelector<HTMLTextAreaElement>('#candidate-json')!;
const status = document.querySelector<HTMLParagraphElement>('#editor-status')!;
let aliasIndex: Index | undefined;
let lexicalIndex: Index | undefined;
let generation = 0;
let menuGeneration = 0;

function drawResults(target: string, results: Awaited<ReturnType<Index['search']>>['results']) {
  const root = document.querySelector<HTMLDivElement>(target)!;
  root.replaceChildren();
  if (!results.length) {
    const empty = document.createElement('div');
    empty.className = 'empty';
    empty.textContent = query.value.trim() ? 'No matching destinations' : 'Type a query to find a destination';
    root.append(empty);
    return;
  }
  const list = document.createElement('ol');
  list.className = 'result-list';
  for (const [i, result] of results.entries()) {
    const li = document.createElement('li');
    const rank = document.createElement('span');
    rank.className = 'rank';
    rank.textContent = String(i + 1).padStart(2, '0');
    const body = document.createElement('div');
    body.className = 'result-body';
    const label = document.createElement('span');
    label.className = 'result-label';
    label.textContent = result.label;
    const reason = document.createElement('span');
    reason.className = 'reason';
    reason.textContent = `${result.reason} · ${result.matchedField}`;
    body.append(label, reason);
    const badge = document.createElement('span');
    badge.className = `match match-${result.match}`;
    badge.textContent = result.match;
    li.append(rank, body, badge);
    list.append(li);
  }
  root.append(list);
}

async function search() {
  if (!aliasIndex || !lexicalIndex) return;
  const current = ++generation;
  try {
    const [aliases, lexical] = await Promise.all([aliasIndex.search(query.value, { limit: 5 }), lexicalIndex.search(query.value, { limit: 5 })]);
    if (current !== generation) return;
    drawResults('#alias-results', aliases.results);
    drawResults('#lexical-results', lexical.results);
    document.querySelector('#run-meta')!.textContent = `Backend: ${aliases.backend} · ${aliases.results.length} alias-enriched / ${lexical.results.length} label-only results · ${aliases.degraded || lexical.degraded ? 'Degraded' : 'Ready'}`;
    document.querySelectorAll<HTMLButtonElement>('[data-query]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.query === query.value)));
  } catch (error) {
    if (current === generation) {
      document.querySelector('#alias-results')!.replaceChildren();
      document.querySelector('#lexical-results')!.replaceChildren();
      document.querySelector('#run-meta')!.textContent = error instanceof Error ? error.message : 'Search failed.';
    }
  }
}

async function setCandidates(candidates: Candidate[]) {
  const current = ++menuGeneration;
  const nextAlias = await createIndex(candidates, { semantic: false });
  let nextLexical: Index;
  try { nextLexical = await createIndex(candidates.map(({ aliases: _aliases, ...candidate }) => candidate), { semantic: false }); }
  catch (error) { nextAlias.dispose(); throw error; }
  if (current !== menuGeneration) { nextAlias.dispose(); nextLexical.dispose(); return; }
  generation++;
  aliasIndex?.dispose();
  lexicalIndex?.dispose();
  aliasIndex = nextAlias;
  lexicalIndex = nextLexical;
  const root = document.querySelector('#candidate-list')!;
  root.replaceChildren();
  for (const candidate of candidates) {
    const row = document.createElement('div');
    row.className = 'candidate';
    const label = document.createElement('span');
    label.textContent = candidate.label;
    const alias = document.createElement('span');
    alias.className = 'candidate-alias';
    alias.textContent = candidate.aliases?.join(', ') || '—';
    row.append(label, alias);
    root.append(row);
  }
  document.querySelector('#candidate-count')!.textContent = `${candidates.length} DESTINATIONS`;
  await search();
}

editor.value = JSON.stringify(initial, null, 2);
query.addEventListener('input', () => void search());
document.querySelectorAll<HTMLButtonElement>('[data-query]').forEach(button => button.addEventListener('click', () => { query.value = button.dataset.query!; void search(); }));
document.addEventListener('keydown', event => {
  if (event.key === '/' && !event.metaKey && !event.ctrlKey && !event.altKey && !(event.target instanceof HTMLInputElement) && !(event.target instanceof HTMLTextAreaElement)) { event.preventDefault(); query.focus(); }
});
document.querySelector('#apply')!.addEventListener('click', async () => {
  try {
    const candidates: unknown = JSON.parse(editor.value);
    if (!Array.isArray(candidates)) throw new Error('Provide a JSON array of candidate records.');
    await setCandidates(candidates as Candidate[]);
    status.textContent = 'Menu updated. Both indexes use the same labels.';
    status.className = 'success';
  } catch (error) { status.textContent = error instanceof Error ? error.message : 'Could not apply this menu.'; status.className = 'error'; }
});
document.querySelector('#reset')!.addEventListener('click', async () => { editor.value = JSON.stringify(initial, null, 2); await setCandidates(initial); status.textContent = 'Example menu restored.'; status.className = 'success'; });
void setCandidates(initial).catch(error => { document.querySelector('#run-meta')!.textContent = String(error); });
window.addEventListener('pagehide', event => { if (!event.persisted) { aliasIndex?.dispose(); lexicalIndex?.dispose(); } });
