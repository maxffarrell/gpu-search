/** Deterministic, local retrieval. No assets or GPU are accessed at import time. */
export type Backend = 'cpu' | 'webgpu' | 'auto';
export type Candidate = { id: string; label: string; aliases?: readonly string[]; context?: string };
export type Match = 'exact' | 'lexical' | 'semantic';
export type SearchResult = {
  id: string; label: string; match: Match;
  reason: 'label-equality' | 'alias-equality' | 'prefix' | 'token-prefix' | 'acronym' | 'typo' | 'concept';
  matchedField: 'label' | 'alias' | 'semantic';
  lexicalScore?: number; semanticScore?: number;
};
export type SearchResponse = { results: SearchResult[]; backend: 'cpu' | 'webgpu' | 'lexical'; degraded: boolean };
export type SearchOptions = { limit?: number; signal?: AbortSignal };
export type IndexOptions = { backend?: Backend; semantic?: boolean };
export type SearchIndex = { search(query: string, options?: SearchOptions): Promise<SearchResponse>; dispose(): void };

/** Build constants, not model quality or latency guarantees. */
export const INPUT_LIMITS = Object.freeze({ candidates: 50_000, label: 256, query: 256, aliases: 8, alias: 128, context: 256 });
export class SearchInputError extends TypeError {
  readonly code = 'ERR_SEARCH_INPUT';
  constructor(message: string, readonly path: string) { super(`${path}: ${message}`); this.name = 'SearchInputError'; }
}
export class IndexDisposedError extends Error {
  readonly code = 'ERR_INDEX_DISPOSED';
  constructor() { super('Search index has been disposed'); this.name = 'IndexDisposedError'; }
}

const asciiLower = (text: string): string => text.replace(/[A-Z]/g, c => c.toLowerCase());
/** NFKC, ASCII case only, Unicode White_Space collapse. Punctuation is retained. */
export const normalizeKey = (text: string): string => asciiLower(text.normalize('NFKC')).replace(/\p{White_Space}+/gu, ' ').replace(/^ +| +$/g, '');
/** Token splitting is deliberately separate from strict equality. */
export function tokenize(text: string): string[] {
  return asciiLower(text.normalize('NFKC').replace(/([a-z0-9])([A-Z])/g, '$1 $2'))
    .split(/[\p{White_Space}_-]+/u).filter(Boolean);
}
const length = (text: string): number => Array.from(text).length;
const nonSpaceLength = (text: string): number => length(text.replace(/\p{White_Space}/gu, ''));
function validateText(value: unknown, path: string, max: number, nonempty = false): asserts value is string {
  if (typeof value !== 'string') throw new SearchInputError('expected a string', path);
  if (/[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/u.test(value)) throw new SearchInputError('unpaired surrogate is not a Unicode scalar', path);
  if (length(value) > max) throw new SearchInputError(`exceeds ${max} Unicode scalars`, path);
  if (nonempty && !normalizeKey(value)) throw new SearchInputError('must not be empty', path);
}
type Field = { key: string; tokens: string[]; kind: 'label' | 'alias' };
type Entry = { id: string; label: string; fields: Field[]; order: number };
type Ranked = { result: SearchResult; tier: number; subclass: number; score: number; order: number };
function field(text: string, kind: Field['kind']): Field { return { key: normalizeKey(text), tokens: tokenize(text), kind }; }

/** Optimal string alignment distance, including adjacent transpositions (not unrestricted DL). */
export function osaDistance(left: string, right: string): number {
  const a = Array.from(left), b = Array.from(right);
  let previous = Array.from({ length: b.length + 1 }, (_, i) => i);
  let beforePrevious = previous;
  for (let i = 1; i <= a.length; i++) {
    const current = new Array<number>(b.length + 1); current[0] = i;
    for (let j = 1; j <= b.length; j++) {
      current[j] = Math.min(previous[j]! + 1, current[j - 1]! + 1, previous[j - 1]! + (a[i - 1] === b[j - 1] ? 0 : 1));
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) current[j] = Math.min(current[j]!, beforePrevious[j - 2]! + 1);
    }
    beforePrevious = previous; previous = current;
  }
  return previous[b.length]!;
}
function qualifies(query: string, tokens: string[], candidate: Field): { subclass: number; score: number; reason: SearchResult['reason'] } | undefined {
  const qLength = length(query), fLength = length(candidate.key);
  if (candidate.kind === 'alias' && query === candidate.key) return { subclass: 5, score: 1, reason: 'alias-equality' };
  const prefixScore = Math.min(1, nonSpaceLength(query) / Math.max(1, nonSpaceLength(candidate.key)));
  if (qLength >= 2 && candidate.key.startsWith(query)) return { subclass: 4, score: prefixScore, reason: 'prefix' };
  if (nonSpaceLength(query) >= 3 && tokens.length > 0) {
    let next = 0;
    for (const token of candidate.tokens) if (next < tokens.length && token.startsWith(tokens[next]!)) next++;
    if (next === tokens.length) return { subclass: 3, score: prefixScore, reason: 'token-prefix' };
  }
  if (qLength >= 2 && qLength <= 6 && candidate.tokens.length >= 2 && query === candidate.tokens.map(t => Array.from(t)[0]).join('')) return { subclass: 2, score: 1, reason: 'acronym' };
  const longest = Math.max(qLength, fLength), maxDistance = longest >= 8 ? 2 : 1;
  if (qLength >= 4 && fLength >= 4 && Math.abs(qLength - fLength) <= maxDistance) {
    const distance = osaDistance(query, candidate.key);
    if (distance <= maxDistance && distance / longest <= 0.25) return { subclass: 1, score: 1 - distance / longest, reason: 'typo' };
  }
  return undefined;
}
function compare(a: Ranked, b: Ranked): number { return b.tier - a.tier || b.subclass - a.subclass || b.score - a.score || a.order - b.order; }

export async function createIndex(candidates: readonly Candidate[], options: IndexOptions = {}): Promise<SearchIndex> {
  if (!Array.isArray(candidates)) throw new SearchInputError('expected an array', 'candidates');
  if (candidates.length > INPUT_LIMITS.candidates) throw new SearchInputError('too many candidates', 'candidates');
  if (!options || typeof options !== 'object' || Array.isArray(options)) throw new SearchInputError('expected an options object', 'options');
  if (options.backend !== undefined && !['auto', 'cpu', 'webgpu'].includes(options.backend)) throw new SearchInputError('expected auto, cpu, or webgpu', 'backend');
  if (options.semantic !== undefined && typeof options.semantic !== 'boolean') throw new SearchInputError('expected a boolean', 'semantic');
  const degraded = options.semantic !== false;
  const ids = new Set<string>();
  let entries: Entry[] = [];
  // A plain loop also validates sparse array holes.
  for (let order = 0; order < candidates.length; order++) {
    const item = candidates[order], path = `candidates[${order}]`;
    if (!item || typeof item !== 'object' || Array.isArray(item)) throw new SearchInputError('expected a candidate record', path);
    validateText(item.id, `${path}.id`, Number.MAX_SAFE_INTEGER, true);
    validateText(item.label, `${path}.label`, INPUT_LIMITS.label, true);
    if (ids.has(item.id)) throw new SearchInputError('duplicate ID', `${path}.id`);
    ids.add(item.id);
    const fields = [field(item.label, 'label')];
    if (item.aliases !== undefined) {
      if (!Array.isArray(item.aliases) || item.aliases.length > INPUT_LIMITS.aliases) throw new SearchInputError('expected at most 8 aliases', `${path}.aliases`);
      for (let i = 0; i < item.aliases.length; i++) {
        const alias = item.aliases[i]; validateText(alias, `${path}.aliases[${i}]`, INPUT_LIMITS.alias, true); fields.push(field(alias, 'alias'));
      }
    }
    if (item.context !== undefined) validateText(item.context, `${path}.context`, INPUT_LIMITS.context);
    entries.push({ id: item.id, label: item.label, fields, order });
  }
  let disposed = false;
  return {
    async search(query, searchOptions = {}) {
      if (disposed) throw new IndexDisposedError();
      validateText(query, 'query', INPUT_LIMITS.query);
      if (!searchOptions || typeof searchOptions !== 'object' || Array.isArray(searchOptions)) throw new SearchInputError('expected an options object', 'searchOptions');
      const { limit = 10, signal } = searchOptions;
      if (!Number.isSafeInteger(limit) || limit < 0) throw new SearchInputError('expected a nonnegative safe integer', 'limit');
      const check = () => { if (disposed) throw new IndexDisposedError(); if (signal?.aborted) throw signal.reason ?? new DOMException('Search aborted', 'AbortError'); };
      check();
      const key = normalizeKey(query), tokens = tokenize(query), ranked: Ranked[] = [];
      if (key && limit) for (const entry of entries) {
        if (entry.fields[0]!.key === key) {
          ranked.push({ result: { id: entry.id, label: entry.label, match: 'exact', reason: 'label-equality', matchedField: 'label' }, tier: 3, subclass: 0, score: 1, order: entry.order });
          continue;
        }
        let best: Ranked | undefined;
        for (const candidateField of entry.fields) {
          const found = qualifies(key, tokens, candidateField);
          if (!found) continue;
          const item: Ranked = { result: { id: entry.id, label: entry.label, match: 'lexical', reason: found.reason, matchedField: candidateField.kind, lexicalScore: found.score }, tier: 2, subclass: found.subclass, score: found.score, order: entry.order };
          if (!best || compare(item, best) < 0) best = item;
        }
        if (best) ranked.push(best);
      }
      ranked.sort(compare);
      // Yield before delivery so immediate abort/dispose cannot deliver stale work.
      await Promise.resolve(); check();
      return { results: ranked.slice(0, limit).map(item => item.result), backend: 'lexical', degraded };
    },
    dispose() { disposed = true; entries = []; },
  };
}
