import { createIndex as createBaseIndex, type Candidate, type SearchIndex } from './index.js';
export * from './index.js';
/** Explicit lexical-only entry. No missing-model degradation is reported. */
export function createIndex(candidates: readonly Candidate[]): Promise<SearchIndex> {
  return createBaseIndex(candidates, { semantic: false });
}
