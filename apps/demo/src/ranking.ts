import type { ModelScore } from '../../../packages/model/runtime';

type TextMatch = { id: string; label: string; reason: string };
export type DisplayResult = { id: string; label: string; detail: string };

/** Text matches take precedence. Only fall back to explicitly labelled learned suggestions. */
export function searchResults(matches: readonly TextMatch[], suggestions: readonly ModelScore[], limit = 5): DisplayResult[] {
  if (matches.length) return matches.slice(0, limit).map(row => ({ id: row.id, label: row.label, detail: row.reason }));
  return suggestions.slice(0, limit).map(row => ({ id: row.id, label: row.label, detail: 'model suggestion' }));
}
