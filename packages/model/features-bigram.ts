import { features, hash, tokenize } from './features.js';

const utf8 = new TextEncoder();
export const BIGRAM_FEATURE_VERSION = 'gpu-search-features-bigram25-v1';

/** Ordered adjacent tokens share the word table; boundaries are length-prefixed. */
export function bigramFeatures(text: string) {
  const base = features(text);
  const tokens = tokenize(text).slice(0, 32).map(token => Array.from(token).slice(0, 64).join(''));
  const bigramIds: number[] = [];
  for (let i = 1; i < tokens.length; i++) {
    const left = utf8.encode(tokens[i - 1]), right = utf8.encode(tokens[i]);
    const bytes = new Uint8Array(10 + left.length + right.length);
    const view = new DataView(bytes.buffer);
    bytes.set([98, 58]);
    view.setUint32(2, left.length, true);
    bytes.set(left, 6);
    view.setUint32(6 + left.length, right.length, true);
    bytes.set(right, 10 + left.length);
    bigramIds.push(hash(bytes));
  }
  return { ...base, bigramIds, bigramCount: bigramIds.length };
}
