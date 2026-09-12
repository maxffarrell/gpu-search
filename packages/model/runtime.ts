import { features } from './features.js';
import type { Candidate } from '../core/src/index.js';

export type Model = {
  readonly id: string;
  readonly hash: string;
  encode(text: string): Float32Array;
  /** Raw cosine inspection: all nonzero candidate vectors, without a quality cutoff. */
  score(query: string, candidates: readonly Candidate[]): { id: string; label: string; score: number }[];
};
export class ModelAssetError extends Error {
  constructor(message: string) { super(message); this.name = 'ModelAssetError'; }
}
const DIMENSION = 16, ELEMENTS = 1024 * DIMENSION;
function requireAsset(condition: unknown, message: string): asserts condition {
  if (!condition) throw new ModelAssetError(message);
}
function record(value: unknown): Record<string, unknown> {
  requireAsset(value !== null && typeof value === 'object' && !Array.isArray(value), 'Expected manifest object');
  return value as Record<string, unknown>;
}
function normalize(vector: Float32Array): Float32Array {
  let squared = 0;
  for (const value of vector) squared = Math.fround(squared + Math.fround(value * value));
  const norm = Math.fround(Math.sqrt(squared));
  if (!Number.isFinite(norm) || norm < 1e-8) return new Float32Array(DIMENSION);
  return vector.map(value => Math.fround(value / norm));
}
function valid(vector: Float32Array): boolean { return vector.some(value => value !== 0); }

/** Loads the actual experimental trained pooled encoder. No model quality approval is implied. */
export async function loadModel(input: unknown, payload: ArrayBuffer): Promise<Model> {
  const manifest = record(input);
  const versions = {
    formatVersion: 'gpu-search-experimental-v1', featureVersion: 'gpu-search-features-v1',
    normalizationVersion: 'nfkc-ascii-v1', candidateCompositionVersion: 'mean-alias-context025-v1',
    architecture: 'pooled', dimension: DIMENSION, featureFamily: 'both', byteOrder: 'little-endian',
  };
  for (const [key, expected] of Object.entries(versions)) requireAsset(manifest[key] === expected, `Unsupported ${key}`);
  requireAsset(typeof manifest.modelId === 'string' && manifest.modelId.length > 0, 'Missing model ID');
  requireAsset(typeof manifest.payloadSha256 === 'string' && /^[a-f0-9]{64}$/.test(manifest.payloadSha256), 'Invalid SHA-256');
  requireAsset(manifest.validatedSemanticCutoff === null, 'Experimental format must not claim a validated cutoff');
  requireAsset(payload instanceof ArrayBuffer, 'Expected ArrayBuffer payload');
  requireAsset(manifest.payloadBytes === ELEMENTS * 2 && payload.byteLength === manifest.payloadBytes, 'Invalid payload length');
  requireAsset(Array.isArray(manifest.tensors) && manifest.tensors.length === 2, 'Expected word and char tensors');
  const id = manifest.modelId, hash = manifest.payloadSha256;
  // Snapshot before awaiting hashing; callers cannot mutate accepted bytes or metadata during loading.
  const bytes = payload.slice(0), codes = new Int8Array(bytes);
  const weights: Float32Array[] = [];
  for (let i = 0; i < 2; i++) {
    const tensor = record(manifest.tensors[i]);
    requireAsset(tensor.name === ['word', 'char'][i], 'Unexpected tensor order/name');
    requireAsset(Array.isArray(tensor.shape) && tensor.shape.length === 2 && tensor.shape[0] === 1024 && tensor.shape[1] === DIMENSION, 'Unexpected tensor shape');
    requireAsset(tensor.elementCount === ELEMENTS && tensor.bits === 8, 'Unexpected tensor count/quantization');
    requireAsset(tensor.byteOffset === i * ELEMENTS && tensor.byteLength === ELEMENTS, 'Invalid, overlapping, or noncontiguous tensor range');
    requireAsset(typeof tensor.scale === 'number' && Number.isFinite(tensor.scale) && tensor.scale > 0 && Math.fround(tensor.scale) === tensor.scale, 'Invalid f32 scale');
    requireAsset(typeof tensor.scaleF32LE === 'string' && /^[0-9a-f]{8}$/.test(tensor.scaleF32LE), 'Invalid serialized scale');
    const scaleBytes = new Uint8Array(tensor.scaleF32LE.match(/../g)!.map(hex => parseInt(hex, 16)));
    requireAsset(new DataView(scaleBytes.buffer).getFloat32(0, true) === tensor.scale, 'Scale serialization mismatch');
    const decoded = new Float32Array(ELEMENTS);
    for (let j = 0; j < ELEMENTS; j++) {
      const code = codes[i * ELEMENTS + j]!;
      requireAsset(code !== -128, 'Reserved int8 code');
      decoded[j] = Math.fround(code * tensor.scale);
      requireAsset(Number.isFinite(decoded[j]), 'Dequantized weight overflow');
    }
    weights.push(decoded);
  }
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  const actualHash = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
  requireAsset(actualHash === hash, 'Payload SHA-256 mismatch');
  function encode(text: string): Float32Array {
    if (typeof text !== 'string') throw new TypeError('Expected text string');
    const extracted = features(text), pooled = new Float32Array(DIMENSION);
    for (const [family, ids] of [extracted.wordIds, extracted.charIds].entries()) {
      if (!ids.length) continue;
      const mean = new Float32Array(DIMENSION), table = weights[family]!;
      // Preserve repeated occurrences and collisions instead of treating IDs as a set.
      for (const bucket of ids) for (let d = 0; d < DIMENSION; d++) mean[d] = Math.fround(mean[d]! + table[bucket * DIMENSION + d]!);
      for (let d = 0; d < DIMENSION; d++) pooled[d] = Math.fround(pooled[d]! + Math.fround(Math.fround(mean[d]! / ids.length) * 0.5));
    }
    return normalize(pooled);
  }
  function compose(candidate: Candidate): Float32Array {
    const vectors = [candidate.label, ...candidate.aliases ?? []].map(encode).filter(valid);
    const composed = new Float32Array(DIMENSION);
    if (vectors.length) {
      for (const vector of vectors) for (let d = 0; d < DIMENSION; d++) composed[d] = Math.fround(composed[d]! + vector[d]!);
      for (let d = 0; d < DIMENSION; d++) composed[d] = Math.fround(composed[d]! / vectors.length);
    }
    if (candidate.context) {
      const context = encode(candidate.context);
      for (let d = 0; d < DIMENSION; d++) composed[d] = Math.fround(composed[d]! + Math.fround(context[d]! * 0.25));
    }
    return normalize(composed);
  }
  return Object.freeze({ id, hash, encode, score(query: string, candidates: readonly Candidate[]) {
    const queryVector = encode(query);
    if (!valid(queryVector)) return [];
    return candidates.map((candidate, order) => {
      const vector = compose(candidate);
      if (!valid(vector)) return null;
      let score = 0;
      for (let d = 0; d < DIMENSION; d++) score = Math.fround(score + Math.fround(queryVector[d]! * vector[d]!));
      return { id: candidate.id, label: candidate.label, score: Math.max(-1, Math.min(1, score)), order };
    }).filter(item => item !== null).sort((a, b) => b.score - a.score || a.order - b.order)
      .map(({ id, label, score }) => ({ id, label, score }));
  } });
}
