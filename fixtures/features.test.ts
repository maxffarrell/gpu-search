import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import { features,normalizeKey,tokenize } from '../packages/model/features.js';
const fixtures=JSON.parse(readFileSync(new URL('./features.json',import.meta.url),'utf8'));
test('Python and TypeScript normalization, tokens, repeated features, IDs and counts agree exactly',()=>{
 for(const {text,normalized,tokens,...expected} of fixtures){
 assert.equal(normalizeKey(text),normalized,JSON.stringify(text));assert.deepEqual(tokenize(text),tokens);assert.deepEqual(features(text),expected);
 }
});
