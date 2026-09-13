import { normalizeKey, tokenize } from '../core/src/index.js';
export { normalizeKey, tokenize };
const utf8 = new TextEncoder();
export function hash(bytes:number[]|Uint8Array):number {let h=2166136261;for(const b of bytes)h=Math.imul(h^b,16777619)>>>0;return h&1023}
export function features(text:string){
 const all=tokenize(text),tokens=all.slice(0,32).map(t=>Array.from(t).slice(0,64).join(''));
 const wordIds:number[]=[],charIds:number[]=[];let total=0;
 for(const token of tokens){
 wordIds.push(hash(utf8.encode('w:'+token)));
 const elements=[[1],...Array.from(token).map(c=>{const n=c.codePointAt(0)!;return [0,n&255,(n>>>8)&255,(n>>>16)&255,(n>>>24)&255]}),[2]];
 for(const n of [2,3,4])for(let i=0;i<=elements.length-n;i++){total++;if(charIds.length<512)charIds.push(hash([99,58,...elements.slice(i,i+n).flat()]))}
 }
 return {wordIds,charIds,wordCount:wordIds.length,charCount:charIds.length,truncated:all.length>32||all.slice(0,32).some(t=>Array.from(t).length>64)||total>512};
}
