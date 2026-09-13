/** Extract settings descriptions from pinned MIT VS Code source without executing it. */
import ts from 'typescript';
import {readFileSync,writeFileSync,readdirSync,mkdirSync} from 'node:fs';
import {join,relative} from 'node:path';
import {createHash} from 'node:crypto';
import {normalizeKey,tokenize} from '../packages/core/src/index.js';
const revision='8e35945bae3f2b0b3d0276963281180f1ce10cb0';
const root=process.argv[2]??`/tmp/gpu-search-vscode-source/vscode-${revision}`;
const out='data/ui-settings';mkdirSync(out,{recursive:true});
const walk=(dir:string):string[]=>readdirSync(dir,{withFileTypes:true}).flatMap(e=>e.isDirectory()?walk(join(dir,e.name)):e.name.endsWith('.ts')?[join(dir,e.name)]:[]);
const pairs=new Map<string,{id:string;label:string;query:string;family:string;path:string;line:number}>();
function string(node:ts.Node|undefined):string|undefined{return node&&(ts.isStringLiteral(node)||ts.isNoSubstitutionTemplateLiteral(node))?node.text:undefined}
function localized(node:ts.Expression):string|undefined{
 if(ts.isCallExpression(node)&&node.expression.getText().match(/(?:^|\.)localize2?$/))return string(node.arguments[1]);
 return string(node);
}
for(const path of walk(join(root,'src'))){
 if(path.includes('/test/'))continue;
 const source=ts.createSourceFile(path,readFileSync(path,'utf8'),ts.ScriptTarget.Latest,true);
 function visit(node:ts.Node){
 if(ts.isPropertyAssignment(node)&&ts.isObjectLiteralExpression(node.initializer)){
 const id=string(node.name);const props=node.initializer.properties.filter(ts.isPropertyAssignment);
 const description=props.find(p=>['description','markdownDescription'].includes(p.name.getText().replace(/['"]/g,'')));
 const isSetting=props.some(p=>['type','default','enum'].includes(p.name.getText().replace(/['"]/g,'')));
 const parent=node.parent.parent;
 const inProperties=ts.isPropertyAssignment(parent)&&parent.name.getText().replace(/['"]/g,'')==='properties';
 if(id&&/^[a-zA-Z][\w]*(?:\.[\w]+)+$/.test(id)&&description&&isSetting&&inProperties){
 const raw=localized(description.initializer);
 if(raw&&!/\{\d+\}/.test(raw)){
 const query=raw.replace(/\[([^\]]+)\]\([^)]*\)/g,'$1').replace(/[`*_]/g,'').replace(/\s+/g,' ').trim();
 const label=id.split('.').map(p=>tokenize(p).join(' ')).join(' ').replace(/^./,x=>x.toUpperCase());
 if(Array.from(query).length>=12&&Array.from(query).length<=256&&Array.from(label).length<=128&&!pairs.has(id))pairs.set(id,{id,label,query,family:id.split('.')[0],path:relative(root,path),line:source.getLineAndCharacterOfPosition(node.getStart()).line+1});
 }
 }
 }
 ts.forEachChild(node,visit);
 }
 visit(source);
}
const rows=[...pairs.values()].sort((a,b)=>a.id.localeCompare(b.id));
writeFileSync(join(out,'pairs.json'),JSON.stringify(rows,null,2)+'\n');
writeFileSync(join(out,'LICENSE-vscode.txt'),readFileSync(join(root,'LICENSE.txt')));
writeFileSync(join(out,'source.json'),JSON.stringify({source:'https://github.com/microsoft/vscode',revision,license:'MIT',copyright:'Microsoft Corporation',retrievedAt:'2026-09-13',extraction:'TypeScript AST: configuration properties with literal setting ID and description/localize string. Skip tests, placeholders, and inputs beyond library bounds. Query is cleaned original description; label is mechanically humanized setting ID.',queryMeaning:'Descriptions are documentation, not user-authored search queries. Relevance inherited from configuration association; alternative settings unreviewed.',pairs:rows.length,scriptSha256:createHash('sha256').update(readFileSync('training/prepare_settings.ts')).digest('hex')},null,2)+'\n');
console.log(JSON.stringify({pairs:rows.length,families:[...new Set(rows.map(r=>r.family))].length,sample:rows.slice(0,5)},null,2));
