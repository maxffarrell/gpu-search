import {chromium} from 'playwright';
import {preview} from 'vite';
import assert from 'node:assert/strict';
import {mkdirSync,writeFileSync} from 'node:fs';
const server=await preview({root:'apps/demo',preview:{host:'127.0.0.1',port:4180,strictPort:true}});
const browser=await chromium.launch({headless:true,...(process.env.CHROME_CHANNEL?{channel:process.env.CHROME_CHANNEL}:{})});
try{
 const page=await browser.newPage();const errors:string[]=[],requests:string[]=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://127.0.0.1:4180');await page.locator('#alias-results .result-label').first().waitFor();
 page.on('request',r=>requests.push(r.url()));
 await page.locator('#query').fill('profile');await page.waitForFunction(()=>document.querySelector('#alias-results')?.textContent?.includes('label-equality'));
 await page.locator('[data-query=coworkers]').click();await page.waitForFunction(()=>document.querySelector('#alias-results')?.textContent?.includes('Members'));
 assert.match(await page.locator('#lexical-results').innerText(),/No matching/);
 await page.locator('[data-query=xqzv]').click();await page.waitForFunction(()=>document.querySelector('#alias-results')?.textContent?.includes('No matching'));
 await page.locator('summary').click();await page.locator('#candidate-json').fill('invalid json');await page.locator('#apply').click();assert.ok((await page.locator('#editor-status').innerText()).length>0);
 await page.locator('#candidate-json').fill(JSON.stringify([{id:'safe',label:'<img src=x onerror=alert(1)>'}]));await page.locator('#apply').click();
 await page.locator('#query').fill('<img');await page.waitForFunction(()=>document.querySelector('#alias-results')?.textContent?.includes('<img'));
 assert.equal(await page.locator('#candidate-list img, #alias-results img').count(),0);
 await page.locator('#reset').click();await page.setViewportSize({width:390,height:844});
 assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
 await page.emulateMedia({colorScheme:'dark'});mkdirSync('bench/screenshots',{recursive:true});await page.screenshot({path:'bench/screenshots/mobile-dark.png',fullPage:true});
 assert.deepEqual(errors,[]);assert.deepEqual(requests,[]);
 mkdirSync('bench/reports',{recursive:true});writeFileSync('bench/reports/demo.json',JSON.stringify({generatedAt:new Date().toISOString(),browser:browser.version(),checks:['exact label','alias vs labels-only','no-match','invalid JSON','safe label text','390px containment','dark mode','no page errors','zero requests after initial asset load'],requests,errors},null,2)+'\n');console.log('Demo interaction and privacy checks passed');
}finally{await browser.close();await new Promise<void>(resolve=>server.httpServer.close(()=>resolve()))}
