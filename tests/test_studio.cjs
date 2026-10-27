// Browser event contract smoke test, using DOM/WebSocket doubles; no camera permission.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const elements = new Map();
function element(id) {
  if (!elements.has(id)) elements.set(id, {value:'', style:{}, readyState:2,
    getContext:()=>({drawImage(){}}), pause(){}, load(){}, removeAttribute(){}});
  return elements.get(id);
}
for (const [k,v] of Object.entries({prompt:'a wizard',backend:'ltx',fps:'24',steps:'7',seed:'42',strength:'.3'})) element(k).value=v;
let ws;
class Socket {
  static OPEN=1;
  constructor(url){this.url=String(url);this.readyState=1;this.sent=[];ws=this;}
  send(data){this.sent.push(JSON.parse(data));}
  close(){this.readyState=3;}
}
const context = {document:{getElementById:element},location:{href:'https://pod.example/api/krea-live/studio',protocol:'https:',hash:''},
  window:{addEventListener(){}}, WebSocket:Socket,URL, setInterval:()=>1,clearInterval(){}, setTimeout(){},performance:{now:()=>1000}};
vm.createContext(context);
vm.runInContext(fs.readFileSync(path.join(__dirname,'../studio/studio.js'),'utf8'),context);
element('start').onclick();
assert.equal(ws.url,'wss://pod.example/api/krea-live/ws');
ws.onopen();
assert.equal(ws.sent[0].settings.backend,'ltx');
assert.equal(ws.sent[0].settings.fps,24);
assert.equal(element('backend').disabled,true);
(async()=>{
 await ws.onmessage({data:JSON.stringify({type:'ready',gpu:'FAKE',vram_gib:32,width:512,height:320})});
 assert.equal(element('output').width,512);
 assert.equal(element('output').height,320);
 element('stop').onclick();
 assert.equal(element('backend').disabled,false);
 console.log('Studio backend selection, proxy URL, output sizing and stop checks passed.');
})();
