// Run with Node >=18: node tests/test_camera_recording.js
// Exercise the real viewer state machine; HTTP storage/interpolation has Python tests.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const crypto = require('node:crypto').webcrypto;
const path = require('node:path');
const page = fs.readFileSync(path.join(__dirname, '../scripts/target_camera_viewer.html'), 'utf8');
const pageIds = new Set([...page.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]));
const elements = new Map(), handlers = {}, requests = [];
let now = 0, failSave = false, saved = {cameras: [], chunks: []};
function element(id) {
  // Match the browser: absent HTML elements return null, not an invented stub.
  if (!pageIds.has(id)) return null;
  if (!elements.has(id)) elements.set(id, {
    id, value: '', style: {}, children: [], disabled: false,
    clientWidth: 1000, clientHeight: 800,
    append(child) { this.children.push(child); },
    replaceChildren() { this.children = []; },
    querySelectorAll() { return this.children.flatMap(c => c.children || []).filter(c => c.tagName === 'BUTTON'); },
    addEventListener() {}, focus() { context.document.activeElement = this; },
  });
  return elements.get(id);
}
const clone = structuredClone;
const context = vm.createContext({
  console, structuredClone, Float32Array, crypto, devicePixelRatio: 1,
  performance: {now: () => now}, requestAnimationFrame() {},
  window: {addEventListener() {}},
  document: {
    getElementById: element,
    createElement(tag) { return {tagName: tag.toUpperCase(), children: [], append(child) {this.children.push(child);}}; },
    addEventListener(name, handler) {handlers[name] = handler;},
  },
  fetch: async (url, options) => {
    if (!options) return {ok: true, json: async () => clone(saved)};
    const body = JSON.parse(options.body);
    requests.push({url, body});
    if (url === '/api/import') {
      if (body.scene_id !== 'test-world') return {ok: false, json: async () => ({error: 'Different scene'})};
      saved = clone(body);
      return {ok: true, json: async () => ({output_dir: '/tmp/test'})};
    }
    if (url === '/api/chunks') {
      if (failSave) return {ok: false, json: async () => ({error: 'Simulated storage error'})};
      const first = saved.cameras.length;
      const count=body.advance_frames+body.panorama_frames;
      const chunk = {chunk_id: saved.chunks.length, frame_count: count,
        target_ids: Array.from({length: count}, (_, i) => first+i), restore_camera: clone(body.start_camera)};
      for (let i=0; i<count; i++) saved.cameras.push(clone(body.camera));
      saved.chunks.push(chunk);
      return {ok: true, json: async () => ({chunk, output_dir: '/tmp/test'})};
    }
    const chunk = saved.chunks.pop(); saved.cameras.splice(chunk.target_ids[0]);
    return {ok: true, json: async () => ({restore_camera: clone(chunk.restore_camera)})};
  },
});
const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script); // Check the entire script, including initialization.
vm.runInContext(script.slice(0, script.lastIndexOf('(async()=>')), context);
const run = code => vm.runInContext(code, context);
run(`cfg={poses:[],intrinsics:[],span:1,shown_points:0,total_points:0};
 camera=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
 K=[[80,0,32],[0,90,24],[0,0,1]];W=64;H=48;
 gl={deleteBuffer(){},createBuffer(){return {};},bindBuffer(){},bufferData(){}};
 initialCamera=currentCamera();$('advancecount').value='5';$('panoramacount').value='25';$('upaxis').value='initial';setupEvents();`);
function press(key, extra={}) {handlers.keydown({key, target:{tagName:'CANVAS'}, preventDefault(){}, ...extra});}
const settle = () => new Promise(resolve => setImmediate(resolve));
(async () => {
  run('move(2,0,0);'); press('r');
  assert.equal(run('saving'), true);
  assert.equal(element('width').disabled, true);
  press('r', {repeat:true}); await settle();
  assert.equal(saved.chunks.length, 1); assert.equal(saved.cameras.length, 30);
  assert.equal(requests[0].body.kind, 'advance_panorama');
  assert.equal(requests[0].body.start_camera.c2w[0][3], 0);
  assert.equal(requests[0].body.camera.c2w[0][3], 2);
  assert.equal('samples' in requests[0].body, false);
  assert.equal('time' in requests[0].body, false);
  assert.equal(element('advancecount').disabled, false);
  element('advancecount').value='3';element('panoramacount').value='8';
  run('move(3,0,0);');press('t');await settle();
  assert.equal(saved.cameras.length, 41);
  assert.equal(requests.at(-1).body.start_camera.c2w[0][3], 2);
  press('z', {ctrlKey:true});await settle();
  assert.equal(saved.chunks.length, 1);assert.equal(run('camera[0][3]'), 2);
  failSave=true;run('move(2,0,0);');press('r');await settle();
  assert.equal(run('!!pendingRecording'), true);
  const id=requests.at(-1).body.recording_id;
  failSave=false;press('r');await settle();
  assert.equal(requests.at(-1).body.recording_id,id);
  assert.equal(saved.chunks.length,2);
  failSave=true;run('move(3,0,0);');press('r');await settle();
  press('u');await settle();assert.equal(run('pendingRecording'),null);
  assert.equal(run('camera[0][3]'),4);assert.equal(saved.chunks.length,2);
  failSave=false;press('u');await settle();press('u');await settle();
  assert.equal(saved.chunks.length,0);assert.equal(run('camera[0][3]'),0);
  element('advancecount').value='0';press('r');await settle();assert.equal(saved.chunks.length,0);
  element('advancecount').value='5';element('panoramacount').value='1000';
  press('r');await settle();assert.equal(saved.chunks.length,0);
  element('panoramacount').value='25';
  press('r', {target:{tagName:'INPUT'}});assert.equal(run('pendingRecording'),null);
  run('camera=[[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]];');
  press('r');await settle();assert.equal(saved.chunks.length,0); // Vertical view has no horizontal yaw.
  const base=run('structuredClone(initialCamera)');base.c2w[0][3]=10;
  const imported={scene_id:'test-world',cameras:Array.from({length:4},()=>clone(base)),chunks:[{
    kind:'advance_panorama',chunk_id:0,frame_count:4,target_ids:[0,1,2,3],
    advance_frames:1,panorama_frames:3,world_up:[0,-1,0],restore_camera:clone(base)}]};
  element('targetfile').files=[{name:'target_cameras.json',size:1000,text:async()=>JSON.stringify(imported)}];
  await run('loadTargets()');
  assert.equal(run('camera[0][3]'),10);assert.equal(element('advancecount').value,1);
  assert.equal(element('panoramacount').value,3);assert.equal(element('upaxis').value,'loaded');
  assert.deepEqual(Array.from(run('panoramaUp()')),[0,-1,0]);
  run('move(3,0,0);');press('r');await settle();
  assert.equal(saved.chunks.length,2);assert.equal(requests.at(-1).body.start_camera.c2w[0][3],10);
  press('u');await settle();assert.equal(saved.chunks.length,1);assert.equal(run('camera[0][3]'),10);
  element('targetfile').files=[{name:'wrong.json',size:100,text:async()=>JSON.stringify({...imported,scene_id:'wrong'})}];
  await run('loadTargets()');assert.equal(saved.chunks.length,1);assert.equal(run('camera[0][3]'),10);
  assert.equal(run('saving'),false);assert.equal(element('loadtargets').disabled,false);
  console.log('OK: panorama chunks, continuity, undo, retry, import, restore controls and append from imported endpoint.');
})().catch(error => {console.error(error);process.exitCode=1;});
