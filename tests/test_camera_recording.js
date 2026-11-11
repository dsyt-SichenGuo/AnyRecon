// Run with Node >=18: node tests/test_camera_recording.js
// Exercise the real viewer state machine; HTTP storage/interpolation has Python tests.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const crypto = require('node:crypto').webcrypto;
const path = require('node:path');
const elements = new Map(), handlers = {}, requests = [];
let now = 0, failSave = false, saved = {cameras: [], chunks: []};
function element(id) {
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
    if (url === '/api/chunks') {
      if (failSave) return {ok: false, json: async () => ({error: 'Simulated storage error'})};
      const first = saved.cameras.length;
      const chunk = {chunk_id: saved.chunks.length, frame_count: body.frame_count,
        target_ids: Array.from({length: body.frame_count}, (_, i) => first+i), restore_camera: clone(body.camera)};
      for (let i=0; i<body.frame_count; i++) saved.cameras.push(clone(body.camera));
      saved.chunks.push(chunk);
      return {ok: true, json: async () => ({chunk, output_dir: '/tmp/test'})};
    }
    const chunk = saved.chunks.pop(); saved.cameras.splice(chunk.target_ids[0]);
    return {ok: true, json: async () => ({restore_camera: clone(chunk.restore_camera)})};
  },
});
const page = fs.readFileSync(path.join(__dirname, '../scripts/target_camera_viewer.html'), 'utf8');
const script = page.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script); // Check the entire script, including initialization.
vm.runInContext(script.slice(0, script.lastIndexOf('(async()=>')), context);
const run = code => vm.runInContext(code, context);
run(`cfg={poses:[],intrinsics:[],span:1,shown_points:0,total_points:0};
 camera=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
 K=[[80,0,32],[0,90,24],[0,0,1]];W=64;H=48;
 gl={deleteBuffer(){},createBuffer(){return {};},bindBuffer(){},bufferData(){}};
 $('framecount').value='5';setupEvents();`);
function press(key, extra={}) {handlers.keydown({key, target:{tagName:'CANVAS'}, preventDefault(){}, ...extra});}
const settle = () => new Promise(resolve => setImmediate(resolve));
(async () => {
  press('r');
  assert.equal(run('recording.frame_count'), 5);
  assert.equal(element('width').disabled, true);
  press('r', {repeat:true}); assert.equal(run('!!recording'), true);
  run('captureRecording();captureRecording();');
  assert.equal(run('recording.samples.length'), 1); // Stationary waits add no samples.
  now=1000; run('move(1,0,0);captureRecording();');
  now=2000; run('move(1,0,0);'); press('r'); await settle();
  assert.equal(saved.chunks.length, 1); assert.equal(saved.cameras.length, 5);
  assert.deepEqual(requests[0].body.samples.map(s => s.c2w[0][3]), [0,1,2]);
  assert.equal(requests[0].body.samples.some(s => 'time' in s), false);
  assert.equal('started' in requests[0].body, false);
  assert.equal('path_length' in requests[0].body, false);
  assert.equal(run('saving'), false); assert.equal(element('framecount').disabled, false);
  element('framecount').value='3'; press('t');
  now=2500; run('move(4,0,0);'); press('u'); await settle();
  assert.equal(run('camera[0][3]'), 2); assert.equal(saved.chunks.length, 1); // Cancel is local.
  press('r'); now=3000; run('move(3,0,0);'); press('r'); await settle();
  assert.equal(saved.chunks.length, 2); assert.equal(saved.cameras.length, 8);
  press('z', {ctrlKey:true}); await settle();
  assert.equal(saved.chunks.length, 1); assert.equal(run('camera[0][3]'), 2);
  assert.equal(requests.at(-1).url, '/api/undo');
  failSave=true; press('r'); now=3500; run('move(2,0,0);'); press('r'); await settle();
  assert.equal(run('!!pendingRecording'), true); assert.equal(run('saving'), false);
  const id=requests.at(-1).body.recording_id;
  failSave=false; press('r'); await settle();
  assert.equal(requests.at(-1).body.recording_id, id); assert.equal(run('pendingRecording'), null);
  assert.equal(saved.chunks.length, 2);
  press('u'); await settle(); press('u'); await settle();
  assert.equal(saved.chunks.length, 0); assert.equal(run('camera[0][3]'), 0);
  element('framecount').value='1'; press('r'); assert.equal(run('recording'), null);
  element('framecount').value='4';
  press('r', {target:{tagName:'INPUT'}}); assert.equal(run('recording'), null);
  press('r'); run('rotate(column(2),0.3);'); press('r'); await settle();
  assert.equal(run('!!recording'), true); // Pure rotation cannot finish a distance-based chunk.
  assert.equal(saved.chunks.length, 0);
  run('move(1,0,0);'); context.document.hidden=true; now=4000; handlers.visibilitychange(); await settle();
  assert.equal(saved.chunks.length, 1); assert.equal(saved.cameras.length, 4);
  console.log('OK: recording hotkeys, position-only snapshots without timestamps, per-chunk N, cancel/undo restoration, failed-save retry and hidden-tab save.');
})().catch(error => {console.error(error);process.exitCode=1;});
