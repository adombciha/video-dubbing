import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
import {AudioSegments, RATE, captureOptions} from '../live/audio.mjs';

test('silent capture produces no segments and speech buffers stay bounded', () => {
  const audio = [];
  const segments = new AudioSegments(value => audio.push(value));
  for (let i = 0; i < 100; i++) segments.push(new Int16Array(1600));
  assert.equal(audio.length, 0);
  for (let i = 0; i < 100; i++) segments.push(new Int16Array(1600).fill(1000));
  segments.flush();
  assert(audio.length > 1);
  assert(audio.every(chunk => chunk.length <= RATE * 5.5));
  assert.equal(audio.reduce((sum, chunk) => sum + chunk.length, 0), 161600);
  assert.throws(() => segments.push(new Float32Array(1600)));
});

test('short phrases are combined and manual stop preserves the remaining samples', () => {
  const audio = [];
  const expected = [];
  const segments = new AudioSegments(chunk => audio.push(chunk));
  for (let phrase = 0; phrase < 3; phrase++) {
    for (let frame = 0; frame < 12; frame++) {
      const samples = new Int16Array(1600).fill(frame < 5 ? 1000 + phrase : 0);
      expected.push(samples);
      segments.push(samples);
    }
    if (phrase === 0) assert.equal(audio.length, 0);
  }
  assert.equal(audio.length, 1);
  assert(audio[0].length >= RATE * 3);
  for (let frame = 0; frame < 5; frame++) {
    const samples = new Int16Array(1600).fill(2000);
    expected.push(samples);
    segments.push(samples);
  }
  segments.flush();
  assert.equal(audio.length, 2);
  assert.deepEqual(Buffer.concat(audio.map(chunk => Buffer.from(chunk.buffer))),
    Buffer.concat(expected.map(chunk => Buffer.from(chunk.buffer))));
});

test('capture selects audio directly and suppresses local tab playback', () => {
  assert.equal(captureOptions('tab', true).audio.suppressLocalAudioPlayback, true);
  assert.equal(captureOptions('tab', true).video.displaySurface, 'browser');
  assert.equal(captureOptions('system', true).systemAudio, 'include');
  assert.equal(captureOptions('system', true).video.displaySurface, 'monitor');
  assert.equal(captureOptions('system', true).audio.suppressLocalAudioPlayback, false);
});

function harness() {
  const ids = ['home', 'bar', 'audio-source', 'source', 'target', 'mute', 'start', 'stop', 'float', 'caption', 'original', 'metrics', 'status'];
  const nodes = Object.fromEntries(ids.map(id => [id, {
    id, dataset: {}, handlers: {}, value: '', textContent: '', disabled: false,
    addEventListener(name, callback) { this.handlers[name] = callback; },
  }]));
  nodes['audio-source'].value = 'tab';
  nodes.source.value = 'en';
  nodes.target.value = 'zh';
  nodes.mute.checked = true;
  let floating = false;
  nodes.home.append = () => { floating = false; };
  const document = {
    querySelectorAll: () => Object.values(nodes),
    getElementById: id => floating && id !== 'home' ? null : nodes[id],
  };
  const popup = {
    document: {createElement: () => ({}), head: {append() {}}, body: {append() { floating = true; }}},
    handlers: {}, addEventListener(name, callback) { this.handlers[name] = callback; },
  };
  const sockets = [], streams = [], contexts = [], processors = [];
  class Socket {
    static OPEN = 1;
    constructor() { this.readyState = 1; this.sent = []; sockets.push(this); }
    send(data) { this.sent.push(data); }
    close() { this.readyState = 3; this.onclose?.(); }
  }
  class AudioContext {
    constructor() {
      this.sampleRate = RATE;
      this.audioWorklet = {addModule: async () => {}};
      this.destination = {};
      contexts.push(this);
    }
    async resume() {}
    async close() { this.closed = true; }
    createMediaStreamSource() { return {connect() {}}; }
  }
  class AudioWorkletNode {
    constructor() { this.port = {}; processors.push(this); }
    connect() {}
    disconnect() {}
  }
  const window = {isSecureContext: true, addEventListener() {}, documentPictureInPicture: {
    async requestWindow() { this.window = popup; return popup; },
  }};
  const context = vm.createContext({
    document, window, location: {protocol: 'http:', host: '127.0.0.1:7860', href: 'http://127.0.0.1:7860/'},
    navigator: {mediaDevices: {async getDisplayMedia() {
      const tracks = [0, 1].map(() => ({stopped: false, stop() { this.stopped = true; }, addEventListener() {}}));
      const stream = {getTracks: () => tracks, getAudioTracks: () => [tracks[0]]};
      streams.push(stream);
      return stream;
    }}},
    WebSocket: Socket, AudioContext, AudioWorkletNode, AudioSegments, RATE, captureOptions, Int16Array, URL,
  });
  const code = readFileSync(new URL('../live/app.js', import.meta.url), 'utf8').replace(/^import[^\n]*\n/, '');
  vm.runInContext(code, context);
  return {nodes, popup, sockets, streams, contexts, processors};
}

async function begin(h) {
  await h.nodes.start.handlers.click();
  const socket = h.sockets.at(-1);
  socket.onopen();
  await socket.onmessage({data: JSON.stringify({type: 'ready'})});
  return socket;
}

function speech(h, count = 50) {
  for (let i = 0; i < count; i++) {
    h.processors.at(-1).port.onmessage?.({data: new Int16Array(1600).fill(1200).buffer});
  }
}

test('floating bar still translates and stops every media track', async () => {
  const h = harness();
  await h.nodes.float.handlers.click();
  h.nodes.source.value = 'ja';
  h.nodes.target.value = 'en';
  const socket = await begin(h);
  assert.deepEqual(JSON.parse(socket.sent[0]), {source: 'ja', target: 'en'});
  speech(h);
  assert.equal(typeof socket.sent[1], 'object');
  await socket.onmessage({data: JSON.stringify({type: 'result', text: '<b>Hello</b>', source: 'こんにちは',
    seconds: 0.2, asr_seconds: 0.1, translation_seconds: 0.1, audio_seconds: 5})});
  assert.equal(h.nodes.caption.textContent, '<b>Hello</b>');
  assert.equal(h.nodes.original.textContent, 'こんにちは');
  assert.match(h.nodes.metrics.textContent, /辨識 0.1 秒/);
  assert.match(h.nodes.metrics.textContent, /翻譯 0.1 秒/);
  assert.match(h.nodes.metrics.textContent, /25.0×/);
  h.nodes.stop.handlers.click();
  assert.equal(socket.sent.at(-1), 'stop');
  socket.close();
  assert(h.streams[0].getTracks().every(track => track.stopped));
  assert(h.contexts[0].closed);
  assert.equal(h.nodes.start.disabled, false);
  h.popup.handlers.pagehide();
});

test('backlog stops capture and finishes all already captured segments', async () => {
  const h = harness();
  const socket = await begin(h);
  speech(h, 200);
  assert(h.streams[0].getTracks().every(track => track.stopped));
  for (let i = 0; i < 4; i++) {
    await socket.onmessage({data: JSON.stringify({type: 'result', text: '字幕', source: 'speech', seconds: 8,
      asr_seconds: 5, translation_seconds: 3, audio_seconds: 5})});
  }
  assert.equal(socket.sent.filter(value => typeof value === 'object').length, 4);
  assert.equal(socket.sent.at(-1), 'stop');
  socket.close();
  assert.match(h.nodes.status.textContent, /處理速度不足/);
});

test('stop sends a short final phrase without waiting for the minimum chunk length', async () => {
  const h = harness();
  const socket = await begin(h);
  speech(h, 5);
  h.nodes.stop.handlers.click();
  const chunks = socket.sent.filter(value => typeof value === 'object');
  assert.equal(chunks.length, 1);
  assert.equal(new Int16Array(chunks[0]).length, 8000);
  await socket.onmessage({data: JSON.stringify({type: 'result', text: '', source: '', seconds: 0.1,
    asr_seconds: 0.1, translation_seconds: 0, audio_seconds: 0.5})});
  assert.match(h.nodes.metrics.textContent, /翻譯 0 秒/);
  assert.equal(socket.sent.at(-1), 'stop');
  socket.close();
});

test('stopping during model startup releases capture and permits restarting', async () => {
  const h = harness();
  await h.nodes.start.handlers.click();
  h.sockets[0].onopen();
  h.nodes.stop.handlers.click();
  assert(h.streams[0].getTracks().every(track => track.stopped));
  assert.equal(h.nodes.start.disabled, false);
  await begin(h);
  assert.equal(h.streams.length, 2);
  h.nodes.stop.handlers.click();
  h.sockets.at(-1).close();
});

test('worklet mixes channels to PCM16 without audible output', () => {
  let Processor;
  const messages = [];
  class Base { constructor() { this.port = {postMessage: data => messages.push(data)}; } }
  const context = vm.createContext({AudioWorkletProcessor: Base, Int16Array,
    registerProcessor(name, value) { assert.equal(name, 'pcm-capture'); Processor = value; }});
  vm.runInContext(readFileSync(new URL('../live/pcm-worklet.js', import.meta.url), 'utf8'), context);
  const processor = new Processor();
  for (let i = 0; i < 13; i++) processor.process([[new Float32Array(128).fill(0.5), new Float32Array(128).fill(0.5)]]);
  assert.equal(messages.length, 1);
  assert.equal(new Int16Array(messages[0]).length, 1600);
  assert.equal(new Int16Array(messages[0])[0], 16384);
});
