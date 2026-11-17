import {AudioSegments, RATE, captureOptions} from './audio.mjs';

const elements = new Map(Array.from(document.querySelectorAll('[id]'), node => [node.id, node]));
const element = id => elements.get(id);
const controls = ['audio-source', 'source', 'target', 'mute'].map(element);
let socket, stream, context, processor, segmenter;
let queue = [], busy = false, accepting = false, stopping = false, active = false, overload = false;

function status(text, error = false) {
  element('status').textContent = text;
  element('status').dataset.error = String(error);
}

function setActive(value) {
  active = value;
  element('start').disabled = value;
  element('stop').disabled = !value || stopping;
  controls.forEach(control => { control.disabled = value; });
}

function stopMedia() {
  accepting = false;
  if (processor) { processor.port.onmessage = null; processor.disconnect(); processor = null; }
  if (stream) { stream.getTracks().forEach(track => track.stop()); stream = null; }
  if (context) { context.close().catch(() => {}); context = null; }
}

function pump() {
  if (!socket || socket.readyState !== WebSocket.OPEN || busy) return;
  if (queue.length) {
    busy = true;
    socket.send(queue.shift().buffer);
    status(`${stopping ? '已停止收音，完成剩餘片段' : '翻譯中'} · 等待 ${queue.length} 段`);
  } else if (stopping) {
    socket.send('stop');
  }
}

function enqueue(audio) {
  queue.push(audio);
  if (queue.length >= 3 && !stopping) {
    overload = true;
    stopping = true;
    stopMedia();
    element('stop').disabled = true;
    status('處理速度較慢，已停止收音；正在完成已收取的片段。', true);
  }
  pump();
}

function stop() {
  if (!active || stopping) return;
  stopping = true;
  if (accepting) segmenter?.flush();
  stopMedia();
  element('stop').disabled = true;
  status('已停止收音，正在完成剩餘片段。');
  if (!socket || socket.readyState !== WebSocket.OPEN || !segmenter) socket?.close();
  else pump();
}

function failure(error) {
  status(error.message || String(error), true);
  stopMedia();
  const previous = socket;
  socket = null;
  previous?.close();
  queue = [];
  busy = false;
  stopping = false;
  segmenter = null;
  setActive(false);
}

async function beginCapture() {
  if (!active || stopping || !context || !stream) return;
  const currentContext = context;
  try { await currentContext.audioWorklet.addModule('/static/pcm-worklet.js'); }
  catch (error) {
    if (currentContext !== context || !active || stopping) return;
    throw error;
  }
  if (stopping || currentContext !== context || !stream) return;
  const input = context.createMediaStreamSource(stream);
  processor = new AudioWorkletNode(context, 'pcm-capture');
  segmenter = new AudioSegments(enqueue);
  processor.port.onmessage = event => {
    if (accepting) segmenter.push(new Int16Array(event.data));
  };
  input.connect(processor);
  processor.connect(context.destination);
  accepting = true;
  status('正在收音，可以播放來源內容。');
}

async function start() {
  if (active) return;
  try {
    if (!window.isSecureContext || !navigator.mediaDevices?.getDisplayMedia) {
      throw new Error('請在 Windows Chrome 使用 http://127.0.0.1:7860 或 HTTPS 開啟。');
    }
    queue = [];
    busy = stopping = overload = false;
    segmenter = null;
    setActive(true);
    element('caption').lang = element('target').value === 'zh' ? 'zh-Hant' : element('target').value;
    element('original').lang = element('source').value;
    status('請選擇來源，並勾選「分享音訊」。');
    stream = await navigator.mediaDevices.getDisplayMedia(captureOptions(element('audio-source').value, element('mute').checked));
    if (stopping) { stopMedia(); stopping = false; setActive(false); status('已停止，音訊來源已釋放。'); return; }
    if (!stream.getAudioTracks().length) throw new Error('沒有取得音訊，請重新選擇並勾選「分享音訊」。');
    stream.getTracks().forEach(track => track.addEventListener('ended', stop, {once: true}));
    context = new AudioContext({sampleRate: RATE});
    const currentContext = context;
    await currentContext.resume();
    if (stopping || currentContext !== context) {
      stopMedia(); stopping = false; setActive(false); status('已停止，音訊來源已釋放。'); return;
    }
    if (context.sampleRate !== RATE) throw new Error('瀏覽器無法建立需要的音訊格式。');
    const current = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws/live`);
    socket = current;
    current.onopen = () => {
      if (stopping) { current.close(); return; }
      current.send(JSON.stringify({source: element('source').value, target: element('target').value}));
    };
    current.onmessage = async event => {
      if (socket !== current) return;
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'loading') status(data.message);
        if (data.type === 'ready') await beginCapture();
        if (data.type === 'error') throw new Error(data.message);
        if (data.type === 'result') {
          if (data.text) {
            element('caption').textContent = data.text;
            element('original').textContent = data.source;
          }
          const speed = (data.audio_seconds / Math.max(data.seconds, 0.001)).toFixed(1);
          element('metrics').textContent = `音訊 ${data.audio_seconds} 秒 · 辨識 ${data.asr_seconds} 秒 · 翻譯 ${data.translation_seconds} 秒 · 速度 ${speed}×`;
          busy = false;
          status(stopping ? '已停止收音，完成剩餘片段。' : '正在收音。');
          pump();
        }
      } catch (error) { failure(error); }
    };
    current.onerror = () => {
      if (socket === current) failure(new Error('無法連上翻譯服務，請確認啟動程式的終端機。'));
    };
    current.onclose = () => {
      if (socket !== current) return;
      const expected = stopping;
      stopMedia();
      queue = [];
      busy = stopping = false;
      segmenter = null;
      socket = null;
      setActive(false);
      if (overload) status('已完成已收取的片段。處理速度不足，請暫停來源後重新開始。', true);
      else if (expected) status('已停止，音訊來源已釋放。');
      else if (element('status').dataset.error !== 'true') status('翻譯連線已結束，請重新開始。', true);
    };
  } catch (error) { failure(error); }
}

element('start').addEventListener('click', start);
element('stop').addEventListener('click', stop);
element('float').addEventListener('click', async () => {
  try {
    if (!window.documentPictureInPicture) throw new Error('此瀏覽器不支援置頂視窗，可以繼續使用目前字幕列。');
    if (window.documentPictureInPicture.window) return;
    const floating = await window.documentPictureInPicture.requestWindow({width: 1080, height: 260});
    const style = floating.document.createElement('link');
    style.rel = 'stylesheet';
    style.href = new URL('/static/style.css', location.href).href;
    floating.document.head.append(style);
    floating.document.title = '即時翻譯字幕';
    floating.document.body.append(element('bar'));
    floating.addEventListener('pagehide', () => {
      element('home').append(element('bar'));
      stop();
    }, {once: true});
  } catch (error) { status(error.message, true); }
});
window.addEventListener('beforeunload', () => { stopMedia(); socket?.close(); });
