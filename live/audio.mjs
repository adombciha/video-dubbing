export const RATE = 16000;
export const MAX_SEGMENT_SAMPLES = RATE * 5;

export function captureOptions(source, mute) {
  if (!['tab', 'system'].includes(source)) throw new Error('不支援的音訊來源');
  return {
    video: {displaySurface: source === 'tab' ? 'browser' : 'monitor', frameRate: 1},
    audio: {suppressLocalAudioPlayback: source === 'tab' && mute},
    systemAudio: source === 'system' ? 'include' : 'exclude',
    selfBrowserSurface: 'exclude',
  };
}

export class AudioSegments {
  constructor(onSegment) {
    this.onSegment = onSegment;
    this.frames = [];
    this.size = 0;
    this.silence = 0;
    this.previous = null;
  }

  push(samples) {
    if (!(samples instanceof Int16Array) || !samples.length || samples.length > RATE / 2) {
      throw new Error('音訊格式不符');
    }
    const rms = Math.sqrt(samples.reduce((sum, sample) => sum + (sample / 32768) ** 2, 0) / samples.length);
    if (!this.size && rms < 0.003) {
      this.previous = samples;
      return;
    }
    if (!this.size && this.previous) {
      this.frames.push(this.previous);
      this.size += this.previous.length;
      this.previous = null;
    }
    this.frames.push(samples);
    this.size += samples.length;
    this.silence = rms < 0.003 ? this.silence + samples.length : 0;
    if (this.size >= MAX_SEGMENT_SAMPLES || this.silence >= RATE * 0.6) this.flush();
  }

  flush() {
    if (!this.size) return;
    const audio = new Int16Array(this.size);
    let offset = 0;
    for (const frame of this.frames) {
      audio.set(frame, offset);
      offset += frame.length;
    }
    this.frames = [];
    this.size = this.silence = 0;
    this.onSegment(audio);
  }
}
