class PcmDownsampler extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.targetRate = options.processorOptions.targetRate;
    this.ratio = sampleRate / this.targetRate;
    this.buffer = [];
    this.acc = 0;
    this.chunk = Math.round(this.targetRate * 0.25);
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];
    for (let i = 0; i < ch.length; i++) {
      this.acc += 1;
      if (this.acc >= this.ratio) {
        this.acc -= this.ratio;
        const s = Math.max(-1, Math.min(1, ch[i]));
        this.buffer.push(s < 0 ? s * 32768 : s * 32767);
      }
    }
    if (this.buffer.length >= this.chunk) {
      const out = new Int16Array(this.buffer.splice(0, this.chunk));
      this.port.postMessage(out.buffer, [out.buffer]);
    }
    return true;
  }
}
registerProcessor("pcm-downsampler", PcmDownsampler);
