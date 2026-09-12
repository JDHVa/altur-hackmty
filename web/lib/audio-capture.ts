export type Capture = {
  analyser: AnalyserNode;
  stop: () => void;
};

export async function startMicCapture(targetRate: number, onChunk: (pcm: ArrayBuffer) => void): Promise<Capture> {
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false, autoGainControl: true },
  });
  const ctx = new AudioContext();
  await ctx.audioWorklet.addModule("/pcm-worklet.js");
  const source = ctx.createMediaStreamSource(stream);
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  const node = new AudioWorkletNode(ctx, "pcm-downsampler", { processorOptions: { targetRate } });
  node.port.onmessage = (e) => onChunk(e.data as ArrayBuffer);
  source.connect(analyser);
  source.connect(node);
  node.connect(ctx.destination);
  return {
    analyser,
    stop: () => {
      node.port.onmessage = null;
      node.disconnect();
      source.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      ctx.close();
    },
  };
}

export async function playUrlWithAnalyser(url: string, speed: number, onEnded?: () => void): Promise<Capture> {
  const ctx = new AudioContext();
  const buf = await fetch(url).then((r) => r.arrayBuffer());
  const audio = await ctx.decodeAudioData(buf);
  const src = ctx.createBufferSource();
  src.buffer = audio;
  src.playbackRate.value = speed;
  const analyser = ctx.createAnalyser();
  analyser.fftSize = 1024;
  src.connect(analyser);
  analyser.connect(ctx.destination);
  src.onended = () => onEnded?.();
  src.start();
  return {
    analyser,
    stop: () => {
      src.onended = null;
      try {
        src.stop();
      } catch {}
      src.disconnect();
      analyser.disconnect();
      ctx.close();
    },
  };
}
