"use client";

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

export function LiveWaveform({ analyser, active, className, color = "var(--live)" }: { analyser: AnalyserNode | null; active: boolean; className?: string; color?: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    let raf = 0;
    const data = analyser ? new Uint8Array(analyser.fftSize) : null;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth;
      const h = canvas.clientHeight;
      if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const mid = h / 2;
      const styles = getComputedStyle(canvas);
      ctx.strokeStyle = styles.getPropertyValue("--border").trim() || "#243040";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, mid);
      ctx.lineTo(w, mid);
      ctx.stroke();
      if (analyser && data && active) {
        analyser.getByteTimeDomainData(data);
        const bars = 64;
        const step = Math.floor(data.length / bars);
        const bw = w / bars;
        ctx.fillStyle = styles.color;
        for (let i = 0; i < bars; i++) {
          let peak = 0;
          for (let j = 0; j < step; j++) peak = Math.max(peak, Math.abs(data[i * step + j] - 128) / 128);
          const bh = Math.max(2, peak * h * 0.95);
          ctx.globalAlpha = 0.35 + peak * 0.65;
          ctx.beginPath();
          ctx.roundRect(i * bw + bw * 0.25, mid - bh / 2, bw * 0.5, bh, 2);
          ctx.fill();
        }
        ctx.globalAlpha = 1;
      }
      raf = requestAnimationFrame(draw);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [analyser, active]);
  return <canvas ref={ref} className={cn("h-16 w-full", className)} style={{ color }} />;
}
