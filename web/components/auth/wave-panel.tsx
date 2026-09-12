"use client";

import { motion } from "motion/react";
import { ShieldCheck } from "lucide-react";

const BARS = Array.from({ length: 28 }, (_, i) => i);

export function WavePanel() {
  return (
    <div className="relative hidden lg:flex flex-col justify-between overflow-hidden bg-sidebar p-10 grid-bg">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_left,color-mix(in_srgb,var(--live)_18%,transparent),transparent_55%)]" />
      <div className="relative flex items-center gap-2.5">
        <span className="grid size-9 place-items-center rounded-lg bg-primary/15 text-primary">
          <ShieldCheck className="size-5" />
        </span>
        <div className="leading-tight">
          <div className="font-semibold tracking-tight">Centinela Altur</div>
          <div className="text-[11px] text-muted-foreground font-mono uppercase tracking-wider">anti-spoofing de voz</div>
        </div>
      </div>
      <div className="relative">
        <div className="flex h-40 items-end gap-1.5">
          {BARS.map((i) => (
            <motion.span
              key={i}
              className="w-full rounded-full bg-live/80"
              initial={{ height: 8 }}
              animate={{ height: [8, 24 + ((i * 37) % 96), 12, 40 + ((i * 53) % 70), 8] }}
              transition={{ duration: 2.4 + (i % 5) * 0.3, repeat: Infinity, ease: "easeInOut", delay: i * 0.05 }}
            />
          ))}
        </div>
        <p className="mt-8 max-w-sm text-2xl font-semibold tracking-tight">Detecta voces sintéticas antes de que el fraude ocurra.</p>
        <p className="mt-3 max-w-sm text-sm text-muted-foreground">
          Análisis conversacional, anti-spoofing acústico y biomarcadores de voz en tiempo real, mientras la llamada sucede.
        </p>
      </div>
      <div className="relative text-[11px] font-mono text-muted-foreground">HackMTY 2026 · Altur Challenge</div>
    </div>
  );
}
