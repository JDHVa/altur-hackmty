"use client";

import { motion } from "motion/react";
import { Info } from "lucide-react";
import { cn, pct } from "@/lib/utils";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { AudioSignals } from "@/lib/types";

function Bar({ label, hint, value, color, muted }: { label: string; hint: string; value: number | null | undefined; color: string; muted?: boolean }) {
  const v = value ?? null;
  return (
    <div className={cn("space-y-1.5", muted && "opacity-60")}>
      <div className="flex items-center justify-between text-xs">
        <span className="flex items-center gap-1.5 text-muted-foreground">
          {label}
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="size-3.5 opacity-60" />
            </TooltipTrigger>
            <TooltipContent className="max-w-64">{hint}</TooltipContent>
          </Tooltip>
        </span>
        <span className={cn("font-mono tabular-nums", v === null && "text-muted-foreground")}>{pct(v)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-secondary">
        <motion.div className="h-full rounded-full" style={{ background: color }} initial={{ width: "0%" }} animate={{ width: `${(v ?? 0) * 100}%` }} transition={{ type: "spring", stiffness: 80, damping: 20 }} />
      </div>
    </div>
  );
}

export const AUDIO_SIGNAL_META: Record<keyof AudioSignals, { label: string; hint: string; color: string }> = {
  xlsr: { label: "B · XLS-R-SLS", hint: "XLS-R 300M congelado con selección de capas sensibles + MLP. La señal más robusta ante motores TTS no vistos (AUC 1.0 en Piper held-out).", color: "var(--chart-1)" },
  wavlm: { label: "B · WavLM", hint: "Cabeza entrenada sobre WavLM-base-plus congelado (fallback ResNet mel-spec). Contrato base del Camino B.", color: "var(--chart-2)" },
  flow: { label: "B · Flow-LLR", hint: "Normalizing flows (zuko) por clase sobre embeddings WavLM → razón de verosimilitud. Fuerte en el dominio Altur.", color: "var(--chart-3)" },
};

export function SignalBreakdown({ pAudio, pTabular, signals }: { pAudio: number | null; pTabular: number | null; signals?: AudioSignals | null }) {
  const hasSignals = signals && Object.keys(signals).length > 0;
  return (
    <div className="space-y-4">
      <Bar label="A · Conversacional" hint="Latencias, solapamientos, silencios y ritmo de turnos (LightGBM sobre 45 features). Necesita al menos 10 s de llamada." value={pTabular} color="var(--chart-5)" />
      {hasSignals ? (
        <>
          {(Object.keys(AUDIO_SIGNAL_META) as (keyof AudioSignals)[])
            .filter((k) => signals![k] !== undefined)
            .map((k) => (
              <Bar key={k} label={AUDIO_SIGNAL_META[k].label} hint={AUDIO_SIGNAL_META[k].hint} value={signals![k]} color={AUDIO_SIGNAL_META[k].color} />
            ))}
          <Bar label="B · Audio combinado" hint="Promedio ponderado de las señales de audio disponibles (XLS-R 0.5 · WavLM 0.3 · Flow 0.2). Es lo que se fusiona con A." value={pAudio} color="var(--foreground)" muted />
        </>
      ) : (
        <Bar label="B · Anti-spoofing acústico" hint="Cabeza entrenada sobre WavLM congelado + ResNet mel-spec. Busca artefactos de síntesis en el canal del que llama." value={pAudio} color="var(--chart-1)" />
      )}
    </div>
  );
}
