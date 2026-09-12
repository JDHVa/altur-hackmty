"use client";

import { motion } from "motion/react";
import { Info } from "lucide-react";
import { cn, pct } from "@/lib/utils";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

function Bar({ label, hint, value, color }: { label: string; hint: string; value: number | null; color: string }) {
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-xs">
        <span className="flex items-center gap-1.5 text-muted-foreground">
          {label}
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="size-3.5 opacity-60" />
            </TooltipTrigger>
            <TooltipContent className="max-w-60">{hint}</TooltipContent>
          </Tooltip>
        </span>
        <span className={cn("font-mono tabular-nums", value === null && "text-muted-foreground")}>{pct(value)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded-full bg-secondary">
        <motion.div className="h-full rounded-full" style={{ background: color }} initial={{ width: "0%" }} animate={{ width: `${(value ?? 0) * 100}%` }} transition={{ type: "spring", stiffness: 80, damping: 20 }} />
      </div>
    </div>
  );
}

export function SignalBreakdown({ pAudio, pTabular }: { pAudio: number | null; pTabular: number | null }) {
  return (
    <div className="space-y-4">
      <Bar label="A · Conversacional" hint="Latencias, solapamientos, silencios y ritmo de turnos (LightGBM sobre 45 features). Necesita al menos 10 s de llamada." value={pTabular} color="var(--chart-5)" />
      <Bar label="B · Anti-spoofing acústico" hint="Cabeza entrenada sobre WavLM congelado + ResNet mel-spec. Busca artefactos de síntesis en el canal del que llama." value={pAudio} color="var(--chart-1)" />
    </div>
  );
}
