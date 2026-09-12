"use client";

import { motion } from "motion/react";
import { BIO_REFERENCE, type BioFeatures } from "@/lib/types";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";

function Row({ k, value }: { k: keyof BioFeatures; value: number | undefined }) {
  const ref = BIO_REFERENCE[k];
  const lo = Math.min(ref.human, ref.synthetic);
  const hi = Math.max(ref.human, ref.synthetic);
  const span = hi - lo || 1;
  const min = lo - span * 0.5;
  const range = span * 2;
  const pos = value === undefined ? null : Math.max(0, Math.min(1, (value - min) / range));
  const humanPos = (ref.human - min) / range;
  const synthPos = (ref.synthetic - min) / range;
  return (
    <div className="grid grid-cols-[96px_1fr_64px] items-center gap-3 text-xs">
      <Tooltip>
        <TooltipTrigger asChild>
          <span className="cursor-default text-muted-foreground underline decoration-dotted underline-offset-4">{ref.label}</span>
        </TooltipTrigger>
        <TooltipContent className="max-w-60">{ref.hint}</TooltipContent>
      </Tooltip>
      <div className="relative h-5">
        <div className="absolute inset-x-0 top-1/2 h-px bg-border" />
        <span className="absolute top-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-human" style={{ left: `${humanPos * 100}%` }} />
        <span className="absolute top-1/2 size-2 -translate-x-1/2 -translate-y-1/2 rounded-full bg-synthetic" style={{ left: `${synthPos * 100}%` }} />
        {pos !== null ? (
          <motion.span className="absolute top-1/2 h-4 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-foreground shadow" animate={{ left: `${pos * 100}%` }} transition={{ type: "spring", stiffness: 90, damping: 20 }} />
        ) : null}
      </div>
      <span className="text-right font-mono tabular-nums">{value === undefined ? "—" : value.toFixed(3)}</span>
    </div>
  );
}

export function BioTable({ bio }: { bio: Partial<BioFeatures> | null }) {
  return (
    <div className="space-y-3">
      {(Object.keys(BIO_REFERENCE) as (keyof BioFeatures)[]).map((k) => (
        <Row key={k} k={k} value={bio?.[k]} />
      ))}
      <div className="flex flex-wrap items-center gap-4 pt-1 text-[11px] text-muted-foreground">
        <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-human" /> típico humano</span>
        <span className="flex items-center gap-1.5"><span className="size-2 rounded-full bg-synthetic" /> típico IA</span>
        <span className="flex items-center gap-1.5"><span className="h-3 w-1 rounded-full bg-foreground" /> esta llamada</span>
      </div>
    </div>
  );
}
