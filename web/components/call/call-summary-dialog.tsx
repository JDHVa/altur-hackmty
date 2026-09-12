"use client";

import { useState } from "react";
import { motion } from "motion/react";
import { Eye, ExternalLink, RotateCcw, Loader2, Check, X } from "lucide-react";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { cn, pct, fmtDuration, API_URL } from "@/lib/utils";
import { recommend, RECOMMENDATION_META, toneClasses } from "@/lib/recommendation";
import type { FinalVerdict } from "@/lib/types";

export function CallSummaryDialog({
  open,
  final,
  datasetAnonId,
  saving,
  savedId,
  onReveal,
  onNewCall,
  onOpenDetail,
}: {
  open: boolean;
  final: FinalVerdict | null;
  datasetAnonId: string | null;
  saving: boolean;
  savedId: string | null;
  onReveal: (label: string) => void;
  onNewCall: () => void;
  onOpenDetail: () => void;
}) {
  const [label, setLabel] = useState<string | null>(null);
  const [revealing, setRevealing] = useState(false);
  if (!final) return null;
  const rec = recommend(final.p_final, final.threshold);
  const tone = toneClasses(RECOMMENDATION_META[rec].tone);
  const predicted = final.is_synthetic ? "synthetic" : "human";
  const correct = label ? label === predicted : null;

  async function reveal() {
    if (!datasetAnonId) return;
    setRevealing(true);
    const r = await fetch(`${API_URL}/dataset/calls/${datasetAnonId}/label`);
    const j = (await r.json()) as { label: string };
    setLabel(j.label);
    onReveal(j.label);
    setRevealing(false);
  }

  return (
    <Dialog open={open}>
      <DialogContent className="sm:max-w-md" showCloseButton={false}>
        <DialogHeader>
          <DialogTitle>Llamada finalizada</DialogTitle>
          <DialogDescription>Veredicto del ensemble sobre la llamada completa ({fmtDuration(final.duration_s)}).</DialogDescription>
        </DialogHeader>

        <motion.div initial={{ opacity: 0, scale: 0.96 }} animate={{ opacity: 1, scale: 1 }} className={cn("rounded-xl border p-5 text-center", tone.border, tone.soft)}>
          <div className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">veredicto</div>
          <div className={cn("mt-1 text-3xl font-semibold tracking-tight", tone.text)}>{final.is_synthetic ? "Voz sintética" : "Persona real"}</div>
          <div className="mt-1 font-mono text-sm text-muted-foreground">confianza {pct(final.confidence, 1)} · umbral {pct(final.threshold)}</div>
          <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
            <Stat label="Ensemble" v={final.p_final} />
            <Stat label="A · conv." v={final.p_tabular} />
            <Stat label="B · audio" v={final.p_audio} />
          </div>
        </motion.div>

        {datasetAnonId ? (
          <div className="flex items-center justify-between rounded-lg border border-border px-3 py-2">
            <div className="text-xs">
              <div className="text-muted-foreground">Etiqueta real del dataset</div>
              {label ? (
                <div className="mt-0.5 flex items-center gap-2">
                  <Badge variant="outline" className="font-mono">{label}</Badge>
                  {correct ? (
                    <span className="flex items-center gap-1 text-human"><Check className="size-3.5" /> acierto</span>
                  ) : (
                    <span className="flex items-center gap-1 text-synthetic"><X className="size-3.5" /> fallo</span>
                  )}
                </div>
              ) : (
                <div className="font-mono text-muted-foreground">oculta</div>
              )}
            </div>
            {!label ? (
              <Button size="sm" variant="outline" onClick={reveal} disabled={revealing}>
                {revealing ? <Loader2 className="size-3.5 animate-spin" /> : <Eye className="size-3.5" />}
                Revelar
              </Button>
            ) : null}
          </div>
        ) : null}

        <div className="flex items-center justify-between gap-2 pt-2">
          <div className="text-xs text-muted-foreground">
            {saving ? (
              <span className="flex items-center gap-1.5"><Loader2 className="size-3.5 animate-spin" /> Guardando en Tiger Data…</span>
            ) : savedId ? (
              <span className="flex items-center gap-1.5 text-human"><Check className="size-3.5" /> Guardada</span>
            ) : null}
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={onNewCall}><RotateCcw className="size-4" /> Nueva llamada</Button>
            <Button onClick={onOpenDetail} disabled={!savedId}><ExternalLink className="size-4" /> Ver detalle</Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}

function Stat({ label, v }: { label: string; v: number | null }) {
  return (
    <div className="rounded-lg bg-background/60 px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="font-mono text-sm tabular-nums">{pct(v)}</div>
    </div>
  );
}
