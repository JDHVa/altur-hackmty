"use client";

import { motion } from "motion/react";
import { Mic, Phone, PhoneOff, Database, Loader2 } from "lucide-react";
import { cn, fmtDuration } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { LiveWaveform } from "./live-waveform";
import type { LiveStatus } from "@/lib/use-live-call";
import type { CallSource, DatasetCall } from "@/lib/types";

const STATUS_TEXT: Record<LiveStatus, string> = {
  idle: "Listo para marcar",
  connecting: "Conectando con el motor…",
  ringing: "Marcando…",
  live: "En llamada",
  finalizing: "Cerrando llamada…",
  ended: "Llamada finalizada",
  error: "Sin conexión",
};

export function PhoneSim({
  source,
  onSourceChange,
  status,
  elapsed,
  number,
  analyser,
  datasetCalls,
  selected,
  onSelect,
  speed,
  onSpeed,
  onCall,
  onHangup,
}: {
  source: CallSource;
  onSourceChange: (s: CallSource) => void;
  status: LiveStatus;
  elapsed: number;
  number: string;
  analyser: AnalyserNode | null;
  datasetCalls: DatasetCall[];
  selected: string | null;
  onSelect: (id: string) => void;
  speed: number;
  onSpeed: (s: number) => void;
  onCall: () => void;
  onHangup: () => void;
}) {
  const active = status === "live" || status === "ringing" || status === "connecting" || status === "finalizing";
  const canCall = status === "idle" || status === "ended" || status === "error";
  return (
    <div className="flex h-full flex-col rounded-2xl border border-border bg-card">
      <div className="flex items-center justify-between border-b border-border px-5 py-3">
        <div>
          <div className="text-sm font-semibold">Quien llama</div>
          <div className="text-[11px] uppercase tracking-wider text-muted-foreground">simulación</div>
        </div>
        <Tabs value={source} onValueChange={(v) => onSourceChange(v as CallSource)}>
          <TabsList>
            <TabsTrigger value="mic" disabled={active}><Mic className="size-3.5" /> Micrófono</TabsTrigger>
            <TabsTrigger value="dataset" disabled={active}><Database className="size-3.5" /> Dataset</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      <div className="flex flex-1 flex-col items-center justify-center gap-6 px-6 py-8">
        <div className="text-center">
          <div className="font-mono text-2xl tracking-[0.12em] tabular-nums">{number}</div>
          <div className="mt-1 flex items-center justify-center gap-2 text-sm text-muted-foreground">
            {active ? <motion.span className="size-2 rounded-full bg-live" animate={{ opacity: [1, 0.2, 1] }} transition={{ duration: 1.2, repeat: Infinity }} /> : null}
            {STATUS_TEXT[status]}
            {status === "live" ? <span className="font-mono tabular-nums">· {fmtDuration(elapsed)}</span> : null}
          </div>
        </div>

        <div className="w-full rounded-xl border border-border bg-background/60 px-3 py-2">
          <LiveWaveform analyser={analyser} active={status === "live"} className="h-20" />
        </div>

        {source === "dataset" ? (
          <div className="grid w-full grid-cols-[1fr_auto] gap-2">
            <Select value={selected ?? undefined} onValueChange={onSelect} disabled={active}>
              <SelectTrigger className="w-full font-mono text-xs">
                <SelectValue placeholder="Elige una llamada real del dataset" />
              </SelectTrigger>
              <SelectContent className="max-h-72">
                {datasetCalls.map((c) => (
                  <SelectItem key={c.anon_id} value={c.anon_id} className="font-mono text-xs">
                    {c.anon_id} · {fmtDuration(c.duration_s)} · {c.split}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Select value={String(speed)} onValueChange={(v) => onSpeed(Number(v))} disabled={active}>
              <SelectTrigger className="w-20 font-mono text-xs"><SelectValue /></SelectTrigger>
              <SelectContent>
                {[1, 2, 4, 8].map((s) => <SelectItem key={s} value={String(s)} className="font-mono text-xs">{s}×</SelectItem>)}
              </SelectContent>
            </Select>
          </div>
        ) : (
          <p className="max-w-xs text-center text-xs text-muted-foreground">
            Habla al micrófono como si llamaras al banco. El audio se procesa como canal telefónico a 8 kHz.
          </p>
        )}

        <div className="flex items-center gap-4">
          {canCall ? (
            <Button
              size="lg"
              onClick={onCall}
              disabled={source === "dataset" && !selected}
              className="h-16 w-16 rounded-full bg-human text-background shadow-lg shadow-human/30 hover:bg-human/90"
              aria-label="Llamar"
            >
              <Phone className="size-7" />
            </Button>
          ) : (
            <Button size="lg" onClick={onHangup} disabled={status !== "live"} className="h-16 w-16 rounded-full bg-synthetic text-white shadow-lg shadow-synthetic/30 hover:bg-synthetic/90" aria-label="Colgar">
              {status === "finalizing" || status === "connecting" ? <Loader2 className="size-7 animate-spin" /> : <PhoneOff className="size-7" />}
            </Button>
          )}
        </div>
      </div>

      <div className={cn("border-t border-border px-5 py-2.5 text-[11px] text-muted-foreground", status === "error" && "text-synthetic")}>
        {status === "error" ? "No se pudo conectar con la API. ¿Está corriendo uvicorn en :8000?" : "Canal 0 = quien llama · canal 1 = agente · 8 kHz PCM"}
      </div>
    </div>
  );
}
