"use client";

import { useState, useTransition } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { ArrowLeft, Download, Mic, Database, PhoneOff, Save, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Topbar } from "@/components/layout/topbar";
import { VerdictGauge } from "@/components/call/verdict-gauge";
import { ScoreTimeline } from "@/components/call/score-timeline";
import { SignalBreakdown } from "@/components/call/signal-breakdown";
import { VerdictBadge, RecommendationBadge } from "./verdict-badge";
import { updateCallNotes } from "@/app/(app)/call/actions";
import { fmtDuration, pct } from "@/lib/utils";
import type { getCall } from "@/lib/queries";
import type { ScoreFrame } from "@/lib/types";
import { recommend } from "@/lib/recommendation";

type Data = NonNullable<Awaited<ReturnType<typeof getCall>>>;

export function CallDetail({ data }: { data: Data }) {
  const { call, scores } = data;
  const [notes, setNotes] = useState(call.notes ?? "");
  const [pending, start] = useTransition();
  const threshold = call.threshold ?? 0.5;
  const signals = Object.fromEntries(
    Object.entries({ wavlm: call.pWavlm, xlsr: call.pXlsr, flow: call.pFlow }).filter(([, v]) => v !== null && v !== undefined),
  ) as Record<string, number>;
  const frames: ScoreFrame[] = scores.map((s) => ({
    t: s.t,
    p_audio: s.pAudio,
    p_tabular: s.pTabular,
    p_final: s.pFinal,
    signals: Object.fromEntries(Object.entries({ wavlm: s.pWavlm, xlsr: s.pXlsr, flow: s.pFlow }).filter(([, v]) => v !== null)) as Record<string, number>,
    threshold,
    bio: { hnr: s.hnr ?? undefined, shimmer_local: s.shimmer ?? undefined, jitter_local: s.jitter ?? undefined, voiced_frac: s.voicedFrac ?? undefined },
    recommendation: recommend(s.pFinal, threshold),
  }));

  function save() {
    start(async () => {
      await updateCallNotes(call.id, notes);
      toast.success("Notas guardadas");
    });
  }

  return (
    <>
      <Topbar
        title={call.callerNumber ?? "Llamada"}
        subtitle={`${new Date(call.startedAt).toLocaleString("es-MX")} · ${call.operator ?? "operador"}`}
        actions={
          <>
            <Button variant="ghost" size="sm" asChild><Link href="/calls"><ArrowLeft className="size-4" /> Historial</Link></Button>
            <Button variant="outline" size="sm" asChild><a href={`/calls/${call.id}/export`}><Download className="size-4" /> JSON</a></Button>
          </>
        }
      />
      <main className="grid min-w-0 flex-1 gap-4 p-4 lg:grid-cols-12 lg:p-6">
        <Card className="min-w-0 lg:col-span-4">
          <CardHeader><CardTitle className="text-sm">Veredicto final</CardTitle></CardHeader>
          <CardContent className="flex flex-col items-center gap-4">
            <VerdictGauge p={call.pFinal} threshold={threshold} active={false} size={200} />
            <div className="flex flex-wrap items-center justify-center gap-2">
              <VerdictBadge verdict={call.verdict} />
              <RecommendationBadge rec={call.recommendation} />
              {call.hungUpByOperator ? <span className="flex items-center gap-1 text-xs text-muted-foreground"><PhoneOff className="size-3.5" /> colgó el operador</span> : null}
            </div>
            <dl className="grid w-full grid-cols-2 gap-x-4 gap-y-2 text-xs">
              <Meta k="Confianza" v={pct(call.confidence, 1)} />
              <Meta k="Umbral" v={pct(threshold)} />
              <Meta k="Duración" v={call.durationS ? fmtDuration(call.durationS) : "—"} />
              <Meta k="Ventanas" v={String(call.frames)} />
              <Meta k="Fuente" v={call.source === "mic" ? "micrófono" : "dataset"} icon={call.source === "mic" ? Mic : Database} />
              <Meta k="Etiqueta real" v={call.revealedLabel ?? "no revelada"} />
            </dl>
            {call.hasAudio ? <audio controls src={`/calls/${call.id}/audio`} className="w-full" /> : null}
          </CardContent>
        </Card>

        <div className="flex min-w-0 flex-col gap-4 lg:col-span-8">
          <Card>
            <CardHeader><CardTitle className="text-sm">Evolución del score durante la llamada</CardTitle></CardHeader>
            <CardContent>
              {frames.length ? <ScoreTimeline frames={frames} threshold={threshold} height={220} /> : <p className="text-sm text-muted-foreground">Sin ventanas registradas.</p>}
            </CardContent>
          </Card>
          <div className="grid gap-4 md:grid-cols-2">
            <Card>
              <CardHeader><CardTitle className="text-sm">Señales del ensemble</CardTitle></CardHeader>
              <CardContent><SignalBreakdown pAudio={call.pAudio} pTabular={call.pTabular} signals={signals} /></CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle className="text-sm">Notas del operador</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <Textarea value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="Qué observaste, qué preguntaste, cómo respondió…" rows={4} />
                <Button size="sm" onClick={save} disabled={pending}>
                  {pending ? <Loader2 className="size-4 animate-spin" /> : <Save className="size-4" />} Guardar notas
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>
    </>
  );
}

function Meta({ k, v, icon: Icon }: { k: string; v: string; icon?: React.ComponentType<{ className?: string }> }) {
  return (
    <div className="rounded-lg bg-secondary/50 px-3 py-2">
      <dt className="text-[10px] uppercase tracking-wider text-muted-foreground">{k}</dt>
      <dd className="mt-0.5 flex items-center gap-1.5 font-mono">{Icon ? <Icon className="size-3.5 text-muted-foreground" /> : null}{v}</dd>
    </div>
  );
}
