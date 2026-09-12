"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Radio } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { PhoneSim } from "./phone-sim";
import { VerdictGauge } from "./verdict-gauge";
import { RecommendationBanner } from "./recommendation-banner";
import { ScoreTimeline } from "./score-timeline";
import { SignalBreakdown } from "./signal-breakdown";
import { BioTable } from "./bio-table";
import { CallSummaryDialog } from "./call-summary-dialog";
import { useLiveCall } from "@/lib/use-live-call";
import { startMicCapture, playUrlWithAnalyser, type Capture } from "@/lib/audio-capture";
import { API_URL, pct } from "@/lib/utils";
import type { CallSource, DatasetCall } from "@/lib/types";
import { saveCall, setRevealedLabel } from "@/app/(app)/call/actions";

function randomMxNumber() {
  const n = () => Math.floor(Math.random() * 10);
  return `+52 ${n()}${n()} ${n()}${n()}${n()}${n()} ${n()}${n()}${n()}${n()}`;
}

export function CallRoom() {
  const router = useRouter();
  const { state, connect, sendPcm, hangup, reset } = useLiveCall();
  const [source, setSource] = useState<CallSource>("mic");
  const [datasetCalls, setDatasetCalls] = useState<DatasetCall[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [speed, setSpeed] = useState(1);
  const [number, setNumber] = useState("+52 ·· ···· ····");
  const [analyser, setAnalyser] = useState<AnalyserNode | null>(null);
  const [wallElapsed, setWallElapsed] = useState(0);
  const [saving, setSaving] = useState(false);
  const [savedId, setSavedId] = useState<string | null>(null);
  const captureRef = useRef<Capture | null>(null);
  const hungUpRef = useRef(false);
  const savedRef = useRef(false);
  const revealedRef = useRef<string | null>(null);

  useEffect(() => {
    setNumber(randomMxNumber());
    fetch(`${API_URL}/dataset/calls`)
      .then((r) => r.json())
      .then((j: { calls: DatasetCall[] }) => setDatasetCalls(j.calls))
      .catch(() => toast.error("No se pudo cargar el dataset desde la API"));
  }, []);

  useEffect(() => {
    if (state.status !== "live") return;
    const id = setInterval(() => setWallElapsed(Math.floor((Date.now() - (state.startedAt ?? Date.now())) / 1000)), 500);
    return () => clearInterval(id);
  }, [state.status, state.startedAt]);

  const stopMedia = useCallback(() => {
    captureRef.current?.stop();
    captureRef.current = null;
    setAnalyser(null);
  }, []);

  const onCall = useCallback(async () => {
    hungUpRef.current = false;
    savedRef.current = false;
    revealedRef.current = null;
    setSavedId(null);
    setWallElapsed(0);
    if (source === "mic") {
      try {
        const cap = await startMicCapture(8000, sendPcm);
        captureRef.current = cap;
        setAnalyser(cap.analyser);
      } catch {
        toast.error("No se pudo acceder al micrófono");
        return;
      }
      connect("mic");
    } else if (selected) {
      connect("dataset", { anonId: selected, speed });
      try {
        const cap = await playUrlWithAnalyser(`${API_URL}/dataset/calls/${selected}/audio`, speed);
        captureRef.current = cap;
        setAnalyser(cap.analyser);
      } catch {
        toast.error("No se pudo reproducir el audio del dataset");
      }
    }
  }, [source, selected, speed, connect, sendPcm]);

  const onHangup = useCallback(() => {
    hungUpRef.current = true;
    hangup();
    stopMedia();
  }, [hangup, stopMedia]);

  useEffect(() => {
    if (state.status !== "ended" || !state.final || savedRef.current) return;
    savedRef.current = true;
    stopMedia();
    setSaving(true);
    const { audio_base64, ...final } = state.final;
    saveCall({
      source,
      datasetAnonId: source === "dataset" ? selected : null,
      callerNumber: number,
      startedAt: state.startedAt ?? Date.now(),
      hungUpByOperator: hungUpRef.current,
      frames: state.frames,
      final,
      audioBase64: source === "mic" ? audio_base64 ?? null : null,
      revealedLabel: revealedRef.current,
    })
      .then((r) => setSavedId(r.id))
      .catch(() => toast.error("No se pudo guardar la llamada"))
      .finally(() => setSaving(false));
  }, [state.status, state.final, state.frames, state.startedAt, source, selected, number, stopMedia]);

  useEffect(() => {
    if (state.status === "error" && state.error) {
      toast.error(state.error);
      stopMedia();
    }
  }, [state.status, state.error, stopMedia]);

  const onNewCall = useCallback(() => {
    reset();
    stopMedia();
    setSavedId(null);
    setNumber(randomMxNumber());
  }, [reset, stopMedia]);

  const latest = state.latest;
  const live = state.status === "live";
  const elapsed = latest?.t ?? wallElapsed;
  const rec = useMemo(() => (state.final ? state.final.recommendation ?? null : latest?.recommendation ?? null), [state.final, latest]);

  return (
    <div className="grid flex-1 gap-4 overflow-hidden p-4 lg:grid-cols-12 lg:p-6">
      <div className="min-w-0 lg:col-span-5 xl:col-span-4">
        <PhoneSim
          source={source}
          onSourceChange={setSource}
          status={state.status}
          elapsed={elapsed}
          number={number}
          analyser={analyser}
          datasetCalls={datasetCalls}
          selected={selected}
          onSelect={setSelected}
          speed={speed}
          onSpeed={setSpeed}
          onCall={onCall}
          onHangup={onHangup}
        />
      </div>

      <div className="flex min-w-0 flex-col gap-4 lg:col-span-7 xl:col-span-8">
        <Card>
          <CardHeader className="flex-row items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-sm">
              Consola del operador
              {live ? (
                <Badge variant="outline" className="gap-1.5 border-live/40 text-live">
                  <Radio className="size-3 animate-pulse" /> en vivo
                </Badge>
              ) : null}
            </CardTitle>
            <div className="font-mono text-[11px] text-muted-foreground">
              {state.frames.length} ventanas · umbral {pct(state.threshold)}
            </div>
          </CardHeader>
          <CardContent className="grid gap-6 md:grid-cols-[auto_minmax(0,1fr)] md:items-center">
            <VerdictGauge p={state.final?.p_final ?? latest?.p_final ?? null} threshold={state.threshold} active={live} />
            <div className="min-w-0 space-y-4">
              <RecommendationBanner rec={rec} live={live} onHangup={onHangup} />
              <ScoreTimeline frames={state.frames} threshold={state.threshold} height={150} />
            </div>
          </CardContent>
        </Card>

        <div className="grid gap-4 md:grid-cols-2">
          <Card>
            <CardHeader><CardTitle className="text-sm">Señales del ensemble</CardTitle></CardHeader>
            <CardContent>
              <SignalBreakdown pAudio={state.final?.p_audio ?? latest?.p_audio ?? null} pTabular={state.final?.p_tabular ?? latest?.p_tabular ?? null} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle className="text-sm">Biomarcadores de voz (últimos 8 s)</CardTitle></CardHeader>
            <CardContent>
              <BioTable bio={state.final?.bio ?? latest?.bio ?? null} />
            </CardContent>
          </Card>
        </div>
      </div>

      <CallSummaryDialog
        open={state.status === "ended"}
        final={state.final}
        datasetAnonId={source === "dataset" ? selected : null}
        saving={saving}
        savedId={savedId}
        onReveal={(l) => {
          revealedRef.current = l;
          if (savedId) setRevealedLabel(savedId, l).catch(() => undefined);
        }}
        onNewCall={onNewCall}
        onOpenDetail={() => savedId && router.push(`/calls/${savedId}`)}
      />
    </div>
  );
}
