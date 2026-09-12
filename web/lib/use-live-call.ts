"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { WS_URL } from "./utils";
import type { CallSource, FinalVerdict, ScoreFrame, WsServerMessage } from "./types";

export type LiveStatus = "idle" | "connecting" | "ringing" | "live" | "finalizing" | "ended" | "error";

export type LiveState = {
  status: LiveStatus;
  threshold: number;
  frames: ScoreFrame[];
  latest: ScoreFrame | null;
  final: (FinalVerdict & { audio_base64?: string }) | null;
  error: string | null;
  startedAt: number | null;
};

const initial: LiveState = { status: "idle", threshold: 0.5, frames: [], latest: null, final: null, error: null, startedAt: null };

export function useLiveCall() {
  const [state, setState] = useState<LiveState>(initial);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback((source: CallSource, opts?: { anonId?: string; speed?: number }) => {
    wsRef.current?.close();
    setState({ ...initial, status: "connecting", startedAt: Date.now() });
    const params = new URLSearchParams({ source });
    if (opts?.anonId) params.set("anon_id", opts.anonId);
    if (opts?.speed) params.set("speed", String(opts.speed));
    const ws = new WebSocket(`${WS_URL}/ws/call?${params.toString()}`);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;
    ws.onopen = () => setState((s) => ({ ...s, status: "ringing" }));
    ws.onmessage = (ev) => {
      const msg = JSON.parse(ev.data) as WsServerMessage;
      setState((s) => {
        if (msg.type === "ready") return { ...s, status: "live", threshold: msg.threshold };
        if (msg.type === "frame") {
          const { type: _t, ...frame } = msg;
          void _t;
          return { ...s, frames: [...s.frames, frame], latest: frame, threshold: frame.threshold };
        }
        if (msg.type === "final") {
          const { type: _t, ...final } = msg;
          void _t;
          return { ...s, status: "ended", final };
        }
        return { ...s, status: "error", error: msg.detail };
      });
    };
    ws.onerror = () => setState((s) => (s.status === "ended" ? s : { ...s, status: "error", error: "No se pudo conectar con el motor de detección." }));
    ws.onclose = () => setState((s) => (s.status === "ended" || s.status === "error" ? s : { ...s, status: s.frames.length ? "finalizing" : "error", error: s.frames.length ? null : "Conexión cerrada." }));
    return ws;
  }, []);

  const sendPcm = useCallback((buf: ArrayBuffer) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(buf);
  }, []);

  const hangup = useCallback(() => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      setState((s) => ({ ...s, status: "finalizing" }));
      ws.send(JSON.stringify({ type: "hangup" }));
    }
  }, []);

  const reset = useCallback(() => {
    wsRef.current?.close();
    wsRef.current = null;
    setState(initial);
  }, []);

  useEffect(() => () => wsRef.current?.close(), []);

  return { state, connect, sendPcm, hangup, reset };
}
