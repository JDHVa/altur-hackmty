"use client";

import { Area, AreaChart, CartesianGrid, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ScoreFrame } from "@/lib/types";
import { VERIFY_FLOOR } from "@/lib/recommendation";
import { fmtDuration } from "@/lib/utils";

const NAMES: Record<string, string> = { final: "Ensemble", audio: "Audio (B)", tabular: "Conversacional (A)" };

export function ScoreTimeline({ frames, threshold, height = 176 }: { frames: ScoreFrame[]; threshold: number; height?: number }) {
  const data = frames.map((f) => ({ t: f.t, final: f.p_final, audio: f.p_audio ?? undefined, tabular: f.p_tabular ?? undefined }));
  return (
    <div className="w-full" style={{ height }}>
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ top: 8, right: 8, left: -24, bottom: 0 }}>
          <defs>
            <linearGradient id="gFinal" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--synthetic)" stopOpacity={0.55} />
              <stop offset={`${(1 - threshold) * 100}%`} stopColor="var(--warn)" stopOpacity={0.3} />
              <stop offset="100%" stopColor="var(--human)" stopOpacity={0.08} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="t" tickFormatter={(v) => fmtDuration(Number(v))} stroke="var(--muted-foreground)" fontSize={11} tickLine={false} axisLine={false} minTickGap={32} />
          <YAxis domain={[0, 1]} ticks={[0, 0.5, 1]} tickFormatter={(v) => `${Math.round(Number(v) * 100)}%`} stroke="var(--muted-foreground)" fontSize={11} tickLine={false} axisLine={false} />
          <Tooltip
            contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12 }}
            labelFormatter={(v) => `t = ${fmtDuration(Number(v))}`}
            formatter={(v, name) => [`${Math.round(Number(v) * 100)}%`, NAMES[String(name)] ?? String(name)]}
          />
          <ReferenceLine y={VERIFY_FLOOR} stroke="var(--warn)" strokeDasharray="4 4" />
          <ReferenceLine y={threshold} stroke="var(--synthetic)" strokeDasharray="4 4" />
          <Area type="monotone" dataKey="audio" stroke="var(--chart-1)" strokeWidth={1} strokeOpacity={0.5} fill="none" dot={false} isAnimationActive={false} />
          <Area type="monotone" dataKey="tabular" stroke="var(--chart-5)" strokeWidth={1} strokeOpacity={0.6} fill="none" dot={false} isAnimationActive={false} />
          <Area type="monotone" dataKey="final" stroke="var(--foreground)" strokeWidth={2} fill="url(#gFinal)" dot={false} isAnimationActive={false} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
