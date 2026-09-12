"use client";

import { motion, useSpring, useTransform, useMotionValueEvent } from "motion/react";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { recommend, RECOMMENDATION_META, toneClasses, VERIFY_FLOOR } from "@/lib/recommendation";

const R = 84;
const C = 2 * Math.PI * R;

export function VerdictGauge({ p, threshold, active, size = 240 }: { p: number | null; threshold: number; active: boolean; size?: number }) {
  const value = p ?? 0;
  const spring = useSpring(value, { stiffness: 60, damping: 18, mass: 0.8 });
  const [display, setDisplay] = useState(value);
  useEffect(() => spring.set(value), [value, spring]);
  useMotionValueEvent(spring, "change", (v) => setDisplay(v));
  const offset = useTransform(spring, (v) => C * (1 - Math.max(0, Math.min(1, v))));
  const rec = recommend(value, threshold);
  const tone = toneClasses(RECOMMENDATION_META[rec].tone);
  const thrAngle = threshold * 360;
  const verifyAngle = VERIFY_FLOOR * 360;

  return (
    <div className="relative grid place-items-center" style={{ width: size, height: size }}>
      <svg viewBox="0 0 200 200" className="absolute inset-0 -rotate-90" width={size} height={size}>
        <circle cx="100" cy="100" r={R} fill="none" stroke="var(--border)" strokeWidth="10" />
        <circle cx="100" cy="100" r={R} fill="none" stroke="var(--warn)" strokeWidth="2" strokeDasharray={`${(threshold - VERIFY_FLOOR) * C} ${C}`} strokeDashoffset={-VERIFY_FLOOR * C} opacity="0.5" />
        <circle cx="100" cy="100" r={R} fill="none" stroke="var(--synthetic)" strokeWidth="2" strokeDasharray={`${(1 - threshold) * C} ${C}`} strokeDashoffset={-threshold * C} opacity="0.5" />
        <motion.circle
          cx="100"
          cy="100"
          r={R}
          fill="none"
          className={cn("transition-[stroke] duration-500", p === null ? "text-muted-foreground" : tone.text)}
          stroke="currentColor"
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={C}
          style={{ strokeDashoffset: offset }}
        />
      </svg>
      <svg viewBox="0 0 200 200" className="absolute inset-0" width={size} height={size}>
        <g transform={`rotate(${verifyAngle} 100 100)`}><line x1="100" y1="6" x2="100" y2="14" stroke="var(--warn)" strokeWidth="2" /></g>
        <g transform={`rotate(${thrAngle} 100 100)`}><line x1="100" y1="4" x2="100" y2="16" stroke="var(--synthetic)" strokeWidth="2.5" /></g>
      </svg>
      {active && p !== null && rec === "hangup" ? (
        <motion.span
          className="absolute inset-3 rounded-full"
          animate={{ boxShadow: ["0 0 0 0px var(--synthetic-glow)", "0 0 0 18px rgba(244,63,94,0)"] }}
          transition={{ duration: 1.4, repeat: Infinity, ease: "easeOut" }}
        />
      ) : null}
      <div className="relative text-center">
        <div className={cn("font-mono text-5xl font-semibold tabular-nums tracking-tight transition-colors duration-500", p === null ? "text-muted-foreground" : tone.text)}>
          {p === null ? "--" : Math.round(display * 100)}
          <span className="align-top text-2xl">%</span>
        </div>
        <div className="mt-1 text-[11px] font-medium uppercase tracking-[0.18em] text-muted-foreground">prob. de voz IA</div>
      </div>
    </div>
  );
}
