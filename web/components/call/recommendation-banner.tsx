"use client";

import { AnimatePresence, motion } from "motion/react";
import { PhoneOff, ShieldAlert, ShieldCheck, ShieldQuestion } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { RECOMMENDATION_META, toneClasses } from "@/lib/recommendation";
import type { Recommendation } from "@/lib/types";

const ICONS = { continue: ShieldCheck, verify: ShieldQuestion, hangup: ShieldAlert };

export function RecommendationBanner({ rec, live, onHangup }: { rec: Recommendation | null; live: boolean; onHangup: () => void }) {
  if (!rec) {
    return (
      <div className="rounded-xl border border-dashed border-border p-4 text-sm text-muted-foreground">
        Esperando audio del que llama. La recomendación aparece a los pocos segundos.
      </div>
    );
  }
  const meta = RECOMMENDATION_META[rec];
  const tone = toneClasses(meta.tone);
  const Icon = ICONS[rec];
  return (
    <AnimatePresence mode="popLayout" initial={false}>
      <motion.div
        key={rec}
        initial={{ opacity: 0, y: 8, scale: 0.98 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        exit={{ opacity: 0, y: -8, scale: 0.98 }}
        transition={{ type: "spring", stiffness: 260, damping: 24 }}
        className={cn("flex items-center gap-4 rounded-xl border p-4", tone.border, tone.soft, rec === "hangup" && "glow-synthetic")}
      >
        <span className={cn("grid size-11 shrink-0 place-items-center rounded-lg text-background", tone.bg)}>
          <Icon className="size-6" />
        </span>
        <div className="min-w-0 flex-1">
          <div className={cn("text-base font-semibold", tone.text)}>{meta.title}</div>
          <p className="text-sm text-muted-foreground">{meta.body}</p>
        </div>
        {live ? (
          <Button
            size="lg"
            onClick={onHangup}
            variant={rec === "hangup" ? "default" : "outline"}
            className={cn("shrink-0 font-semibold", rec === "hangup" && "bg-synthetic text-white hover:bg-synthetic/90")}
          >
            <PhoneOff className="size-4" />
            Colgar
          </Button>
        ) : null}
      </motion.div>
    </AnimatePresence>
  );
}
