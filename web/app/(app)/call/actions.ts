"use server";

import { eq } from "drizzle-orm";
import { db, schema } from "@/lib/db";
import { requireSession } from "@/lib/session";
import { recommend } from "@/lib/recommendation";
import type { CallSource, FinalVerdict, ScoreFrame } from "@/lib/types";

export type SaveCallInput = {
  source: CallSource;
  datasetAnonId?: string | null;
  callerNumber: string;
  startedAt: number;
  hungUpByOperator: boolean;
  frames: ScoreFrame[];
  final: FinalVerdict;
  audioBase64?: string | null;
  revealedLabel?: string | null;
};

export async function saveCall(input: SaveCallInput) {
  const session = await requireSession();
  const startedAt = new Date(input.startedAt);
  const endedAt = new Date();
  const [row] = await db
    .insert(schema.calls)
    .values({
      operatorId: session.user.id,
      source: input.source,
      datasetAnonId: input.datasetAnonId ?? null,
      callerNumber: input.callerNumber,
      startedAt,
      endedAt,
      durationS: input.final.duration_s,
      verdict: input.final.is_synthetic ? "synthetic" : "human",
      confidence: input.final.confidence,
      pFinal: input.final.p_final,
      pTabular: input.final.p_tabular,
      pAudio: input.final.p_audio,
      threshold: input.final.threshold,
      recommendation: recommend(input.final.p_final, input.final.threshold),
      hungUpByOperator: input.hungUpByOperator,
      revealedLabel: input.revealedLabel ?? null,
      frames: input.frames.length,
      audio: input.audioBase64 ? Buffer.from(input.audioBase64, "base64") : null,
    })
    .returning({ id: schema.calls.id });

  if (input.frames.length) {
    await db.insert(schema.callScores).values(
      input.frames.map((f) => ({
        time: new Date(input.startedAt + f.t * 1000),
        callId: row.id,
        t: f.t,
        pAudio: f.p_audio,
        pTabular: f.p_tabular,
        pFinal: f.p_final,
        hnr: f.bio.hnr ?? null,
        shimmer: f.bio.shimmer_local ?? null,
        jitter: f.bio.jitter_local ?? null,
        voicedFrac: f.bio.voiced_frac ?? null,
      })),
    );
  }
  return { id: row.id };
}

export async function updateCallNotes(id: string, notes: string) {
  await requireSession();
  await db.update(schema.calls).set({ notes }).where(eq(schema.calls.id, id));
}

export async function setRevealedLabel(id: string, label: string) {
  await requireSession();
  await db.update(schema.calls).set({ revealedLabel: label }).where(eq(schema.calls.id, id));
}
