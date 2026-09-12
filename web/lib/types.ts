export type Recommendation = "continue" | "verify" | "hangup";
export type Verdict = "human" | "synthetic";
export type CallSource = "mic" | "dataset";

export type BioFeatures = {
  hnr: number;
  shimmer_local: number;
  jitter_local: number;
  voiced_frac: number;
};

export type ScoreFrame = {
  t: number;
  p_audio: number | null;
  p_audio_raw?: number;
  p_tabular: number | null;
  p_final: number;
  threshold: number;
  bio: Partial<BioFeatures>;
  recommendation: Recommendation;
  transcript?: string;
};

export type FinalVerdict = {
  is_synthetic: boolean;
  confidence: number;
  p_final: number;
  p_audio: number | null;
  p_tabular: number | null;
  threshold: number;
  duration_s: number;
  n_turns?: number;
  bio: Partial<BioFeatures>;
  recommendation: Recommendation;
};

export type WsServerMessage =
  | { type: "ready"; threshold: number }
  | ({ type: "frame" } & ScoreFrame)
  | ({ type: "final" } & FinalVerdict)
  | { type: "error"; detail: string };

export type DatasetCall = {
  anon_id: string;
  split: string;
  duration_s: number;
};

export const BIO_REFERENCE: Record<keyof BioFeatures, { human: number; synthetic: number; label: string; hint: string }> = {
  hnr: { human: 9.8, synthetic: 14.2, label: "HNR", hint: "Relación armónico-ruido. Las voces sintéticas suelen ser más limpias." },
  shimmer_local: { human: 0.101, synthetic: 0.091, label: "Shimmer", hint: "Variación de amplitud ciclo a ciclo. Los humanos varían más." },
  jitter_local: { human: 0.019, synthetic: 0.024, label: "Jitter", hint: "Variación de tono ciclo a ciclo." },
  voiced_frac: { human: 0.45, synthetic: 0.57, label: "Fracción sonora", hint: "Proporción del tiempo con voz sonora." },
};
