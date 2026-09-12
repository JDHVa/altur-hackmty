import type { Recommendation } from "./types";

export const VERIFY_FLOOR = 0.35;

export function recommend(p: number, threshold: number): Recommendation {
  if (p >= threshold) return "hangup";
  if (p >= VERIFY_FLOOR) return "verify";
  return "continue";
}

export const RECOMMENDATION_META: Record<Recommendation, { title: string; body: string; tone: "human" | "warn" | "synthetic" }> = {
  continue: { title: "Continuar", body: "Patrones consistentes con una persona real. Atiende con normalidad.", tone: "human" },
  verify: { title: "Verificar identidad", body: "Señales mixtas. Aplica preguntas de seguridad y observa la latencia de respuesta.", tone: "warn" },
  hangup: { title: "Colgar y escalar", body: "Alta probabilidad de voz sintética. Termina la llamada y reporta a fraude.", tone: "synthetic" },
};

export function toneClasses(tone: "human" | "warn" | "synthetic") {
  return {
    human: { text: "text-human", bg: "bg-human", border: "border-human", glow: "glow-human", soft: "bg-human/10" },
    warn: { text: "text-warn", bg: "bg-warn", border: "border-warn", glow: "glow-warn", soft: "bg-warn/10" },
    synthetic: { text: "text-synthetic", bg: "bg-synthetic", border: "border-synthetic", glow: "glow-synthetic", soft: "bg-synthetic/10" },
  }[tone];
}
