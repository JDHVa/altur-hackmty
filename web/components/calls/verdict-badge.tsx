import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Recommendation, Verdict } from "@/lib/types";

export function VerdictBadge({ verdict }: { verdict: Verdict | null }) {
  if (!verdict) return <Badge variant="outline">sin veredicto</Badge>;
  const synthetic = verdict === "synthetic";
  return (
    <Badge variant="outline" className={cn("gap-1.5 font-medium", synthetic ? "border-synthetic/40 text-synthetic" : "border-human/40 text-human")}>
      <span className={cn("size-1.5 rounded-full", synthetic ? "bg-synthetic" : "bg-human")} />
      {synthetic ? "Voz IA" : "Humano"}
    </Badge>
  );
}

const REC_LABEL: Record<Recommendation, string> = { continue: "Continuar", verify: "Verificar", hangup: "Colgar" };

export function RecommendationBadge({ rec }: { rec: Recommendation | null }) {
  if (!rec) return null;
  const cls = { continue: "border-human/40 text-human", verify: "border-warn/40 text-warn", hangup: "border-synthetic/40 text-synthetic" }[rec];
  return <Badge variant="outline" className={cn("font-medium", cls)}>{REC_LABEL[rec]}</Badge>;
}
