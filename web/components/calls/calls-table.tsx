import Link from "next/link";
import { Mic, Database, PhoneOff } from "lucide-react";
import { VerdictBadge, RecommendationBadge } from "./verdict-badge";
import { fmtDuration, pct } from "@/lib/utils";
import type { listCalls } from "@/lib/queries";

type Row = Awaited<ReturnType<typeof listCalls>>[number];

export function CallsTable({ rows, compact = false }: { rows: Row[]; compact?: boolean }) {
  if (!rows.length) {
    return (
      <div className="rounded-xl border border-dashed border-border p-8 text-center text-sm text-muted-foreground">
        Aún no hay llamadas. Ve a la <Link href="/call/new" className="text-primary underline-offset-4 hover:underline">sala de llamada</Link> para simular una.
      </div>
    );
  }
  return (
    <div className="overflow-x-auto rounded-xl border border-border">
      <table className="w-full text-sm">
        <thead className="bg-secondary/50 text-left text-[11px] uppercase tracking-wider text-muted-foreground">
          <tr>
            <th className="px-4 py-2.5 font-medium">Llamada</th>
            <th className="px-4 py-2.5 font-medium">Veredicto</th>
            <th className="px-4 py-2.5 font-medium">Prob. IA</th>
            {!compact ? <th className="px-4 py-2.5 font-medium">Recomendación</th> : null}
            <th className="px-4 py-2.5 font-medium">Duración</th>
            {!compact ? <th className="px-4 py-2.5 font-medium">Operador</th> : null}
            <th className="px-4 py-2.5 font-medium">Inicio</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.id} className="border-t border-border transition-colors hover:bg-secondary/40">
              <td className="px-4 py-2.5">
                <Link href={`/calls/${r.id}`} className="flex items-center gap-2.5">
                  <span className="grid size-7 place-items-center rounded-md bg-secondary text-muted-foreground">
                    {r.source === "mic" ? <Mic className="size-3.5" /> : <Database className="size-3.5" />}
                  </span>
                  <span className="leading-tight">
                    <span className="block font-mono text-xs">{r.callerNumber ?? "—"}</span>
                    <span className="block text-[11px] text-muted-foreground">{r.datasetAnonId ?? "micrófono"}</span>
                  </span>
                </Link>
              </td>
              <td className="px-4 py-2.5">
                <div className="flex items-center gap-2">
                  <VerdictBadge verdict={r.verdict} />
                  {r.revealedLabel ? (
                    <span className={r.revealedLabel === r.verdict ? "text-[11px] text-human" : "text-[11px] text-synthetic"}>
                      {r.revealedLabel === r.verdict ? "✓" : "✗"} real: {r.revealedLabel}
                    </span>
                  ) : null}
                </div>
              </td>
              <td className="px-4 py-2.5 font-mono tabular-nums">{pct(r.pFinal)}</td>
              {!compact ? (
                <td className="px-4 py-2.5">
                  <div className="flex items-center gap-2">
                    <RecommendationBadge rec={r.recommendation} />
                    {r.hungUpByOperator ? <PhoneOff className="size-3.5 text-muted-foreground" /> : null}
                  </div>
                </td>
              ) : null}
              <td className="px-4 py-2.5 font-mono tabular-nums">{r.durationS ? fmtDuration(r.durationS) : "—"}</td>
              {!compact ? <td className="px-4 py-2.5 text-muted-foreground">{r.operator ?? "—"}</td> : null}
              <td className="px-4 py-2.5 text-muted-foreground">{new Date(r.startedAt).toLocaleString("es-MX", { dateStyle: "short", timeStyle: "short" })}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
