import Link from "next/link";
import { PhoneCall, Bot, PhoneOff, Target, Timer } from "lucide-react";
import { Topbar } from "@/components/layout/topbar";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { CallsTable } from "@/components/calls/calls-table";
import { HourlyChart } from "@/components/dashboard/hourly-chart";
import { dashboardStats } from "@/lib/queries";
import { fmtDuration, pct } from "@/lib/utils";

export const dynamic = "force-dynamic";

export default async function DashboardPage() {
  const { totals, hourly, recent } = await dashboardStats();
  const synthRate = totals.total ? totals.synthetic / totals.total : null;
  return (
    <>
      <Topbar
        title="Panel"
        subtitle="Últimas 24 horas"
        actions={<Button asChild><Link href="/call/new"><PhoneCall className="size-4" /> Nueva llamada</Link></Button>}
      />
      <main className="flex-1 space-y-4 p-4 lg:p-6">
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
          <Kpi label="Llamadas" value={String(totals.total)} icon={PhoneCall} />
          <Kpi label="Voz IA detectada" value={pct(synthRate)} hint={`${totals.synthetic} llamadas`} icon={Bot} tone="synthetic" />
          <Kpi label="Colgadas por operador" value={String(totals.hangups)} icon={PhoneOff} />
          <Kpi label="Acierto vs etiqueta" value={pct(totals.accuracy)} hint="solo llamadas reveladas" icon={Target} tone="human" />
          <Kpi label="Duración media" value={fmtDuration(totals.avgDuration)} icon={Timer} />
        </div>
        <div className="grid min-w-0 gap-4 lg:grid-cols-3">
          <Card className="min-w-0 lg:col-span-2">
            <CardHeader><CardTitle className="text-sm">Llamadas por hora</CardTitle></CardHeader>
            <CardContent><HourlyChart rows={hourly} /></CardContent>
          </Card>
          <Card>
            <CardHeader><CardTitle className="text-sm">Cómo leer el semáforo</CardTitle></CardHeader>
            <CardContent className="space-y-3 text-sm">
              <Row tone="bg-human" title="Continuar" body="Prob. de IA baja. Atiende normal." />
              <Row tone="bg-warn" title="Verificar" body="Zona gris. Preguntas de seguridad y observa latencias." />
              <Row tone="bg-synthetic" title="Colgar" body="Por arriba del umbral calibrado. Termina y escala a fraude." />
            </CardContent>
          </Card>
        </div>
        <Card>
          <CardHeader>
            <CardTitle className="text-sm">Llamadas recientes</CardTitle>
            <CardAction><Button variant="ghost" size="sm" asChild><Link href="/calls">Ver todas</Link></Button></CardAction>
          </CardHeader>
          <CardContent><CallsTable rows={recent} compact /></CardContent>
        </Card>
      </main>
    </>
  );
}

function Kpi({ label, value, hint, icon: Icon, tone }: { label: string; value: string; hint?: string; icon: React.ComponentType<{ className?: string }>; tone?: "human" | "synthetic" }) {
  const color = tone === "human" ? "text-human" : tone === "synthetic" ? "text-synthetic" : "text-primary";
  return (
    <Card size="sm">
      <CardContent className="flex items-start justify-between gap-3 pt-4">
        <div>
          <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{label}</div>
          <div className="mt-1 font-mono text-2xl font-semibold tabular-nums">{value}</div>
          {hint ? <div className="text-[11px] text-muted-foreground">{hint}</div> : null}
        </div>
        <span className={`grid size-9 shrink-0 place-items-center rounded-lg bg-secondary ${color}`}><Icon className="size-4" /></span>
      </CardContent>
    </Card>
  );
}

function Row({ tone, title, body }: { tone: string; title: string; body: string }) {
  return (
    <div className="flex gap-3">
      <span className={`mt-1.5 size-2 shrink-0 rounded-full ${tone}`} />
      <div><span className="font-medium">{title}</span> <span className="text-muted-foreground">— {body}</span></div>
    </div>
  );
}
