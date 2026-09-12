"use client";

import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

export function HourlyChart({ rows }: { rows: { bucket: string; calls: number; synthetic: number }[] }) {
  const data = rows.map((r) => ({
    hour: new Date(r.bucket).toLocaleTimeString("es-MX", { hour: "2-digit", minute: "2-digit" }),
    humanas: Number(r.calls) - Number(r.synthetic),
    sinteticas: Number(r.synthetic),
  }));
  if (!data.length) {
    return <div className="grid h-52 place-items-center text-sm text-muted-foreground">Sin llamadas en las últimas 24 h.</div>;
  }
  return (
    <div className="h-52 w-full min-w-0 overflow-hidden">
      <ResponsiveContainer width="100%" height="100%" debounce={50}>
        <BarChart data={data} barCategoryGap="30%" maxBarSize={48} margin={{ top: 8, right: 8, left: -24, bottom: 0 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" vertical={false} />
          <XAxis dataKey="hour" stroke="var(--muted-foreground)" fontSize={11} tickLine={false} axisLine={false} />
          <YAxis allowDecimals={false} stroke="var(--muted-foreground)" fontSize={11} tickLine={false} axisLine={false} />
          <Tooltip cursor={{ fill: "var(--secondary)" }} contentStyle={{ background: "var(--popover)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12 }} />
          <Bar dataKey="humanas" name="Humanas" stackId="a" fill="var(--human)" radius={[0, 0, 4, 4]} />
          <Bar dataKey="sinteticas" name="Sintéticas" stackId="a" fill="var(--synthetic)" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
