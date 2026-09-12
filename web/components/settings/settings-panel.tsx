"use client";

import { useEffect, useState } from "react";
import { Moon, Sun, Activity, CheckCircle2, XCircle, Loader2 } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { API_URL, WS_URL } from "@/lib/utils";

export function SettingsPanel({ user }: { user: { name: string; email: string; role: string } }) {
  const [dark, setDark] = useState(true);
  const [health, setHealth] = useState<"checking" | "ok" | "down">("checking");

  useEffect(() => {
    try {
      const stored = localStorage.getItem("centinela-theme");
      if (stored === "light") {
        setDark(false);
        document.documentElement.classList.remove("dark");
      }
    } catch {}
  }, []);

  useEffect(() => {
    fetch(`${API_URL}/health`).then((r) => setHealth(r.ok ? "ok" : "down")).catch(() => setHealth("down"));
  }, []);

  function toggleTheme() {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("centinela-theme", next ? "dark" : "light");
    } catch {}
  }

  return (
    <div className="grid max-w-3xl gap-4">
      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Cuenta</CardTitle>
          <CardDescription>Sesión activa en la consola.</CardDescription>
        </CardHeader>
        <CardContent className="flex items-center gap-4">
          <span className="grid size-11 place-items-center rounded-full bg-secondary text-sm font-semibold uppercase">{user.name.slice(0, 2)}</span>
          <div className="flex-1">
            <div className="font-medium">{user.name}</div>
            <div className="text-sm text-muted-foreground">{user.email}</div>
          </div>
          <Badge variant="outline" className="font-mono">{user.role}</Badge>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Motor de detección</CardTitle>
          <CardDescription>API FastAPI con el ensemble A+B y el WebSocket de scoring en vivo.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div className="flex items-center justify-between rounded-lg bg-secondary/50 px-3 py-2">
            <span className="font-mono text-xs">{API_URL}</span>
            {health === "checking" ? <Loader2 className="size-4 animate-spin text-muted-foreground" /> : health === "ok" ? <span className="flex items-center gap-1.5 text-human"><CheckCircle2 className="size-4" /> en línea</span> : <span className="flex items-center gap-1.5 text-synthetic"><XCircle className="size-4" /> sin respuesta</span>}
          </div>
          <div className="flex items-center justify-between rounded-lg bg-secondary/50 px-3 py-2">
            <span className="font-mono text-xs">{WS_URL}/ws/call</span>
            <Activity className="size-4 text-muted-foreground" />
          </div>
          <p className="text-xs text-muted-foreground">El umbral de decisión viene calibrado del ensemble (EER en validación) y se muestra en cada llamada.</p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-sm">Apariencia</CardTitle>
          <CardDescription>Tema oscuro pensado para consola nocturna; claro para presentaciones.</CardDescription>
        </CardHeader>
        <CardContent>
          <Button variant="outline" onClick={toggleTheme}>
            {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
            {dark ? "Cambiar a claro" : "Cambiar a oscuro"}
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
