"use client";

import { useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { Loader2 } from "lucide-react";
import { signIn, signUp } from "@/lib/auth-client";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

export function AuthForm({ mode }: { mode: "login" | "register" }) {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const isLogin = mode === "login";

  async function onSubmit(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const fd = new FormData(e.currentTarget);
    const email = String(fd.get("email"));
    const password = String(fd.get("password"));
    const name = String(fd.get("name") ?? "");
    setLoading(true);
    const res = isLogin
      ? await signIn.email({ email, password })
      : await signUp.email({ email, password, name });
    setLoading(false);
    if (res.error) {
      toast.error(res.error.message ?? "No se pudo autenticar");
      return;
    }
    router.push("/");
    router.refresh();
  }

  async function google() {
    await signIn.social({ provider: "google", callbackURL: "/" });
  }

  return (
    <div className="w-full max-w-sm">
      <h1 className="text-2xl font-semibold tracking-tight">{isLogin ? "Bienvenido de vuelta" : "Crear cuenta de operador"}</h1>
      <p className="mt-1 text-sm text-muted-foreground">
        {isLogin ? "Inicia sesión para entrar a la consola." : "Registra tu cuenta para operar la consola."}
      </p>
      <form onSubmit={onSubmit} className="mt-8 space-y-4">
        {!isLogin ? (
          <div className="space-y-2">
            <Label htmlFor="name">Nombre</Label>
            <Input id="name" name="name" required autoComplete="name" placeholder="Ana Operadora" />
          </div>
        ) : null}
        <div className="space-y-2">
          <Label htmlFor="email">Correo</Label>
          <Input id="email" name="email" type="email" required autoComplete="email" placeholder="ana@banco.mx" />
        </div>
        <div className="space-y-2">
          <Label htmlFor="password">Contraseña</Label>
          <Input id="password" name="password" type="password" required minLength={8} autoComplete={isLogin ? "current-password" : "new-password"} />
        </div>
        <Button type="submit" className="w-full" disabled={loading}>
          {loading ? <Loader2 className="size-4 animate-spin" /> : null}
          {isLogin ? "Entrar" : "Crear cuenta"}
        </Button>
      </form>
      {process.env.NEXT_PUBLIC_GOOGLE_AUTH === "1" ? (
        <Button type="button" variant="outline" className="mt-3 w-full" onClick={google}>
          Continuar con Google
        </Button>
      ) : null}
      <p className="mt-6 text-center text-sm text-muted-foreground">
        {isLogin ? "¿Sin cuenta? " : "¿Ya tienes cuenta? "}
        <Link href={isLogin ? "/register" : "/login"} className="text-primary underline-offset-4 hover:underline">
          {isLogin ? "Regístrate" : "Inicia sesión"}
        </Link>
      </p>
    </div>
  );
}
