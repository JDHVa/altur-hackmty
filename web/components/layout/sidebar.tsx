"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity, PhoneCall, History, Settings, ShieldCheck, LogOut } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

const NAV = [
  { href: "/", label: "Panel", icon: Activity },
  { href: "/call/new", label: "Sala de llamada", icon: PhoneCall },
  { href: "/calls", label: "Historial", icon: History },
  { href: "/settings", label: "Ajustes", icon: Settings },
];

export function Sidebar({ user, onSignOut }: { user?: { name: string; email: string; role?: string } | null; onSignOut?: () => void }) {
  const pathname = usePathname();
  return (
    <aside className="hidden md:flex w-60 shrink-0 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground">
      <div className="flex items-center gap-2.5 px-5 h-16 border-b border-sidebar-border">
        <span className="grid size-8 place-items-center rounded-lg bg-primary/15 text-primary">
          <ShieldCheck className="size-4.5" />
        </span>
        <div className="leading-tight">
          <div className="font-semibold tracking-tight">Centinela</div>
          <div className="text-[11px] text-muted-foreground font-mono uppercase tracking-wider">Altur · anti-spoofing</div>
        </div>
      </div>
      <nav className="flex-1 p-3 space-y-1">
        {NAV.map(({ href, label, icon: Icon }) => {
          const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors",
                active ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium" : "text-muted-foreground hover:bg-sidebar-accent/60 hover:text-sidebar-foreground",
              )}
            >
              <Icon className={cn("size-4", active && "text-primary")} />
              {label}
            </Link>
          );
        })}
      </nav>
      <div className="p-3 border-t border-sidebar-border">
        {user ? (
          <div className="flex items-center gap-3 rounded-lg px-2 py-2">
            <span className="grid size-8 place-items-center rounded-full bg-secondary text-xs font-semibold uppercase">{user.name.slice(0, 2)}</span>
            <div className="min-w-0 flex-1 leading-tight">
              <div className="truncate text-sm font-medium">{user.name}</div>
              <div className="truncate text-[11px] text-muted-foreground">{user.role ?? "operador"}</div>
            </div>
            <Button size="icon" variant="ghost" onClick={onSignOut} aria-label="Cerrar sesión">
              <LogOut className="size-4" />
            </Button>
          </div>
        ) : null}
      </div>
    </aside>
  );
}
