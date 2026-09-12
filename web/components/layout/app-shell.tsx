"use client";

import { useRouter } from "next/navigation";
import { Sidebar } from "./sidebar";
import { signOut } from "@/lib/auth-client";

export function AppShell({ user, children }: { user: { name: string; email: string; role?: string }; children: React.ReactNode }) {
  const router = useRouter();
  async function onSignOut() {
    await signOut();
    router.push("/login");
    router.refresh();
  }
  return (
    <div className="flex min-h-dvh">
      <Sidebar user={user} onSignOut={onSignOut} />
      <div className="flex min-w-0 flex-1 flex-col">{children}</div>
    </div>
  );
}
