import { AppShell } from "@/components/layout/app-shell";
import { requireSession } from "@/lib/session";

export default async function AppLayout({ children }: LayoutProps<"/">) {
  const session = await requireSession();
  const u = session.user as { name: string; email: string; role?: string | null };
  return <AppShell user={{ name: u.name, email: u.email, role: u.role ?? "operator" }}>{children}</AppShell>;
}
