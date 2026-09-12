import { Topbar } from "@/components/layout/topbar";
import { SettingsPanel } from "@/components/settings/settings-panel";
import { requireSession } from "@/lib/session";

export default async function SettingsPage() {
  const session = await requireSession();
  const u = session.user as { name: string; email: string; role?: string | null };
  return (
    <>
      <Topbar title="Ajustes" />
      <main className="flex-1 p-4 lg:p-6">
        <SettingsPanel user={{ name: u.name, email: u.email, role: u.role ?? "operator" }} />
      </main>
    </>
  );
}
