import { Topbar } from "@/components/layout/topbar";
import { CallsTable } from "@/components/calls/calls-table";
import { listCalls } from "@/lib/queries";

export const dynamic = "force-dynamic";

export default async function CallsPage() {
  const rows = await listCalls(100);
  return (
    <>
      <Topbar title="Historial" subtitle={`${rows.length} llamadas registradas`} />
      <main className="flex-1 p-4 lg:p-6">
        <CallsTable rows={rows} />
      </main>
    </>
  );
}
