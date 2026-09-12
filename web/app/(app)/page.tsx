import { Topbar } from "@/components/layout/topbar";

export default function DashboardPage() {
  return (
    <>
      <Topbar title="Panel" subtitle="Resumen de llamadas y detecciones" />
      <main className="flex-1 p-6" />
    </>
  );
}
