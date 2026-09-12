import { Sidebar } from "@/components/layout/sidebar";

export default function AppLayout({ children }: LayoutProps<"/">) {
  return (
    <div className="flex min-h-dvh">
      <Sidebar user={{ name: "Operador", email: "", role: "operador" }} />
      <div className="flex min-w-0 flex-1 flex-col">{children}</div>
    </div>
  );
}
