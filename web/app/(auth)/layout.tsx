import { WavePanel } from "@/components/auth/wave-panel";

export default function AuthLayout({ children }: LayoutProps<"/">) {
  return (
    <div className="grid min-h-dvh lg:grid-cols-2">
      <WavePanel />
      <div className="flex items-center justify-center px-6 py-12">{children}</div>
    </div>
  );
}
