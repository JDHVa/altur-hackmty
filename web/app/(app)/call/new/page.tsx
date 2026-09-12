import { Topbar } from "@/components/layout/topbar";
import { CallRoom } from "@/components/call/call-room";

export default function NewCallPage() {
  return (
    <>
      <Topbar title="Sala de llamada" subtitle="Simula al que marca y observa la consola del operador en tiempo real" />
      <CallRoom />
    </>
  );
}
