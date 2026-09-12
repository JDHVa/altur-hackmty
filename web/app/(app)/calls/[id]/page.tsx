import { notFound } from "next/navigation";
import { CallDetail } from "@/components/calls/call-detail";
import { getCall } from "@/lib/queries";

export const dynamic = "force-dynamic";

export default async function CallPage({ params }: PageProps<"/calls/[id]">) {
  const { id } = await params;
  const data = await getCall(id);
  if (!data) notFound();
  return <CallDetail data={data} />;
}
