import { getCallAudio } from "@/lib/queries";
import { requireSession } from "@/lib/session";

export async function GET(_req: Request, ctx: RouteContext<"/calls/[id]/audio">) {
  await requireSession();
  const { id } = await ctx.params;
  const audio = await getCallAudio(id);
  if (!audio) return new Response("sin audio", { status: 404 });
  return new Response(new Uint8Array(audio), { headers: { "content-type": "audio/wav", "cache-control": "private, max-age=3600" } });
}
