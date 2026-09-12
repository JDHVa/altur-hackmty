import { getCall } from "@/lib/queries";
import { requireSession } from "@/lib/session";

export async function GET(_req: Request, ctx: RouteContext<"/calls/[id]/export">) {
  await requireSession();
  const { id } = await ctx.params;
  const data = await getCall(id);
  if (!data) return new Response("no encontrada", { status: 404 });
  return Response.json(data, { headers: { "content-disposition": `attachment; filename="centinela-${id}.json"` } });
}
