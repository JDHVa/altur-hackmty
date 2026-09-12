import { and, desc, eq, gte, sql } from "drizzle-orm";
import { db, schema } from "./db";

export async function listCalls(limit = 50) {
  return db
    .select({
      id: schema.calls.id,
      source: schema.calls.source,
      datasetAnonId: schema.calls.datasetAnonId,
      callerNumber: schema.calls.callerNumber,
      startedAt: schema.calls.startedAt,
      durationS: schema.calls.durationS,
      verdict: schema.calls.verdict,
      confidence: schema.calls.confidence,
      pFinal: schema.calls.pFinal,
      recommendation: schema.calls.recommendation,
      hungUpByOperator: schema.calls.hungUpByOperator,
      revealedLabel: schema.calls.revealedLabel,
      operator: schema.user.name,
    })
    .from(schema.calls)
    .leftJoin(schema.user, eq(schema.user.id, schema.calls.operatorId))
    .orderBy(desc(schema.calls.startedAt))
    .limit(limit);
}

export async function getCall(id: string) {
  const [call] = await db
    .select({
      id: schema.calls.id,
      source: schema.calls.source,
      datasetAnonId: schema.calls.datasetAnonId,
      callerNumber: schema.calls.callerNumber,
      startedAt: schema.calls.startedAt,
      endedAt: schema.calls.endedAt,
      durationS: schema.calls.durationS,
      verdict: schema.calls.verdict,
      confidence: schema.calls.confidence,
      pFinal: schema.calls.pFinal,
      pTabular: schema.calls.pTabular,
      pAudio: schema.calls.pAudio,
      pWavlm: schema.calls.pWavlm,
      pXlsr: schema.calls.pXlsr,
      pFlow: schema.calls.pFlow,
      threshold: schema.calls.threshold,
      recommendation: schema.calls.recommendation,
      hungUpByOperator: schema.calls.hungUpByOperator,
      revealedLabel: schema.calls.revealedLabel,
      notes: schema.calls.notes,
      frames: schema.calls.frames,
      hasAudio: sql<boolean>`${schema.calls.audio} is not null`,
      operator: schema.user.name,
    })
    .from(schema.calls)
    .leftJoin(schema.user, eq(schema.user.id, schema.calls.operatorId))
    .where(eq(schema.calls.id, id));
  if (!call) return null;
  const scores = await db
    .select()
    .from(schema.callScores)
    .where(eq(schema.callScores.callId, id))
    .orderBy(schema.callScores.t);
  return { call, scores };
}

export async function getCallAudio(id: string) {
  const [row] = await db.select({ audio: schema.calls.audio }).from(schema.calls).where(eq(schema.calls.id, id));
  return row?.audio ?? null;
}

export async function dashboardStats() {
  const since = new Date(Date.now() - 24 * 3600 * 1000);
  const [totals] = await db
    .select({
      total: sql<number>`count(*)::int`,
      synthetic: sql<number>`count(*) filter (where ${schema.calls.verdict} = 'synthetic')::int`,
      hangups: sql<number>`count(*) filter (where ${schema.calls.hungUpByOperator})::int`,
      avgDuration: sql<number>`coalesce(avg(${schema.calls.durationS}), 0)::float`,
      accuracy: sql<number | null>`avg(case when ${schema.calls.revealedLabel} is null then null when ${schema.calls.revealedLabel} = ${schema.calls.verdict}::text then 1.0 else 0.0 end)::float`,
    })
    .from(schema.calls)
    .where(gte(schema.calls.startedAt, since));

  const hourly = await db.execute<{ bucket: string; calls: number; synthetic: number }>(sql`
    select time_bucket('1 hour', started_at) as bucket,
           count(*)::int as calls,
           count(*) filter (where verdict = 'synthetic')::int as synthetic
    from calls
    where started_at >= now() - interval '24 hours'
    group by bucket
    order by bucket
  `);

  const recent = await listCalls(8);
  return { totals, hourly: hourly.rows, recent };
}

export async function callsByOperator(operatorId: string, limit = 50) {
  return db
    .select()
    .from(schema.calls)
    .where(and(eq(schema.calls.operatorId, operatorId)))
    .orderBy(desc(schema.calls.startedAt))
    .limit(limit);
}
