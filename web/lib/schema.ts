import { pgTable, pgEnum, text, timestamp, boolean, uuid, real, integer, customType, index } from "drizzle-orm/pg-core";

const bytea = customType<{ data: Buffer }>({ dataType: () => "bytea" });

export const user = pgTable("user", {
  id: text("id").primaryKey(),
  name: text("name").notNull(),
  email: text("email").notNull().unique(),
  emailVerified: boolean("email_verified").notNull().default(false),
  image: text("image"),
  role: text("role").notNull().default("operator"),
  banned: boolean("banned").notNull().default(false),
  banReason: text("ban_reason"),
  banExpires: timestamp("ban_expires"),
  createdAt: timestamp("created_at").notNull().defaultNow(),
  updatedAt: timestamp("updated_at").notNull().defaultNow(),
});

export const session = pgTable("session", {
  id: text("id").primaryKey(),
  expiresAt: timestamp("expires_at").notNull(),
  token: text("token").notNull().unique(),
  createdAt: timestamp("created_at").notNull().defaultNow(),
  updatedAt: timestamp("updated_at").notNull().defaultNow(),
  ipAddress: text("ip_address"),
  userAgent: text("user_agent"),
  userId: text("user_id").notNull().references(() => user.id, { onDelete: "cascade" }),
  impersonatedBy: text("impersonated_by"),
});

export const account = pgTable("account", {
  id: text("id").primaryKey(),
  accountId: text("account_id").notNull(),
  providerId: text("provider_id").notNull(),
  userId: text("user_id").notNull().references(() => user.id, { onDelete: "cascade" }),
  accessToken: text("access_token"),
  refreshToken: text("refresh_token"),
  idToken: text("id_token"),
  accessTokenExpiresAt: timestamp("access_token_expires_at"),
  refreshTokenExpiresAt: timestamp("refresh_token_expires_at"),
  scope: text("scope"),
  password: text("password"),
  createdAt: timestamp("created_at").notNull().defaultNow(),
  updatedAt: timestamp("updated_at").notNull().defaultNow(),
});

export const verification = pgTable("verification", {
  id: text("id").primaryKey(),
  identifier: text("identifier").notNull(),
  value: text("value").notNull(),
  expiresAt: timestamp("expires_at").notNull(),
  createdAt: timestamp("created_at").notNull().defaultNow(),
  updatedAt: timestamp("updated_at").notNull().defaultNow(),
});

export const callSource = pgEnum("call_source", ["mic", "dataset"]);
export const verdict = pgEnum("verdict", ["human", "synthetic"]);
export const recommendation = pgEnum("recommendation", ["continue", "verify", "hangup"]);

export const calls = pgTable(
  "calls",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    operatorId: text("operator_id").references(() => user.id, { onDelete: "set null" }),
    source: callSource("source").notNull(),
    datasetAnonId: text("dataset_anon_id"),
    callerNumber: text("caller_number"),
    startedAt: timestamp("started_at", { withTimezone: true }).notNull().defaultNow(),
    endedAt: timestamp("ended_at", { withTimezone: true }),
    durationS: real("duration_s"),
    verdict: verdict("verdict"),
    confidence: real("confidence"),
    pFinal: real("p_final"),
    pTabular: real("p_tabular"),
    pAudio: real("p_audio"),
    threshold: real("threshold"),
    recommendation: recommendation("recommendation"),
    hungUpByOperator: boolean("hung_up_by_operator").notNull().default(false),
    revealedLabel: text("revealed_label"),
    notes: text("notes"),
    frames: integer("frames").notNull().default(0),
    audio: bytea("audio"),
  },
  (t) => [index("calls_started_at_idx").on(t.startedAt), index("calls_operator_idx").on(t.operatorId)],
);

export const callScores = pgTable(
  "call_scores",
  {
    time: timestamp("time", { withTimezone: true }).notNull(),
    callId: uuid("call_id").notNull(),
    t: real("t").notNull(),
    pAudio: real("p_audio"),
    pTabular: real("p_tabular"),
    pFinal: real("p_final").notNull(),
    hnr: real("hnr"),
    shimmer: real("shimmer"),
    jitter: real("jitter"),
    voicedFrac: real("voiced_frac"),
  },
  (t) => [index("call_scores_call_time_idx").on(t.callId, t.time)],
);

export type Call = typeof calls.$inferSelect;
export type CallScore = typeof callScores.$inferSelect;
