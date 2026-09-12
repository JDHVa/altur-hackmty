CREATE EXTENSION IF NOT EXISTS timescaledb;
--> statement-breakpoint
SELECT create_hypertable('call_scores', by_range('time'), if_not_exists => TRUE);
--> statement-breakpoint
CREATE MATERIALIZED VIEW IF NOT EXISTS call_scores_hourly
WITH (timescaledb.continuous) AS
SELECT time_bucket('1 hour', time) AS bucket,
       call_id,
       avg(p_final) AS avg_p_final,
       max(p_final) AS max_p_final,
       count(*) AS frames
FROM call_scores
GROUP BY bucket, call_id
WITH NO DATA;
--> statement-breakpoint
SELECT add_continuous_aggregate_policy('call_scores_hourly',
  start_offset => INTERVAL '3 days',
  end_offset => INTERVAL '1 minute',
  schedule_interval => INTERVAL '10 minutes',
  if_not_exists => TRUE);
