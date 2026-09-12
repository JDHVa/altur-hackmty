ALTER TABLE "call_scores" ADD COLUMN "p_wavlm" real;--> statement-breakpoint
ALTER TABLE "call_scores" ADD COLUMN "p_xlsr" real;--> statement-breakpoint
ALTER TABLE "call_scores" ADD COLUMN "p_flow" real;--> statement-breakpoint
ALTER TABLE "calls" ADD COLUMN "p_wavlm" real;--> statement-breakpoint
ALTER TABLE "calls" ADD COLUMN "p_xlsr" real;--> statement-breakpoint
ALTER TABLE "calls" ADD COLUMN "p_flow" real;