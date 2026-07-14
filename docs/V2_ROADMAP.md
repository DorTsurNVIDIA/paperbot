# Paperbot v2 roadmap

## Product goal

Turn Paperbot from a broad keyword notifier into a trusted research analyst for speculative decoding: high precision in the main Slack lane, useful coverage of adjacent inference work, and enough structured evidence to decide what deserves a full read.

The useful unit is not “an LLM summary.” It is a ranked, auditable research record with a stable identity, explicit relevance dimensions, extracted evidence, delivery status, and researcher feedback.

## Foundation in this branch

- Two independent scores: `specdec_score` and `inference_score`.
- Controlled tags and an explicit `SPECDEC`/`INFERENCE` Slack label.
- Specdec-first, score-descending delivery.
- Canonical paper identities across arXiv revisions and providers.
- Lossless processing: provider caps, parse failures, and delivery failures are retried rather than marked seen.
- Slack and Semantic Scholar retry handling.
- Atomic local state writes, workflow concurrency, and CI tests.
- A generic OpenAI-compatible provider adapter for internal or self-hosted model endpoints.
- Every qualifying specdec paper plus at most three broader-inference papers per run.
- Webhook-only delivery history and an idempotent weekly recap.

## Phase 1: measure ranking quality

Before iterating on prompts or models, build a small gold set from roughly 100–200 historical papers. Label each paper independently for:

- exact speculative decoding;
- directly enabling speculative-decoding research;
- broader inference optimization;
- irrelevant;
- important tags and whether the abstract is sufficient to decide.

Track metrics that match the channel experience:

- specdec precision among the first 5 and first 10 results;
- recall of known specdec papers;
- false-positive rate in the specdec lane;
- broader-inference coverage;
- JSON/schema failure rate, latency, and cost per processed paper.

Use the set to compare prompts and candidate internal models. Model choice should follow measured ranking quality, not model size alone.

## Phase 2: two-stage research analysis

Keep abstract classification cheap and bounded for every candidate. Deep-read only the highest-ranked or uncertain papers using the PDF, appendices, and linked code repository.

The deep analysis should extract a structured record:

- paper identity, venue, date, authors, PDF, code, and project links;
- method family, draft strategy, verification/acceptance mechanism, and target/draft models;
- evaluated hardware, batch sizes, sequence lengths, and workload;
- baselines and reported latency, throughput, memory, speedup, and quality deltas;
- where each important claim appears in the paper;
- limitations, missing comparisons, and likely relevance to the team;
- confidence and “needs human review” reasons.

This is the stage where an agent SDK can pay off: it needs tools, multi-step PDF/code inspection, and claim checking. The first-stage classifier should remain a direct structured model call.

## Phase 3: durable archive and historical backfill

`seen_papers.json` records only identity, so it cannot produce the requested historical specdec-only list by itself. Re-fetch known metadata where possible, rescore it with the v2 rubric, and persist a versioned result per paper.

Start with JSONL or SQLite while the bot remains a single scheduled job. Store:

- canonical ID and all source aliases;
- raw metadata and content hashes;
- prompt/rubric version, provider, model, scores, tags, rationale, and timestamps;
- delivery attempts and Slack references;
- human labels and corrections.

Paperbot now records successful webhook deliveries and can generate a weekly recap from that history. Expand this lightweight history into a searchable specdec index and richer “what changed” synthesis. Move to a service database only when multiple writers, an interactive Slack bot, or a web UI makes that necessary.

## Phase 4: Slack as a feedback surface

Keep individual daily paper messages, with every exact-specdec result first and no more than three high-scoring adjacent-inference results. Publish one compact weekly recap through the same incoming webhook. Threading deep analyses would require a bot/API path that can reliably capture message timestamps.

Add lightweight feedback such as “specdec,” “adjacent,” “irrelevant,” and “must read.” Reading reactions or handling buttons requires a Slack app with Events API/interactivity and durable event IDs; an incoming webhook can only publish. Feed these judgments back into the gold set and threshold calibration.

## Phase 5: reliability and operations

- Replace the committed state file with a durable outbox before adding multiple workers.
- Attach a stable paper ID to every delivery and make event handling idempotent.
- Alert on source starvation, sudden acceptance-rate shifts, repeated schema errors, and undelivered messages.
- Record retrieval counts and score distributions by source and day.
- Add contract tests against provider response schemas and a replayable fixture corpus.
- Pin dependencies and add automated dependency/security updates.

## Deployment recommendation

Use a strong internal model through the generic OpenAI-compatible adapter for the first-stage scorer. Configure the endpoint, model, and service-scoped credential in private GitHub variables/secrets; do not encode internal infrastructure in the public repo.

Evaluate an agent SDK or managed internal blueprint only for the deep-reading worker. That worker benefits from sandboxing, PDF/repository tools, memory, and observability; the abstract scorer does not.
