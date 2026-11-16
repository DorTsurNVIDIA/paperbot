# paperbot

Paperbot is a scheduled research feed for **speculative decoding** and **LLM inference optimization**. It fetches recent papers from arXiv, Semantic Scholar, Hugging Face, alphaXiv, and OpenReview, scores them with an LLM, and posts ranked papers to Slack.

The feed has two explicit lanes:

- **SPECDEC** — speculative decoding is the main method, or the work directly enables it through drafting, verification, acceptance, multi-token prediction, or a closely related mechanism.
- **INFERENCE** — broader work on latency, throughput, memory, serving, or hardware efficiency.

Every qualifying specdec paper is posted first. Each run then posts at most the three strongest broader-inference papers with an inference score of at least 8. Every Slack item shows both scores plus controlled topic tags.

## How it works

1. **Fetch** papers from the last seven days.
2. **Canonicalize and deduplicate** identities across providers, arXiv versions, DOI metadata, and matching full titles.
3. **Classify** each unseen paper with independent 1–10 specdec and inference scores.
4. **Rank and post** every accepted specdec paper followed by the top three broader-inference papers.
5. **Record successful posts** for a webhook-only weekly recap.
6. **Checkpoint safely**: rejected, intentionally suppressed, and successfully delivered papers are saved; failed, rate-limited, capped, or undelivered papers remain eligible for retry.

The twice-daily schedule catches papers missed by a delayed run without posting known papers again.

## Sources and coverage

| Source | Retrieval |
|---|---|
| arXiv | Keyword searches and recent category submissions |
| Semantic Scholar | Topic searches across its indexed papers |
| Hugging Face | Relevant entries from the current daily papers list |
| alphaXiv | Paginated **Recent** feed for `cs.CL`, `cs.LG`, `cs.AI`, `cs.DC`, `cs.AR`, and `stat.ML`, including papers published directly on alphaXiv |
| OpenReview | API v2 public paper forums matching speculative decoding, LLM inference, language model inference, or LLM serving |

The new sources need no API keys or additional Actions secrets. Both daily runs and
backfill audits use them automatically with the requested lookback window.

alphaXiv-exclusive IDs such as
[`2611.loopfill-speculative-decoding-looped-transformers`](https://www.alphaxiv.org/abs/2611.loopfill-speculative-decoding-looped-transformers)
stay in the `alphaxiv:` namespace. Numeric arXiv mirrors share their `arxiv:` IDs.
alphaXiv uses the original publication date so a revision does not reintroduce an
old paper. The same broad title/abstract keyword check used for Hugging Face
keeps unrelated category papers out of the LLM scoring queue. Its feed is the
public interface used by the website, without a formal
API stability guarantee; response schema changes surface in the workflow logs.

OpenReview uses the first public-release date (`odate`), falling back to the
creation date when unavailable. It excludes replies, deleted notes, and submissions
marked withdrawn or desk rejected. Searches are relevance-ranked, so the bot checks
all retrieved pages rather than stopping at an old result. This currently covers
API v2; legacy API v1-only venues are not crawled directly.

The crawl is bounded: alphaXiv checks up to 100 pages of 100 entries and OpenReview
up to 20 pages of 100 entries per phrase. Reaching a cap logs a coverage warning.
These feeds improve recall but do not guarantee every paper: category labels,
search terms, indexing delays, and upstream outages can still leave gaps. Supplemental
source failures are logged while the remaining sources continue; a complete arXiv
outage still fails the run.

To prevent duplicate Slack posts under unrelated provider IDs, full titles of at
least 20 alphanumeric characters also receive a stable identity alias, ignoring
case, spacing, and punctuation. This is exact normalized matching, not fuzzy
matching. Successful delivery history seeds these aliases for older posts; new
aliases are checkpointed only for rejected, intentionally suppressed, or delivered
papers. Failed and deferred papers retain their retry eligibility. Renamed papers
without shared external IDs may still appear twice. `CLEAR_SEEN_PAPERS` explicitly
bypasses both ID and title history for the one-run rescore.

## Setup

### 1. Create a Slack Incoming Webhook

1. Go to [Slack apps](https://api.slack.com/apps) and create an app from scratch.
2. Enable **Incoming Webhooks**, add a webhook to the target channel, and copy its URL.
3. Add it to the repository as an Actions secret named `SLACK_WEBHOOK_URL`.

Incoming Webhooks are enough for daily publishing and the weekly recap. A future interactive feedback loop based on emoji or buttons will require a Slack app with read permissions rather than only a webhook.

### 2. Configure an LLM

Paperbot supports Anthropic, OpenAI, Gemini, Groq, and any OpenAI-compatible chat-completions endpoint.

For a standard provider, add one corresponding Actions secret:

| Provider | Secret | Optional repository variable |
|---|---|---|
| Anthropic | `ANTHROPIC_API_KEY` | `LLM_PROVIDER=anthropic` |
| OpenAI | `OPENAI_API_KEY` | `LLM_PROVIDER=openai` |
| Gemini | `GEMINI_API_KEY` | `LLM_PROVIDER=gemini` |
| Groq | `GROQ_API_KEY` | `LLM_PROVIDER=groq` |

Without `LLM_PROVIDER`, the first available key is selected in the order shown above.

For an internal or self-hosted OpenAI-compatible service, configure:

| Kind | Name | Value |
|---|---|---|
| Actions variable | `LLM_PROVIDER` | `openai_compatible` |
| Actions variable | `LLM_BASE_URL` | Base URL ending at the API version, as required by the service |
| Actions variable | `LLM_MODEL` | Model ID exposed by the endpoint |
| Actions secret | `LLM_API_KEY` | Service credential, if the endpoint requires one |

Keep internal hostnames, credentials, and deployment-specific instructions out of this public repository.

### 3. Optional: configure Semantic Scholar

Unauthenticated Semantic Scholar requests are frequently rate-limited. Add `SEMANTIC_SCHOLAR_API_KEY` as an Actions secret for more reliable retrieval. Paperbot retries transient 429 and 5xx responses with bounded backoff.

### 4. Allow workflow state updates

The scheduled workflows commit `seen_papers.json`, successful post history, and weekly idempotency state back to the current branch.

1. Open **Settings → Actions → General**.
2. Under **Workflow permissions**, select **Read and write permissions**.
3. Keep GitHub's `actions/checkout` and `actions/setup-python` actions allowed by the repository or organization policy.

The production workflows share a concurrency group so daily and weekly runs cannot update state simultaneously.

### 5. Run it

The paper workflow runs at 08:00 and 20:00 UTC and can also be started from **Actions → Daily Papers Agent → Run workflow**. The manual `clear_seen_papers` option performs a one-run rescore of everything fetched in the current lookback window.

The weekly recap runs Monday at 08:30 UTC for the previous ISO week. It can be run manually for a specific `YYYY-Www` week and will not repost a completed week unless `force` is selected. It reads `posted_papers.json`, not the Slack channel, so no bot token or read permission is required.

For a historical gap, **Actions → Backfill Audit** runs a read-only crawl for 1–30 days, compares the results with `seen_papers.json`, scores unseen candidates, and uploads `backfill_report.json`. This workflow never imports Slack delivery, never posts messages, and has read-only repository permissions. Review its report before doing any manual catch-up post.

## Local development

```bash
python -m pip install -r requirements.txt

export LLM_PROVIDER=openai_compatible
export LLM_BASE_URL=https://your-service.example/v1
export LLM_MODEL=your-model-id
export LLM_API_KEY=your-service-key
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

python -m agent.main
python -m agent.backfill --days 14 --fetch-only
DRY_RUN=true python -m agent.weekly_digest
python -m unittest discover -v
```

For a local scoring-only run, omit `SLACK_WEBHOOK_URL` and set `DRY_RUN=true`. Accepted papers are then treated as handled for that local state file, so use a disposable copy of `seen_papers.json` if you intend to deliver them later. Without a webhook or explicit dry-run flag, Paperbot fails closed instead of silently losing deliveries.

## Configuration

| Setting | Location | Default |
|---|---|---|
| Lookback window | `agent/fetch.py` → `LOOKBACK_HOURS` | 168 hours |
| Specdec threshold | `agent/filter.py` → `SPECDEC_THRESHOLD` | 6 / 10 |
| Inference threshold | `agent/filter.py` → `INFERENCE_THRESHOLD` | 8 / 10 |
| Broader papers per run | `INFERENCE_MAX_PAPERS` | 3 |
| LLM provider | `LLM_PROVIDER` or available key | Anthropic → OpenAI → Gemini → Groq |
| LLM model | `LLM_MODEL` or provider default | Provider-specific |
| Abstract characters | `ABSTRACT_MAX_CHARS` | Full abstract; set a positive integer to cap |
| LLM output tokens | `LLM_MAX_TOKENS` | 512 |
| Papers per LLM request | `LLM_BATCH_SIZE` | 8 for OpenAI-compatible endpoints; 1 otherwise |
| Groq delay | `GROQ_DELAY_SEC` | 3.5 seconds |
| Groq request cap | `GROQ_MAX_PAPERS` | 100 per run; remaining papers are deferred |
| Schedule | `.github/workflows/daily_papers.yml` | 08:00 and 20:00 UTC |
| Weekly recap | `.github/workflows/weekly_digest.yml` | Monday 08:30 UTC |

OpenAI-compatible endpoints use small batches to reduce request overhead while still sending every
full abstract. Set `LLM_BATCH_SIZE=1` to restore one-paper requests.

GLM 5.x requests disable thinking so the model spends its output budget on the score JSON. When
`LLM_MODEL` contains `nemotron-3-super`, Paperbot uses NVIDIA's recommended
`temperature=1.0` and `top_p=0.95` and disables thinking for the first-stage
classifier. This preserves the output budget for the required JSON result.

## Roadmap

See [docs/V2_ROADMAP.md](docs/V2_ROADMAP.md) for the proposed evaluation set, two-stage deep-reading pipeline, searchable historical archive, Slack feedback loop, and deployment plan.

## License

MIT
