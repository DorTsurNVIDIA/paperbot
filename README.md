# paperbot

Paperbot is a scheduled research feed for **speculative decoding** and **LLM inference optimization**. It fetches recent papers from arXiv, Semantic Scholar, and Hugging Face, scores them with an LLM, and posts a ranked digest to Slack.

The feed has two explicit lanes:

- **SPECDEC** — speculative decoding is the main method, or the work directly enables it through drafting, verification, acceptance, multi-token prediction, or a closely related mechanism.
- **INFERENCE** — broader work on latency, throughput, memory, serving, or hardware efficiency.

Specdec papers are posted first. Papers within each lane are ordered by score, and every Slack item shows both scores plus controlled topic tags.

## How it works

1. **Fetch** papers from the last seven days.
2. **Canonicalize and deduplicate** identities across arXiv versions, Hugging Face, Semantic Scholar, and DOI metadata.
3. **Classify** each unseen paper with independent 1–10 specdec and inference scores.
4. **Rank and post** accepted papers to Slack, with the specdec lane first.
5. **Checkpoint safely**: rejected and successfully delivered papers are saved; failed, rate-limited, capped, or undelivered papers remain eligible for retry.

The twice-daily schedule catches papers missed by a delayed run without posting known papers again.

## Setup

### 1. Create a Slack Incoming Webhook

1. Go to [Slack apps](https://api.slack.com/apps) and create an app from scratch.
2. Enable **Incoming Webhooks**, add a webhook to the target channel, and copy its URL.
3. Add it to the repository as an Actions secret named `SLACK_WEBHOOK_URL`.

Incoming Webhooks are enough for publishing. A future interactive feedback loop based on emoji or buttons will require a Slack app with read permissions rather than only a webhook.

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

The scheduled workflow commits `seen_papers.json` back to the current branch.

1. Open **Settings → Actions → General**.
2. Under **Workflow permissions**, select **Read and write permissions**.
3. Keep GitHub's `actions/checkout` and `actions/setup-python` actions allowed by the repository or organization policy.

The production workflow uses a concurrency group so scheduled and manual runs cannot update the state file simultaneously.

### 5. Run it

The workflow runs at 08:00 and 20:00 UTC and can also be started from **Actions → Daily Papers Agent → Run workflow**. The manual `clear_seen_papers` option performs a one-run rescore of everything fetched in the current lookback window.

## Local development

```bash
python -m pip install -r requirements.txt

export LLM_PROVIDER=openai_compatible
export LLM_BASE_URL=https://your-service.example/v1
export LLM_MODEL=your-model-id
export LLM_API_KEY=your-service-key
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

python -m agent.main
python -m unittest discover -v
```

For a local scoring-only run, omit `SLACK_WEBHOOK_URL` and set `DRY_RUN=true`. Accepted papers are then treated as handled for that local state file, so use a disposable copy of `seen_papers.json` if you intend to deliver them later. Without a webhook or explicit dry-run flag, Paperbot fails closed instead of silently losing deliveries.

## Configuration

| Setting | Location | Default |
|---|---|---|
| Lookback window | `agent/fetch.py` → `LOOKBACK_HOURS` | 168 hours |
| Specdec threshold | `agent/filter.py` → `SPECDEC_THRESHOLD` | 6 / 10 |
| Inference threshold | `agent/filter.py` → `INFERENCE_THRESHOLD` | 7 / 10 |
| LLM provider | `LLM_PROVIDER` or available key | Anthropic → OpenAI → Gemini → Groq |
| LLM model | `LLM_MODEL` or provider default | Provider-specific |
| Abstract characters | `ABSTRACT_MAX_CHARS` | Full abstract; set a positive integer to cap |
| LLM output tokens | `LLM_MAX_TOKENS` | 512 |
| Groq delay | `GROQ_DELAY_SEC` | 3.5 seconds |
| Groq request cap | `GROQ_MAX_PAPERS` | 100 per run; remaining papers are deferred |
| Schedule | `.github/workflows/daily_papers.yml` | 08:00 and 20:00 UTC |

When `LLM_MODEL` contains `nemotron-3-super`, Paperbot uses NVIDIA's recommended
`temperature=1.0` and `top_p=0.95` and disables thinking for the first-stage
classifier. This preserves the output budget for the required JSON result.

## Roadmap

See [docs/V2_ROADMAP.md](docs/V2_ROADMAP.md) for the proposed evaluation set, two-stage deep-reading pipeline, searchable historical archive, Slack feedback loop, and deployment plan.

## License

MIT
