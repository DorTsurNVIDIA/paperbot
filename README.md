# paperbot

A GitHub Actions bot that runs daily, fetches new academic papers about **LLM inference efficiency** and **speculative decoding** from arXiv, Semantic Scholar, and Hugging Face, filters them with an LLM (Claude, ChatGPT, or Gemini), and posts summaries to a Slack channel.

## How it works

1. **Fetch** — pulls papers from the last 48 hours from three sources
2. **Dedup** — skips papers already posted (tracked in `seen_papers.json`)
3. **Filter** — uses an LLM to score relevance (1–10); only papers scoring ≥ 6 are kept
4. **Post** — sends formatted Slack messages via Incoming Webhook
5. **Save** — commits updated `seen_papers.json` back to the repo

## Setup

### 1. Create a Slack Incoming Webhook

1. Go to https://api.slack.com/apps → **Create New App** → From Scratch
2. Enable **Incoming Webhooks** → **Add New Webhook to Workspace** → choose a channel
3. Copy the generated webhook URL (`https://hooks.slack.com/services/...`)

### 2. Allow the workflow to push (GitHub authentication)

The agent commits updated `seen_papers.json` back to the repo. The workflow needs permission to push:

1. Open your repo on GitHub and go to **Settings** (tab next to Insights; you need admin access to see it).
2. In the **left sidebar**, click **Actions**, then **General**.  
   Direct URL: `https://github.com/DorTsurNVIDIA/paperbot/settings/actions`
3. **Workflow permissions** — select **Read and write permissions** (so the workflow can push `seen_papers.json`).
4. **Actions permissions** — leave **Allow all actions and reusable workflows**.  
   Do *not* choose "Allow DorTsurNVIDIA, and select non-DorTsurNVIDIA…" for this repo: that blocks GitHub's built-in actions (`actions/checkout`, `actions/setup-python`) unless you add them to an allow list. "Allow all actions" is the right choice here.
5. Click **Save**.

**If you don't see Settings or Actions:** You need admin/write access to the repo. For an org repo, an owner may have disabled Actions in **Organization Settings → Actions**. If your org blocks changing workflow permissions, use a Personal Access Token: create a PAT with `repo` scope, add it as a repository secret named `GH_PAT`, and we can switch the workflow to use it for the push step.

### 3. Add GitHub Secrets

In your repo: **Settings → Secrets and variables → Actions → New repository secret**

Add **one** LLM API key (whichever you have access to) and the Slack webhook:

| Secret name | Value |
|---|---|
| `ANTHROPIC_API_KEY` | Anthropic API key (Claude) |
| `OPENAI_API_KEY` | OpenAI API key (ChatGPT; use with gpt-4o-mini) |
| `GEMINI_API_KEY` | Google AI API key (Gemini; get one at [Google AI Studio](https://aistudio.google.com/apikey)) |
| `GROQ_API_KEY` | **Free tier** — Groq API key (Llama; get one at [console.groq.com](https://console.groq.com)) |
| `SLACK_WEBHOOK_URL` | The webhook URL from step 1 |

The agent uses the first key it finds (Anthropic → OpenAI → Gemini → Groq). To force a provider, add `LLM_PROVIDER` as a repository **variable** (Actions → Variables): `anthropic`, `openai`, `gemini`, or `groq`.

### 4. Push to GitHub

The workflow runs on a **schedule twice daily** (**08:00 UTC** and **20:00 UTC**) so a missed GitHub slot still usually gives you a run the same day. You can also trigger it manually via **Actions → Daily Papers Agent → Run workflow**. To process all fetched papers as new (e.g. after changing the LLM or for a one-time full run), check **Clear seen papers** when running the workflow — this run will treat every fetched paper as unseen, score with the LLM, post to Slack, then save the new seen list.

### If scheduled runs stop working

1. **Actions tab** — Open **Actions**. If you see a yellow banner that scheduled workflows were disabled (e.g. after ~60 days without repo activity on some plans), click to **re-enable** them.
2. **Settings → Actions → General** — Ensure **Allow all actions** (or your org’s equivalent) and **Read and write permissions** for workflows are still set.
3. **Default branch** — The schedule only uses the workflow file on your repo’s **default branch** (`main`). Merge changes there if you develop on another branch.
4. **Delays** — Scheduled jobs can start **up to ~1 hour late**; that’s normal on GitHub’s side.

### How you'll know it works

- **Without Slack:** Run the workflow manually (**Actions → Daily Papers Agent → Run workflow**). Open the run and click the **run-agent** job. If the "Run papers agent" step is green and the log shows lines like `Total fetched: N papers`, `New (unseen) papers: M`, `Relevant papers after LLM filter: K`, then fetch, dedup, and LLM filtering are working. You can add the Slack webhook later.
- **With Slack:** After you add `SLACK_WEBHOOK_URL`, the next run will post to the channel you chose: either a "Daily LLM Inference & Speculative Decoding Papers" message with each paper's title, link, and summary, or "No new … papers found today" if none passed the filter.

## Local testing

```bash
# Use one of these (or set LLM_PROVIDER=groq / openai / gemini to force):
export GROQ_API_KEY=gsk_...        # Free — Groq (https://console.groq.com)
# export OPENAI_API_KEY=sk-...      # ChatGPT
# export GEMINI_API_KEY=...         # Gemini (https://aistudio.google.com/apikey)
# export ANTHROPIC_API_KEY=sk-ant-... # Claude
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

pip install -r requirements.txt
python -m agent.main
```

## Configuration

| Setting | Location | Default |
|---|---|---|
| Lookback window | `agent/fetch.py` → `LOOKBACK_HOURS` | 7 days (168 hours) |
| Relevance threshold | `agent/filter.py` → `RELEVANCE_THRESHOLD` | 6 / 10 |
| LLM provider | env `LLM_PROVIDER` or first key set | anthropic → openai → gemini → groq |
| LLM model | env `LLM_MODEL` or per-provider default | … / gemini-2.0-flash / llama-3.1-8b-instant (Groq) |
| Groq delay | env `GROQ_DELAY_SEC` | 3s between requests |
| Groq max papers | env `GROQ_MAX_PAPERS` | 60 per run (avoids rate limit) |
| Abstract length (Groq) | env `ABSTRACT_MAX_CHARS` | 25000 chars (full abstract); others default 600 |
| LLM output tokens (Groq) | env `LLM_MAX_TOKENS` | 512; others default 128 |
| Cron schedule | `.github/workflows/daily_papers.yml` | `0 8 * * *` & `0 20 * * *` (08:00 & 20:00 UTC) |

## License

MIT
