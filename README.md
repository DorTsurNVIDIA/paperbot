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
| `SLACK_WEBHOOK_URL` | The webhook URL from step 1 |

The agent uses the first key it finds (Anthropic → OpenAI → Gemini). To force a provider, add `LLM_PROVIDER` as a repository **variable** (Actions → Variables): `anthropic`, `openai`, or `gemini`.

### 4. Push to GitHub

The workflow runs automatically at **8am UTC** every day. You can also trigger it manually via **Actions → Daily Papers Agent → Run workflow**.

### How you'll know it works

- **Without Slack:** Run the workflow manually (**Actions → Daily Papers Agent → Run workflow**). Open the run and click the **run-agent** job. If the "Run papers agent" step is green and the log shows lines like `Total fetched: N papers`, `New (unseen) papers: M`, `Relevant papers after LLM filter: K`, then fetch, dedup, and LLM filtering are working. You can add the Slack webhook later.
- **With Slack:** After you add `SLACK_WEBHOOK_URL`, the next run will post to the channel you chose: either a "Daily LLM Inference & Speculative Decoding Papers" message with each paper's title, link, and summary, or "No new … papers found today" if none passed the filter.

## Local testing

```bash
# Use one of these (or set LLM_PROVIDER=openai / gemini to force):
export OPENAI_API_KEY=sk-...        # ChatGPT
# export GEMINI_API_KEY=...         # Gemini (get at https://aistudio.google.com/apikey)
# export ANTHROPIC_API_KEY=sk-ant-... # Claude
export SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...

pip install -r requirements.txt
python -m agent.main
```

## Configuration

| Setting | Location | Default |
|---|---|---|
| Lookback window | `agent/fetch.py` → `LOOKBACK_HOURS` | 48 hours |
| Relevance threshold | `agent/filter.py` → `RELEVANCE_THRESHOLD` | 6 / 10 |
| LLM provider | env `LLM_PROVIDER` or first key set | anthropic → openai → gemini |
| LLM model | env `LLM_MODEL` or per-provider default | claude-haiku-4-5-20251001 / gpt-4o-mini / gemini-1.5-flash |
| Cron schedule | `.github/workflows/daily_papers.yml` | `0 8 * * *` (8am UTC) |

## License

MIT
