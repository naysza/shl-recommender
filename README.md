# SHL Assessment Recommender

A conversational agent that takes a hiring manager from a vague intent
("I'm hiring a Java developer") to a grounded shortlist of SHL Individual
Test Solutions, via dialogue. Built for the SHL Labs AI Intern take-home.

## How it works (short version)

```
POST /chat
   │
   ▼
1. Deterministic guardrails (regex) ── catches prompt injection / off-topic
   before spending an LLM call
   │
   ▼
2. Stage-1 LLM call (Gemini, JSON mode) ── classifies the turn as
   clarify / recommend / refine / compare / refuse, and extracts
   structured requirements from the WHOLE conversation history
   │
   ▼
3a. clarify           -> ask one question, no retrieval
3b. recommend/refine  -> BM25 search over the real SHL catalog using the
                          extracted requirements -> Stage-2 LLM call that
                          may ONLY pick from the retrieved candidates
3c. compare           -> direct catalog lookup by name (no LLM guessing),
                          grounded comparison from real descriptions
   │
   ▼
4. Hard hallucination filter: any recommendation whose URL isn't a real,
   known catalog entry is dropped, regardless of what the LLM produced
```

See `APPROACH.md` for the full design write-up and trade-offs (this is the
2-page document meant for submission).

## Project layout

```
app/
  main.py         FastAPI app: GET /health, POST /chat
  agent.py        Orchestration of the pipeline above
  catalog.py      Fetch/cache/filter the live SHL catalog
  retrieval.py    BM25 index + search
  llm.py          Gemini API wrapper (JSON mode, retries)
  guardrails.py   Regex-based injection/off-topic screen + hallucination filter
  prompts.py      All prompt templates
  models.py       Pydantic request/response schemas (matches the spec exactly)
  config.py       Env-var driven settings
data/
  catalog_sample.json   small offline fixture (real SHL entries) for tests
tests/                   pytest suite
scripts/replay_trace.py  local evaluation harness against the 10 sample traces
```

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit .env and add your GEMINI_API_KEY
```

Get a free Gemini API key at https://aistudio.google.com/apikey (no credit
card required for the free tier).

Run it:

```bash
export $(cat .env | xargs)   # or use python-dotenv / your shell's own method
uvicorn app.main:app --reload
```

Then:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I need an assessment for a mid-level Java developer who works closely with stakeholders"}]}'
```

The first request after startup fetches the live SHL catalog (cached to
`data/catalog_cache.json` afterward, refreshed every 24h). This needs
outbound internet access - it works locally and on Render, but will NOT
work in a fully offline environment. If the live fetch fails and no cache
exists yet, the service falls back to the small bundled fixture in
`data/catalog_sample.json` so `/health` and `/chat` still respond, just
with far fewer assessments to recommend from.

## Tests

```bash
pip install pytest
pytest tests/ -v
```

`tests/test_guardrails.py` needs no external deps and always runs.
`tests/test_catalog_and_retrieval.py` needs `rank_bm25` (in requirements.txt).

## Evaluating against the sample conversation traces

Download the 10 sample traces from the link in the assignment PDF
(`sample_conversations.zip`), unzip them, then:

```bash
python scripts/replay_trace.py --traces-dir ./sample_conversations --api-url http://localhost:8000 --verbose
```

This uses Gemini to play the "user" side (persona + facts from each trace)
and talks to your running `/chat` endpoint for real, multi-turn, the same
way SHL's evaluator will. It reports Recall@10 per trace and a schema
pass rate. Run `--dry-run` first on one trace file to confirm the field
names match what `load_trace()` expects - the exact JSON keys in the zip
weren't visible ahead of time, so that function may need a one-line tweak.

## Deploying to Render (free tier)

1. Push this repo to GitHub.
2. In Render: New -> Blueprint -> point at the repo (uses `render.yaml`).
3. Set the `GEMINI_API_KEY` env var in the Render dashboard (marked
   `sync: false` in the blueprint so it's not committed).
4. Deploy. First request after a cold start can take up to ~60s (free tier
   spin-up + catalog fetch) - well within the assignment's "first /health
   call gets up to 2 minutes" allowance.

Your public endpoints will be:
- `https://<your-service>.onrender.com/health`
- `https://<your-service>.onrender.com/chat`

## Known limitations / honest caveats

- The catalog JSON has no explicit "Individual Test Solutions" vs
  "Pre-packaged Job Solutions" flag. We filter by a name-suffix heuristic
  (`app/catalog.py::_is_packaged_solution`) - verified by spot-checking
  against the live product catalog page, but worth a second look if Recall@10
  looks off for role-bundle-style queries.
- Retrieval is BM25 + light synonym expansion, not embeddings - see
  APPROACH.md for why, and where it would break (heavy paraphrase queries).
- The Stage-1 classifier occasionally needs a nudge on ambiguous refine vs.
  new-request turns; `conversation_complete` is intentionally conservative
  (defaults to false) per the spec's own example response.
