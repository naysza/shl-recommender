# Approach: SHL Assessment Recommender

## Design

The service is a stateless FastAPI app with two endpoints: `GET /health` and
`POST /chat`. Each `/chat` request carries the full conversation history, so the
server does not store per-user state.

The agent uses a two-step flow:

1. A routing call extracts the current hiring requirements from the whole
   conversation and chooses one action: clarify, recommend, refine, compare, or
   refuse.
2. For recommendation/refinement, the service retrieves matching catalog items
   and gives only those candidates to the response-generation call. The model is
   not allowed to invent names or URLs; recommendations are filtered again in
   code before returning.

Comparison requests are handled by looking up assessment names directly in the
loaded catalog, then generating a grounded comparison from those catalog fields.
Obvious off-topic and prompt-injection requests are refused before calling the
model.

## Catalog and Retrieval

The catalog loader fetches SHL product data, caches it locally, and falls back
to a bundled fixture if the network is unavailable. The assignment scope is
Individual Test Solutions, so entries ending in `Solution` or `Solutions` are
excluded because those are typically pre-packaged job solutions.

Retrieval uses BM25 over assessment name, description, job level, duration, and
test category. I chose BM25 because the catalog is small and the queries often
contain literal skill names such as Java, SQL, communication, or OPQ. It is fast,
cheap to run on a free host, and easy to inspect. A small synonym map handles
common wording differences such as `stakeholder` and `communication`.

## Prompting and Guardrails

The routing prompt is responsible for conversation behavior:

- clarify vague first turns instead of recommending immediately
- preserve earlier requirements when the user refines constraints
- compare only named catalog assessments
- refuse legal, HR-policy, salary, job-posting, and prompt-injection requests

The final recommendation prompt receives a closed candidate list and must return
the required schema. The code then removes any recommendation whose URL is not
present in the catalog, which keeps responses catalog-grounded even if the model
misbehaves.

## Evaluation

I added unit tests for guardrails, catalog normalization, retrieval, duration
filtering, fuzzy name lookup, and URL validation. I also smoke-tested the API
locally:

- `/health` returns `{"status":"ok"}`
- `/chat` returns the required schema
- prompt-injection requests return no recommendations
- the full catalog cache loads 370 filtered Individual Test Solutions

The next tuning step would be to run the public trace replay script against the
deployed endpoint and add synonyms from any Recall@10 misses.

## AI Tool Usage

I used AI coding assistance to scaffold parts of the FastAPI app, prompts,
tests, and documentation. I reviewed and adjusted the implementation, including
the catalog parsing fix, retrieval behavior, guardrails, and submission notes.
