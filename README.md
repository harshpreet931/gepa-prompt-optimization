# GEPA prompt optimization against the local Grid AI proxy

[GEPA](https://github.com/gepa-ai/gepa) optimizes a system prompt by running
rollouts, reflecting on traces with an LLM, and mutating the candidate. This
project wires it up against the local `gridAIloaddistributionserver` (which
round-robins API keys to `grid.ai.juspay.net`).

## How the pieces fit

```
run_gepa.py
  └── gepa.optimize(task_lm="openai/nemotron", reflection_lm="openai/deepseek")
        └── litellm.completion(model="openai/<name>")
              └── HTTP POST http://localhost:3456/v1/chat/completions   (OPENAI_API_BASE)
                    └── gridAIloaddistributionserver overwrites Authorization
                          └── https://grid.ai.juspay.net
```

Because the proxy **overwrites** the `Authorization` header (server.js:27),
the `OPENAI_API_KEY` LiteLLM sends is irrelevant — any non-empty string works.
What matters is `OPENAI_API_BASE` pointing at the proxy.

## Setup

1.  Start the load balancer in the other repo:
    ```bash
    cd ~/Desktop/Projects/gridAIloaddistributionserver
    API_KEYS=sk-xxxx,sk-yyyy npm start
    ```
2.  In this repo, copy the env template and install deps:
    ```bash
    cp .env.example .env
    uv sync
    ```
3.  Smoke-test the path end-to-end:
    ```bash
    uv run python smoke_test.py
    ```
    Expected: `content: '156'` for both models.
4.  Run the optimizer:
    ```bash
    uv run python run_gepa.py
    ```
    Artifacts land in `runs/` — `best_candidate.json` and GEPA's run-state.

## What the default demo does

`dataset.py` defines an 18-example payment-intent classification task with six
labels (`refund`, `dispute`, `payment_failed`, `card_help`, `subscription`,
`other`). The seed prompt is intentionally weak:

```
Classify the customer's message. Reply in the format '### <label>'.
```

`gepa.optimize` runs the prompt on the trainset, feeds failed responses (with
the per-example `explanation` from `additional_context`) into the reflection
LM, and mutates the prompt to fix the failures. Scoring is substring match:
`data["answer"] in response`.

You should see the validation score climb across iterations as GEPA learns to
list the labels, anchor edge cases, and lock in the `### <label>` format.

## Picking models

The proxy budget blocks **external paid models** (Claude, Gemini, etc.) on
this account. Only free internal models work. Check what's live:

```bash
curl -s http://localhost:3456/v1/models | jq -r '.data[].id'
```

Verified-working free models for **fast** tasks (≤ ~5s per rollout):

| Model                  | Notes                                     |
|------------------------|-------------------------------------------|
| `openai/nemotron`      | Default task LM — short, accurate answers |
| `openai/deepseek`      | Default reflection LM — strong diagnoses  |
| `openai/kimi-latest`   | Similar to deepseek                       |
| `openai/open-fast`     | Reasoning model — needs ≥2048 max_tokens  |
| `openai/minimaxai/minimax-m2` | Reasoning model — same caveat       |

Override per run:
```bash
GEPA_TASK_LM=openai/kimi-latest uv run python run_gepa.py
```

### ⚠️ Reasoning models hang on hard tasks

All free models on this proxy route through **reasoning** backends. On easy
prompts they answer in 1–2s. On AIME-class math problems, individual rollouts
can take 4+ minutes upstream, and LiteLLM's per-call `timeout` parameter is
not honored end-to-end. `batch_completion` blocks on the slowest item, so a
single hung problem can stall the whole optimizer.

If you want to optimize for a heavy-reasoning task (deep math, multi-step
agents), expect to either:
- request a higher budget for an external paid model, or
- wrap `DefaultAdapter` to enforce a hard ThreadPool deadline (skipping slow
  problems with a 0-score), or
- pre-filter the dataset to remove problems that exceed a per-call budget.

## Pointing it at your own task

Replace `dataset.LABELLED` with your own list of dicts, each shaped like:

```python
{
    "input": "<user message>",
    "answer": "<substring the response must contain to count as correct>",
    "additional_context": {"<key>": "<extra info shown to reflection LM on failure>"},
}
```

`DefaultAdapter`'s `ContainsAnswerEvaluator` uses substring scoring. If you
need exact-match, regex, numeric tolerance, or LLM-as-judge, write a custom
`Evaluator` and pass it via `DefaultAdapter(model=..., evaluator=...)`, then
pass `adapter=your_adapter` to `gepa.optimize` — see
[gepa/adapters](https://github.com/gepa-ai/gepa/tree/main/src/gepa/adapters).

## Budget knobs

- `GEPA_MAX_METRIC_CALLS` — total rollouts allowed (default 60). Each call is
  one model rollout on one example. Set higher for stronger optimization.
- Dataset size — bigger sets are more robust but consume budget faster
  (initial eval costs `len(trainset)+len(valset)` rollouts).

## Troubleshooting

- **`budget_exceeded`** — you picked a paid model (Claude, Gemini). Switch
  to a free internal one (see table above).
- **Empty `content`, full `reasoning_content`** — reasoning model truncated
  before producing the final answer. The bundled task uses short answers so
  this rarely fires, but if you swap tasks bump `max_tokens` in the adapter.
- **Optimizer hangs at 0 rollouts** — your task is too hard for the free
  reasoning models; see the warning above.
- **`Connection refused`** — the load-balancer Node process isn't running.
