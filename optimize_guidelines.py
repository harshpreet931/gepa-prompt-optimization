"""Optimize a CLAUDE.md-style coding-guidelines prompt with an LLM-as-judge.

Run:
    uv run python optimize_guidelines.py

The seed prompt is your CLAUDE.md. The dataset is a set of coding scenarios
designed to surface the anti-patterns the guidelines try to prevent. A judge
LM grades each response against the guidelines and returns a structured
critique; GEPA's reflection LM reads those critiques and mutates the prompt.

Caveats — these are real, not just academic:
- The judge has noise (~0.1-0.2 score variance run-to-run).
- The judge may have its own bias about what "good code behavior" means.
- GEPA can reward-hack: produce prompts that score well on the judge but
  don't actually change the consuming LLM's real-world behavior.
- The task LM here is `nemotron` (free, fast). If you ship the optimized
  prompt to Claude Code, re-validate it against Claude — instruction-following
  varies between models.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import litellm

import gepa
from gepa.adapters.default_adapter.default_adapter import DefaultAdapter, EvaluationResult
from gepa.core.adapter import EvaluationBatch


HERE = Path(__file__).parent
RUN_DIR = HERE / os.environ.get("GEPA_RUN_DIR", "runs_guidelines")
RUN_DIR.mkdir(exist_ok=True)
SEED_FROM = os.environ.get("GEPA_SEED_FROM")  # path to a candidates.json to bootstrap from

TASK_LM = os.environ["GEPA_TASK_LM"]
REFLECTION_LM = os.environ["GEPA_REFLECTION_LM"]
JUDGE_LM = os.environ.get("GEPA_JUDGE_LM", REFLECTION_LM)
MAX_METRIC_CALLS = int(os.environ.get("GEPA_GUIDELINES_BUDGET", "120"))


SEED_PROMPT = """\
# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" -> "Write tests for invalid inputs, then make them pass"
- "Fix the bug" -> "Write a test that reproduces it, then make it pass"
- "Refactor X" -> "Ensure tests pass before and after"

For multi-step tasks, state a brief plan.

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
"""

GUIDELINES_TEXT_FOR_JUDGE = SEED_PROMPT  # what the judge grades against


# ---- Scenarios: each one is a vague-on-purpose coding request that should
# trigger guideline-aligned behavior (clarify, simplify, scope, verify).

SCENARIOS: list[dict] = [
    {
        "input": "Add a function `parse_amount(s)` that takes a string and returns a float.",
        "additional_context": {
            "expected_behavior": (
                "Ask what input formats matter (currency symbols? commas? negatives?). "
                "Or pick the simplest defensible interpretation and state it. "
                "Reject: a 30-line parser supporting every possible currency format."
            )
        },
    },
    {
        "input": (
            "Fix this bug:\n\n"
            "```python\ndef discount(price, pct):\n    return price - price * pct\n```\n\n"
            "It returns wrong values."
        ),
        "additional_context": {
            "expected_behavior": (
                "Ask what 'wrong' means or what the expected output is — `pct` could be 0.1 or 10. "
                "Or state the assumption and write a one-line fix plus a reproducing assert. "
                "Reject: changing the function shape, renaming params, adding type hints."
            )
        },
    },
    {
        "input": (
            "Refactor this for readability:\n\n"
            "```python\ndef avg(xs):\n    s = 0\n    n = 0\n    for x in xs:\n        s += x\n        n += 1\n    return s / n\n```"
        ),
        "additional_context": {
            "expected_behavior": (
                "One-line `return sum(xs) / len(xs)` if the request really is readability. "
                "Or push back: this is already readable; refactor for what reason? "
                "Reject: adding type hints, docstrings, empty-list guards, or renames not requested."
            )
        },
    },
    {
        "input": "Add error handling to `process_payment(amount, card)`.",
        "additional_context": {
            "expected_behavior": (
                "Ask: handle what errors? log? raise? return None? retry? "
                "Reject: silently picking try/except Exception with a print() fallback."
            )
        },
    },
    {
        "input": "Add a config loader for the app.",
        "additional_context": {
            "expected_behavior": (
                "Ask: format (env / yaml / json / toml)? required keys? defaults? validation? "
                "Reject: a 60-line ConfigManager class with schema validation that wasn't asked for."
            )
        },
    },
    {
        "input": (
            "Rename the variable `tmp` to `result` in this function — leave everything else alone:\n\n"
            "```python\ndef compute(xs):\n    tmp = sum(xs)  # accumulator\n    tmp = tmp * 2\n    print('debug:', tmp)\n    return tmp\n```"
        ),
        "additional_context": {
            "expected_behavior": (
                "Rename `tmp` -> `result` in all 4 places. Nothing else. "
                "Reject: removing the print, removing the comment, changing `tmp * 2` style, adding type hints."
            )
        },
    },
    {
        "input": "Make `fetch_user(id)` more robust.",
        "additional_context": {
            "expected_behavior": (
                "Ask what 'robust' means here — retries? timeouts? validation? caching? "
                "Reject: adding all of those silently because they're 'general best practice'."
            )
        },
    },
    {
        "input": "Add a CLI flag `--verbose` that increases logging.",
        "additional_context": {
            "expected_behavior": (
                "A minimal argparse/click addition plus one logging-level change. Maybe 5-10 lines. "
                "Reject: setting up structured logging, log rotation, env-var overrides, or test scaffolding."
            )
        },
    },
    {
        "input": "Write tests for `normalize_phone(s)`.",
        "additional_context": {
            "expected_behavior": (
                "Ask what `normalize_phone` is supposed to do — without the spec, tests are just guesses. "
                "Or state the assumed behavior explicitly and write 3-4 focused tests. "
                "Reject: writing 20 tests for behaviors that may not exist."
            )
        },
    },
    {
        "input": "Optimize this — it's slow:\n\n```python\ndef contains(items, x):\n    return x in items\n```",
        "additional_context": {
            "expected_behavior": (
                "Push back: this is already O(n) for a list and O(1) for a set/dict. Ask what `items` is, "
                "what scale the slowness shows up at, and what was profiled. "
                "Reject: rewriting it to use a Bloom filter or threading."
            )
        },
    },
    {
        "input": "Add caching to `expensive_compute(x)`.",
        "additional_context": {
            "expected_behavior": (
                "Ask: cache scope (process, disk, redis)? size? TTL? key strategy for non-hashable x? "
                "Or use `@functools.lru_cache` and state the trade-off (one line, in-process, unbounded by default)."
            )
        },
    },
    {
        "input": (
            "There's a bug on line 3 of this function — fix it:\n\n"
            "```python\ndef rotate(arr, k):\n    n = len(arr)\n    return arr[-k:] + arr[:-k]  # bug here\n```"
        ),
        "additional_context": {
            "expected_behavior": (
                "Identify the bug (k > n or k = 0 edge cases), fix only line 3 (e.g. `k %= n`), and write one "
                "reproducing test. "
                "Reject: rewriting using collections.deque, adding type hints, restructuring the function."
            )
        },
    },
]


def init_dataset() -> tuple[list[dict], list[dict]]:
    """Roughly 50/50 split."""
    mid = len(SCENARIOS) // 2
    return SCENARIOS[:mid], SCENARIOS[mid:]


# ---- Judge

JUDGE_SYSTEM = """\
You are an evaluator grading whether a coding assistant's response follows specific guidelines.

The guidelines:
---
{guidelines}
---

You will be given a user task, the expected behavior under the guidelines, and the assistant's response. \
Score 0.0 (ignores guidelines) to 1.0 (follows guidelines well). Be strict but fair: partial credit is fine.

Reply with ONLY a JSON object on a single line, no markdown fence:
{{"score": <float 0-1>, "violations": ["<short violation>", ...], "strengths": ["<short strength>", ...]}}
"""

JUDGE_USER_TMPL = """\
TASK:
{task}

EXPECTED BEHAVIOR UNDER GUIDELINES:
{expected}

ASSISTANT RESPONSE:
{response}
"""


def _extract_json(text: str) -> dict | None:
    """Pull the first {...} JSON object out of text. Tolerates fences and trailing prose."""
    if not text:
        return None
    # try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # find the first balanced {...}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


class GuidelinesJudge:
    def __init__(self, judge_lm: str, guidelines: str):
        self.judge_lm = judge_lm
        self.system = JUDGE_SYSTEM.format(guidelines=guidelines)

    def __call__(self, data: dict, response: str) -> EvaluationResult:
        user = JUDGE_USER_TMPL.format(
            task=data["input"],
            expected=data["additional_context"]["expected_behavior"],
            response=response or "[EMPTY RESPONSE]",
        )
        try:
            r = litellm.completion(
                model=self.judge_lm,
                messages=[{"role": "system", "content": self.system}, {"role": "user", "content": user}],
                temperature=0,
                timeout=60,
            )
            raw = (r.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            return EvaluationResult(
                score=0.0,
                feedback=f"[judge unavailable: {type(e).__name__}: {str(e)[:120]}]",
                objective_scores=None,
            )

        parsed = _extract_json(raw)
        if not parsed or "score" not in parsed:
            return EvaluationResult(
                score=0.0,
                feedback=f"[judge returned unparseable output, raw head: {raw[:300]!r}]",
                objective_scores=None,
            )

        try:
            score = max(0.0, min(1.0, float(parsed["score"])))
        except (TypeError, ValueError):
            score = 0.0
        violations = parsed.get("violations") or []
        strengths = parsed.get("strengths") or []
        feedback_parts = []
        if violations:
            feedback_parts.append("Violations: " + "; ".join(str(v) for v in violations))
        if strengths:
            feedback_parts.append("Strengths: " + "; ".join(str(s) for s in strengths))
        feedback_parts.append(f"Expected: {data['additional_context']['expected_behavior']}")
        return EvaluationResult(
            score=score,
            feedback=" || ".join(feedback_parts) or "[no critique]",
            objective_scores=None,
        )


class FaultTolerantAdapter(DefaultAdapter):
    """DefaultAdapter that doesn't crash when batch_completion returns an exception
    in place of a response (observed: BadGatewayError 502 from the proxy)."""

    def evaluate(self, batch, candidate, capture_traces=False):  # type: ignore[override]
        system_content = next(iter(candidate.values()))
        litellm_requests = [
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": d["input"]},
            ]
            for d in batch
        ]

        raw = self.litellm.batch_completion(
            model=self.model,
            messages=litellm_requests,
            max_workers=self.max_litellm_workers,
            **self.litellm_batch_completion_kwargs,
        )

        responses: list[str] = []
        for r in raw:
            try:
                responses.append((r.choices[0].message.content or "").strip())
            except AttributeError:
                responses.append(f"[ERROR: upstream returned {type(r).__name__}: {str(r)[:120]}]")

        outputs, scores, obj_scores = [], [], []
        trajectories = [] if capture_traces else None
        for data, assistant_response in zip(batch, responses, strict=True):
            res = self.evaluator(data, assistant_response)
            outputs.append({"full_assistant_response": assistant_response})
            scores.append(res.score)
            obj_scores.append(res.objective_scores)
            if trajectories is not None:
                trajectories.append(
                    {"data": data, "full_assistant_response": assistant_response, "feedback": res.feedback}
                )

        all_none = all(x is None for x in obj_scores)
        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=None if all_none else obj_scores,
        )


def _resolve_seed() -> str:
    if not SEED_FROM:
        return SEED_PROMPT
    cands = json.loads(Path(SEED_FROM).read_text())
    print(f"Bootstrapping seed from {SEED_FROM} -> last of {len(cands)} candidates")
    return cands[-1]["system_prompt"]


def main() -> None:
    trainset, valset = init_dataset()
    seed_prompt = _resolve_seed()
    print(f"trainset={len(trainset)} valset={len(valset)}")
    print(f"task_lm={TASK_LM} reflection_lm={REFLECTION_LM} judge_lm={JUDGE_LM}")
    print(f"max_metric_calls={MAX_METRIC_CALLS} run_dir={RUN_DIR}")
    print(f"seed prompt: {len(seed_prompt)} chars\n")

    judge = GuidelinesJudge(judge_lm=JUDGE_LM, guidelines=GUIDELINES_TEXT_FOR_JUDGE)
    adapter = FaultTolerantAdapter(
        model=TASK_LM,
        evaluator=judge,
        max_litellm_workers=4,
        litellm_batch_completion_kwargs={"num_retries": 2, "timeout": 90},
    )

    result = gepa.optimize(
        seed_candidate={"system_prompt": seed_prompt},
        trainset=trainset,
        valset=valset,
        adapter=adapter,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        run_dir=str(RUN_DIR),
        display_progress_bar=True,
        seed=0,
    )

    best = result.best_candidate
    out_path = RUN_DIR / "best_claude_md.md"
    out_path.write_text(best["system_prompt"])
    print("\n=== Best CLAUDE.md ===")
    print(best["system_prompt"])
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
