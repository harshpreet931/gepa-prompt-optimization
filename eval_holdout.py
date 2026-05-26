"""Held-out programmatic evaluation: does the GEPA-optimized CLAUDE.md actually
produce different behavior than the seed CLAUDE.md, or did it just earn judge
points by being elaborate?

We run both prompts on 12 brand-new scenarios (none used in optimization),
and grade each response with rule-based checks — no LLM in the loop.

Task LM is the same one GEPA used (`nemotron`) so we're measuring whether
GEPA actually changed `nemotron`'s behavior, not whether the prompt would
work on Claude (it might not — that's a separate question).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv

load_dotenv()

import litellm


HERE = Path(__file__).parent
RUN_DIR = HERE / "eval_holdout_runs"
RUN_DIR.mkdir(exist_ok=True)

TASK_LM = os.environ["GEPA_TASK_LM"]


def _read_text(p: Path) -> str:
    return p.read_text()


SEED_PROMPT = """# CLAUDE.md

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

OPTIMIZED_PROMPT = _read_text(HERE / "runs_guidelines2" / "best_claude_md.md")


# ---- Check helpers (response -> bool; True = behavior present) -------------

def has_question(r: str) -> bool:
    return "?" in r


def has_word(*words: str) -> Callable[[str], bool]:
    pat = re.compile(r"|".join(re.escape(w) for w in words), re.I)
    return lambda r: bool(pat.search(r))


def re_has(pattern: str) -> Callable[[str], bool]:
    p = re.compile(pattern, re.I)
    return lambda r: bool(p.search(r))


def absent(*words: str) -> Callable[[str], bool]:
    pat = re.compile(r"|".join(re.escape(w) for w in words), re.I)
    return lambda r: not pat.search(r)


def no_regex(pattern: str) -> Callable[[str], bool]:
    p = re.compile(pattern, re.I)
    return lambda r: not p.search(r)


def max_chars(n: int) -> Callable[[str], bool]:
    return lambda r: len(r) < n


def max_code_lines(n: int) -> Callable[[str], bool]:
    def f(r: str) -> bool:
        # crude: count lines inside triple-backtick fences
        in_fence = False
        count = 0
        for line in r.splitlines():
            if line.strip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                count += 1
        # if no fences, count any line that looks like code
        if count == 0:
            for line in r.splitlines():
                if line.startswith(("def ", "class ", "    ", "import ", "from ")):
                    count += 1
        return count < n
    return f


@dataclass
class Check:
    name: str
    fn: Callable[[str], bool]


@dataclass
class Scenario:
    id: str
    category: str
    input: str
    checks: list[Check]


# ---- Held-out scenarios (NOT in optimize_guidelines.SCENARIOS) -------------

SCENARIOS: list[Scenario] = [
    # Think Before Coding — should ask or surface ambiguity
    Scenario("T1", "TBC", "Build a URL shortener.", [
        Check("asks_a_question", has_question),
        Check("mentions_scope_or_storage", re_has(r"scope|storage|database|persist|how (long|many)|format|requirement")),
        Check("not_a_huge_implementation", max_chars(2000)),
    ]),
    Scenario("T2", "TBC", "Add login to my app.", [
        Check("asks_a_question", has_question),
        Check("mentions_auth_type", re_has(r"auth|password|oauth|session|jwt|cookie|sso|magic")),
        Check("not_a_huge_implementation", max_chars(2000)),
    ]),
    Scenario("T3", "TBC",
        "Make this faster:\n```python\ndef lookup(items, x):\n    for i in items:\n        if i == x: return True\n    return False\n```",
        [
            Check("asks_or_pushes_back", lambda r: has_question(r) or bool(re.search(r"already|sufficient|profile|measure|context|set\b|dict\b", r, re.I))),
            Check("not_a_huge_rewrite", max_chars(800)),
        ]),

    # Simplicity First — should produce minimal code
    Scenario("S1", "SIM", "Write a function to title-case a string.", [
        Check("short_response", max_chars(600)),
        Check("no_try_except", no_regex(r"\btry\s*:|\bexcept\b")),
        Check("no_type_hints", no_regex(r"->\s*\w+\s*:|def\s+\w+\s*\([^)]*:\s*\w+")),
    ]),
    Scenario("S2", "SIM", "Add a helper to clamp a number between 0 and 100.", [
        Check("short_code", max_code_lines(8)),
        Check("no_input_validation", no_regex(r"isinstance|raise\s+(TypeError|ValueError)|assert\s+isinstance")),
        Check("no_docstring", no_regex(r'"""|\'\'\'')),
    ]),
    Scenario("S3", "SIM", "Implement is_even(n).", [
        Check("super_short", max_chars(300)),
        Check("no_overengineering", no_regex(r"isinstance|TypeError|class\s+\w+|@\w+|abstract")),
    ]),

    # Surgical Changes — should preserve surrounding artifacts
    Scenario("C1", "SUR",
        "Fix the off-by-one bug on the marked line:\n```python\ndef first_n(lst, n):\n    print('debug: called', lst, n)  # debug\n    return lst[:n+1]  # BUG on this line\n```",
        [
            Check("preserves_print_call", re_has(r"print\(\s*['\"]debug")),
            Check("preserves_debug_comment", re_has(r"#\s*debug")),
            Check("not_a_huge_rewrite", max_chars(800)),
        ]),
    Scenario("C2", "SUR",
        "Add a return type hint of `int` to this function. Change NOTHING else.\n```python\ndef total(xs):\n    return sum(xs)\n```",
        [
            Check("preserves_body", re_has(r"return\s+sum\(\s*xs\s*\)")),
            Check("no_docstring_added", no_regex(r'"""|\'\'\'')),
            Check("no_param_hint_added", no_regex(r"xs\s*:\s*\w+")),
        ]),
    Scenario("C3", "SUR",
        "Rename the variable `foo` to `bar` in this function. Leave everything else alone.\n```python\ndef process(items):\n    foo = []  # collect results\n    for x in items:\n        foo.append(x * 2)\n    return foo\n```",
        [
            Check("preserves_collect_comment", re_has(r"#\s*collect\s+results")),
            Check("no_typehints_added", no_regex(r"items\s*:\s*\w+|->\s*\w+\s*:")),
            Check("does_rename", lambda r: "bar" in r),
        ]),

    # Goal-Driven Execution — should set criteria / propose tests
    Scenario("G1", "GOAL", "Add validation to user_signup(email, password).", [
        Check("asks_or_states_criteria", has_question),
        Check("mentions_rules_or_criteria", re_has(r"valid|criteria|rule|require|expect|format")),
    ]),
    Scenario("G2", "GOAL",
        "Fix the off-by-one bug:\n```python\ndef last_n(lst, n):\n    return lst[-n-1:]\n```",
        [
            Check("mentions_test_or_assert", re_has(r"\btest\b|\bassert\b|\bexpect\b|\bverify\b|reproduc")),
            Check("not_a_huge_rewrite", max_chars(800)),
        ]),
    Scenario("G3", "GOAL", "Improve robustness of fetch_user(id).", [
        Check("asks_a_question", has_question),
        Check("clarifies_meaning", re_has(r"timeout|retry|error|robust|fail|cache|valid|what (kind|exactly|do)")),
    ]),
]


# ---- Runner ----------------------------------------------------------------

def run_once(prompt: str, scenario_input: str) -> str:
    try:
        r = litellm.completion(
            model=TASK_LM,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": scenario_input},
            ],
            temperature=0,
            timeout=90,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:  # noqa: BLE001
        return f"[ERROR: {type(e).__name__}: {str(e)[:120]}]"


def evaluate(label: str, prompt: str) -> dict:
    print(f"\n--- evaluating {label} ({len(prompt)} chars) ---")
    results = []
    for sc in SCENARIOS:
        t0 = time.time()
        resp = run_once(prompt, sc.input)
        dt = time.time() - t0
        passed = {c.name: bool(c.fn(resp)) for c in sc.checks}
        results.append({
            "id": sc.id,
            "category": sc.category,
            "response_chars": len(resp),
            "elapsed_s": round(dt, 1),
            "checks": passed,
            "response": resp,
        })
        n_pass = sum(passed.values())
        n_total = len(passed)
        print(f"  [{sc.id}] {sc.category:5} {n_pass}/{n_total}  {dt:5.1f}s  {len(resp):4}ch")
    return {"label": label, "prompt_chars": len(prompt), "results": results}


def summarize(seed: dict, opt: dict) -> None:
    print("\n" + "=" * 64)
    print("HELD-OUT EVALUATION: seed CLAUDE.md vs GEPA-optimized")
    print("=" * 64)
    print(f"Task LM           : {TASK_LM}")
    print(f"Scenarios         : {len(SCENARIOS)}")
    print(f"Seed prompt       : {seed['prompt_chars']:,} chars")
    print(f"Optimized prompt  : {opt['prompt_chars']:,} chars  ({opt['prompt_chars']/seed['prompt_chars']:.1f}x)")

    # by-category
    print(f"\n{'Category':10} {'Checks':>8} {'SEED':>10} {'OPTIM':>10} {'Δ':>6}")
    cat_totals: dict[str, dict] = {}
    for sr, orr in zip(seed["results"], opt["results"], strict=True):
        cat = sr["category"]
        c = cat_totals.setdefault(cat, {"total": 0, "seed_pass": 0, "opt_pass": 0})
        c["total"] += len(sr["checks"])
        c["seed_pass"] += sum(sr["checks"].values())
        c["opt_pass"] += sum(orr["checks"].values())
    overall = {"total": 0, "seed_pass": 0, "opt_pass": 0}
    for cat, c in cat_totals.items():
        overall["total"] += c["total"]
        overall["seed_pass"] += c["seed_pass"]
        overall["opt_pass"] += c["opt_pass"]
        d = c["opt_pass"] - c["seed_pass"]
        sign = "+" if d > 0 else ""
        print(f"{cat:10} {c['total']:>8} {c['seed_pass']:>4}/{c['total']:<5} {c['opt_pass']:>4}/{c['total']:<5} {sign}{d:>5}")
    d = overall["opt_pass"] - overall["seed_pass"]
    sign = "+" if d > 0 else ""
    print(f"{'OVERALL':10} {overall['total']:>8} {overall['seed_pass']:>4}/{overall['total']:<5} {overall['opt_pass']:>4}/{overall['total']:<5} {sign}{d:>5}")

    # response length
    def med(rs):
        xs = sorted(r["response_chars"] for r in rs)
        return xs[len(xs)//2]
    print(f"\nResponse length (median chars):")
    print(f"  SEED       : {med(seed['results']):>5}")
    print(f"  OPTIMIZED  : {med(opt['results']):>5}")

    # specific divergences
    print(f"\nDivergences (per check, SEED vs OPTIMIZED):")
    for sr, orr in zip(seed["results"], opt["results"], strict=True):
        for cname, sv in sr["checks"].items():
            ov = orr["checks"][cname]
            if sv != ov:
                marker = "OPTIM gains" if ov and not sv else "OPTIM regresses"
                print(f"  {sr['id']:3} {sr['category']:4} {cname:32}  seed={int(sv)} opt={int(ov)}  ← {marker}")


def main() -> None:
    seed = evaluate("SEED", SEED_PROMPT)
    opt = evaluate("OPTIMIZED", OPTIMIZED_PROMPT)

    (RUN_DIR / "seed.json").write_text(json.dumps(seed, indent=2))
    (RUN_DIR / "optimized.json").write_text(json.dumps(opt, indent=2))
    summarize(seed, opt)
    print(f"\nFull responses saved -> {RUN_DIR}/")


if __name__ == "__main__":
    main()
