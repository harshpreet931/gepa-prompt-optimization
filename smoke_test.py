"""Verify LiteLLM -> local proxy -> grid.ai.juspay.net path before running GEPA."""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

import litellm

TASK_LM = os.environ.get("GEPA_TASK_LM", "openai/nemotron")
REFLECTION_LM = os.environ.get("GEPA_REFLECTION_LM", "openai/deepseek")


def ping(model: str) -> None:
    print(f"\n--- {model} ---")
    resp = litellm.completion(
        model=model,
        messages=[
            {"role": "system", "content": "Answer with the literal integer only."},
            {"role": "user", "content": "What is 12 * 13?"},
        ],
        max_tokens=512,
        temperature=0,
    )
    msg = resp.choices[0].message
    content = (msg.content or "").strip()
    reasoning = getattr(msg, "reasoning_content", None) or ""
    usage = getattr(resp, "usage", None)
    print(f"content : {content!r}")
    if reasoning:
        print(f"reasoning: {reasoning[:120]!r}{'…' if len(reasoning) > 120 else ''}")
    if usage:
        print(f"tokens  : {usage}")
    if "156" not in content:
        print("WARN: expected 156 in content — model may be cutting off before answer.")


def main() -> int:
    base = os.environ.get("OPENAI_API_BASE") or os.environ.get("OPENAI_BASE_URL")
    if not base:
        print("ERROR: OPENAI_API_BASE not set. Did you copy .env.example to .env?")
        return 1
    print(f"OPENAI_API_BASE = {base}")

    for model in (TASK_LM, REFLECTION_LM):
        try:
            ping(model)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED on {model}: {type(exc).__name__}: {exc}")
            return 2
    print("\nOK: both models reachable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
