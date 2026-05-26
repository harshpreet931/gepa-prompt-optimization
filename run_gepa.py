"""GEPA prompt-optimization run against the local Grid AI proxy.

Run:
    uv run python run_gepa.py

Defaults: payment-intent classification (fast, ~1s per rollout). To swap in
your own task, replace the `init_dataset` import below with a function that
returns (trainset, valset). See dataset.py for the expected item shape.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import gepa

from dataset import init_dataset


HERE = Path(__file__).parent
RUN_DIR = HERE / "runs"
RUN_DIR.mkdir(exist_ok=True)

TASK_LM = os.environ["GEPA_TASK_LM"]
REFLECTION_LM = os.environ["GEPA_REFLECTION_LM"]
MAX_METRIC_CALLS = int(os.environ["GEPA_MAX_METRIC_CALLS"])

SEED = {
    "system_prompt": (
        "Classify the customer's message. Reply in the format '### <label>'."
    )
}


def main() -> None:
    trainset, valset = init_dataset()
    print(f"trainset={len(trainset)} valset={len(valset)}")
    print(f"task_lm={TASK_LM} reflection_lm={REFLECTION_LM}")
    print(f"max_metric_calls={MAX_METRIC_CALLS}")
    print(f"seed prompt: {SEED['system_prompt']!r}\n")

    result = gepa.optimize(
        seed_candidate=SEED,
        trainset=trainset,
        valset=valset,
        task_lm=TASK_LM,
        reflection_lm=REFLECTION_LM,
        max_metric_calls=MAX_METRIC_CALLS,
        run_dir=str(RUN_DIR),
        display_progress_bar=True,
        seed=0,
    )

    best = result.best_candidate
    out_path = RUN_DIR / "best_candidate.json"
    out_path.write_text(json.dumps(best, indent=2))
    print("\n=== Best prompt ===")
    print(best["system_prompt"])
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
