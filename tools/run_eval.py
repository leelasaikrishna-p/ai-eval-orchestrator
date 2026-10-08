"""
run_eval -- a full, deterministic evaluation pass, for run-level drift.

The orchestrator is an agent: it skips checks once it has enough evidence,
so its per-run averages depend on what it chose to call and aren't
comparable run to run. This script runs ALL three LLM tools on EVERY
labeled answer, so two runs are always measured the same way, then
aggregates:

  avg_golden_similarity  mean golden_eval score (0-1)
  avg_groundedness       share of answers judged grounded (0-1)
  avg_judge_score        mean llm_judge score (0-10, computed from its breakdown)

The judge runs reference-guided (it sees the golden answer). A baseline
records the judge mode it was measured with, and a run is only compared
with a baseline from the same mode -- otherwise a "drift" would just be
the method changing.

Usage (PROVIDER / GEMINI_MODEL / OLLAMA_MODEL pick the model):
  python3 tools/run_eval.py --save-baseline   # record this model's baseline
  python3 tools/run_eval.py                   # compare a new run to it

Baselines live in data/baselines/, one per provider/model. Because the
candidate answers here are fixed, a drift flag means the EVALUATOR's
judgments changed (e.g. a provider silently updated a model). Pointed at a
live RAG bot's answers, the same check would catch the bot regressing.
With only 3 labeled answers, one answer changing moves an average by ~33%,
so treat results as a demonstration of the mechanism, not a statistical
signal.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_utils import BASELINE_DIR, baseline_path, load_baseline_scores, load_golden_dataset, load_sample_candidate_runs  # noqa: E402
from decision_policy import run_level_check  # noqa: E402
from llm_provider import get_provider  # noqa: E402
from tools.drift_check import drift_check  # noqa: E402
from tools.golden_eval import golden_eval  # noqa: E402
from tools.groundedness_check import groundedness_check  # noqa: E402
from tools.llm_judge import llm_judge  # noqa: E402

JUDGE_MODE = "reference"


def evaluate_all(provider) -> list[dict]:
    """Every tool on every labeled answer -- no agent, no short-circuits."""
    golden = load_golden_dataset()
    results = []
    for run in load_sample_candidate_runs():
        item = golden[run["id"]]
        g = golden_eval(item["question"], item["golden_answer"], run["candidate_answer"], provider)
        gr = groundedness_check(item["context"], run["candidate_answer"], provider)
        j = llm_judge(item["question"], run["candidate_answer"], provider, reference_answer=item["golden_answer"])
        results.append({
            "id": run["id"],
            "label": run["label"],
            "golden_score": g["score"],
            "grounded": bool(gr["grounded"]),
            "judge_score": j["score"],
        })
    return results


def aggregate(results: list[dict]) -> dict:
    """Pure function: per-answer results -> the three run-level averages."""
    n = len(results)
    if n == 0:
        raise ValueError("no results to aggregate")
    return {
        "avg_golden_similarity": round(sum(r["golden_score"] for r in results) / n, 4),
        "avg_groundedness": round(sum(1 for r in results if r["grounded"]) / n, 4),
        "avg_judge_score": round(sum(r["judge_score"] for r in results) / n, 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--save-baseline", action="store_true", help="record this run as the model's baseline")
    args = parser.parse_args(argv)

    provider = get_provider()
    model = getattr(provider, "model", None)
    print(f"Full evaluation pass: {provider.name}/{model}")
    results = evaluate_all(provider)
    for r in results:
        print(f"  [{r['id']}] {r['label']}: golden={r['golden_score']:.2f} "
              f"grounded={r['grounded']} judge={r['judge_score']:.1f}")
    metrics = aggregate(results)
    print(f"Run averages: {metrics}")

    if args.save_baseline:
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        record = {
            "provider": provider.name,
            "model": model,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "judge_mode": JUDGE_MODE,
            "n": len(results),
            "metrics": metrics,
            "per_answer": results,
        }
        path = baseline_path(provider.name, model)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"Baseline saved: {path.relative_to(Path(__file__).parent.parent)}")
        return 0

    baseline = load_baseline_scores(provider.name, model)
    baseline_mode = baseline.get("judge_mode", "no_reference")
    if baseline_mode != JUDGE_MODE:
        print(f"Baseline was recorded with judge mode {baseline_mode!r}, this run uses {JUDGE_MODE!r} -- "
              f"not comparable. Re-record it: PROVIDER={provider.name} python3 tools/run_eval.py --save-baseline")
        return 2
    drift = drift_check(metrics, baseline)
    verdict = run_level_check(drift)
    print(f"Compared with baseline recorded {baseline['recorded_at']}:")
    for metric, d in drift["details"].items():
        change = "n/a" if d["relative_drop"] is None else f"{(-d['relative_drop'] or 0.0):+.1%}"
        print(f"  {metric}: {d['baseline']} -> {d['current']} ({change}){'  <-- FLAGGED' if d['flagged'] else ''}")
    print(f"RUN VERDICT: {verdict['verdict'].upper()}" + (f"  {verdict['reasons']}" if verdict["reasons"] else ""))
    return 0 if verdict["verdict"] == "approve" else 1


if __name__ == "__main__":
    sys.exit(main())
