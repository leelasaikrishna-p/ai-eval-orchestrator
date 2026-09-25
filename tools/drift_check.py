"""
drift_check -- compares this run's aggregate evaluation scores against a
historical baseline, to catch silent regressions over time.

Deliberately the one tool with NO LLM call: this is pure arithmetic, the
same "compare against last known-good" instinct as a production monitoring
dashboard, just applied to eval scores instead of business metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_utils import load_baseline_scores  # noqa: E402

DEFAULT_THRESHOLD = 0.10  # flag a metric that dropped more than 10% relative to baseline


def drift_check(current_scores: dict, baseline: dict | None = None, threshold: float = DEFAULT_THRESHOLD) -> dict:
    baseline = baseline or load_baseline_scores()
    metrics = ["avg_golden_similarity", "avg_groundedness", "avg_judge_score"]

    details = {}
    drifted = False
    for metric in metrics:
        base_val = baseline[metric]
        cur_val = current_scores[metric]
        relative_drop = (base_val - cur_val) / base_val if base_val else 0.0
        flagged = relative_drop > threshold
        drifted = drifted or flagged
        details[metric] = {
            "baseline": base_val,
            "current": cur_val,
            "relative_drop": round(relative_drop, 4),
            "flagged": flagged,
        }

    return {"drifted": drifted, "threshold": threshold, "details": details}


def _demo():
    baseline = load_baseline_scores()
    print(f"Baseline (recorded {baseline['recorded_at']}): "
          f"golden_similarity={baseline['avg_golden_similarity']}, "
          f"groundedness={baseline['avg_groundedness']}, "
          f"judge_score={baseline['avg_judge_score']}\n")

    scenarios = {
        "no drift (healthy run)": {
            "avg_golden_similarity": 0.87,
            "avg_groundedness": 0.93,
            "avg_judge_score": 8.4,
        },
        "drift detected (groundedness regressed)": {
            "avg_golden_similarity": 0.85,
            "avg_groundedness": 0.79,
            "avg_judge_score": 8.1,
        },
    }
    for label, scores in scenarios.items():
        result = drift_check(scores, baseline)
        print(f"[{label}]")
        print(f"  -> drifted={result['drifted']}")
        for metric, d in result["details"].items():
            flag = " <-- FLAGGED" if d["flagged"] else ""
            pct_change = -d["relative_drop"]  # negative = declined, positive = improved
            print(f"     {metric}: {d['baseline']} -> {d['current']} "
                  f"({pct_change:+.1%}){flag}")
        print()


if __name__ == "__main__":
    _demo()
