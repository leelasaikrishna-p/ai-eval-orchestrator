"""
drift_check -- compares a run's aggregate evaluation scores against the
recorded baseline for the same provider/model, to catch silent regressions.

Deliberately the one tool with NO LLM call: pure arithmetic, the same
"compare against last known-good" instinct as a production monitoring
dashboard, applied to eval scores instead of business metrics.

For each metric: relative_drop = (baseline - current) / baseline, flagged
when it exceeds the threshold (10%). A rise is never flagged. Baselines are
recorded per model by tools/run_eval.py --save-baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_utils import load_baseline_scores  # noqa: E402

DEFAULT_THRESHOLD = 0.10  # flag a metric that dropped more than 10% relative to baseline
METRICS = ("avg_golden_similarity", "avg_groundedness", "avg_judge_score")


def drift_check(current_scores: dict, baseline: dict | None = None, threshold: float = DEFAULT_THRESHOLD) -> dict:
    baseline = baseline or load_baseline_scores()
    base_metrics = baseline.get("metrics", baseline)  # recorded files nest them under "metrics"

    details = {}
    drifted = False
    for metric in METRICS:
        base_val = base_metrics[metric]
        cur_val = current_scores[metric]
        detail = {"baseline": base_val, "current": cur_val}
        if base_val:
            relative_drop = (base_val - cur_val) / base_val
            detail["relative_drop"] = round(relative_drop, 4)
            detail["flagged"] = relative_drop > threshold
        else:
            # A relative drop from 0 is undefined -- say so instead of silently passing.
            detail["relative_drop"] = None
            detail["flagged"] = False
            detail["note"] = "baseline is 0, so a relative drop can't be measured for this metric"
        drifted = drifted or detail["flagged"]
        details[metric] = detail

    return {"drifted": drifted, "threshold": threshold, "details": details}


def _demo():
    """Uses the REAL recorded baseline; the two 'current' runs are synthetic,
    derived from it, to show one healthy run and one regression."""
    baseline = load_baseline_scores()
    base = baseline["metrics"]
    print(f"Baseline for {baseline['provider']}/{baseline['model']} "
          f"(recorded {baseline['recorded_at']}, n={baseline['n']}): {base}\n")

    scenarios = {
        "healthy run (synthetic: same as baseline)": dict(base),
        "regression (synthetic: every metric down 20%)": {m: round(v * 0.8, 4) for m, v in base.items()},
    }
    for label, scores in scenarios.items():
        result = drift_check(scores, baseline)
        print(f"[{label}]\n  -> drifted={result['drifted']}")
        for metric, d in result["details"].items():
            if d["relative_drop"] is None:
                print(f"     {metric}: {d['baseline']} -> {d['current']}  ({d['note']})")
                continue
            flag = " <-- FLAGGED" if d["flagged"] else ""
            print(f"     {metric}: {d['baseline']} -> {d['current']} ({(-d['relative_drop'] or 0.0):+.1%}){flag}")
        print()


if __name__ == "__main__":
    _demo()
