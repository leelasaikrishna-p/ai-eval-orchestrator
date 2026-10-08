"""
measure_judge -- does giving the judge the reference answer make it reliable?

Runs llm_judge on every labeled answer twice, without and with the golden
answer, and counts errors against the labels at the policy's judge floor:
  - false negative: a wrong answer (hallucinated / subtly wrong) scored at
    or above the floor, i.e. the judge would have let it through
  - false positive: a correct answer scored below the floor

It reports the overall score (mean of the three dimensions) AND
correctness alone, because a mean can hide a correctness failure behind
good clarity and tone.

Usage (PROVIDER / GEMINI_MODEL / OLLAMA_MODEL pick the model):
  python3 tools/measure_judge.py
  python3 tools/measure_judge.py --save   # record to data/measurements/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_utils import DATA_DIR, load_golden_dataset, load_sample_candidate_runs  # noqa: E402
from decision_policy import CONSERVATIVE  # noqa: E402
from llm_provider import get_provider  # noqa: E402
from tools.llm_judge import llm_judge  # noqa: E402

MEASUREMENT_DIR = DATA_DIR / "measurements"
CORRECT_LABELS = {"clean pass"}


def count_errors(rows: list[dict], key: str, floor: float) -> dict:
    """Pure function: rows with {correct, <key>} -> false negative/positive ids."""
    fn = [r["id"] for r in rows if not r["correct"] and r[key] >= floor]
    fp = [r["id"] for r in rows if r["correct"] and r[key] < floor]
    return {"false_negatives": fn, "false_positives": fp}


def measure(provider) -> list[dict]:
    golden = load_golden_dataset()
    rows = []
    for run in load_sample_candidate_runs():
        item = golden[run["id"]]
        for mode, ref in (("no_reference", None), ("reference", item["golden_answer"])):
            j = llm_judge(item["question"], run["candidate_answer"], provider, reference_answer=ref)
            rows.append({
                "id": run["id"],
                "label": run["label"],
                "correct": run["label"] in CORRECT_LABELS,
                "mode": mode,
                "score": j["score"],
                "correctness": float(j["breakdown"]["correctness"]),
                "breakdown": j["breakdown"],
                "reasoning": j.get("reasoning", ""),
            })
    return rows


def summarize(rows: list[dict], floor: float) -> dict:
    out = {}
    for mode in ("no_reference", "reference"):
        mode_rows = [r for r in rows if r["mode"] == mode]
        out[mode] = {
            "overall_score": count_errors(mode_rows, "score", floor),
            "correctness_only": count_errors(mode_rows, "correctness", floor),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--save", action="store_true", help="record results to data/measurements/")
    args = parser.parse_args(argv)

    provider = get_provider()
    model = getattr(provider, "model", None)
    floor = CONSERVATIVE["judge_score_floor"]
    print(f"Judge measurement: {provider.name}/{model}  (floor {floor})\n")
    rows = measure(provider)
    for r in rows:
        b = r["breakdown"]
        print(f"  [{r['id']}] {r['label']:<26} {r['mode']:<13} overall={r['score']:>4.1f}  "
              f"correctness={b['correctness']} clarity={b['clarity']} tone={b['tone']}")
    summary = summarize(rows, floor)
    print()
    for mode, s in summary.items():
        for basis, e in s.items():
            print(f"  {mode:<13} {basis:<17} FN={e['false_negatives']} FP={e['false_positives']}")

    if args.save:
        MEASUREMENT_DIR.mkdir(parents=True, exist_ok=True)
        safe = f"{provider.name}__{model or 'default'}".replace("/", "_").replace(":", "_")
        path = MEASUREMENT_DIR / f"judge_{safe}.json"
        record = {
            "provider": provider.name,
            "model": model,
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "floor": floor,
            "summary": summary,
            "rows": rows,
        }
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"\nSaved: {path.relative_to(Path(__file__).parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
