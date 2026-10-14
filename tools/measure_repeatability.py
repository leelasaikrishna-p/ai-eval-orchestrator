"""
measure_repeatability -- does each check give the same verdict on the same
answer every time?

An LLM can answer the same prompt differently from one call to the next, so
a check that flags an answer once may pass it the next time. This runs the
three LLM checks several times on every labeled answer and measures:

  flip rate        share of answers whose verdict was not the same in every run
  score spread     how far the raw score moves for the same answer
  single run vs    whether one run catches what a majority vote over all runs
  majority vote    catches, scored against the human labels

Two steps, like measure_agreement:

  python3 tools/measure_repeatability.py run --runs 5 [--limit 20] [--temperature 0.7]
  python3 tools/measure_repeatability.py report [--temperature 0.7]

Every check already calls the model at temperature 0. --temperature overrides
that for the whole run, to show what an unpinned setting does to the same
checks; results are saved per temperature.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_utils import load_golden_dataset, load_json  # noqa: E402
from decision_policy import CONSERVATIVE, judge_gate_value  # noqa: E402
from tools.measure_agreement import (  # noqa: E402
    CHECKS, MEASUREMENT_DIR, _safe, check_row, confusion, flags, metrics,
)


class FixedTemperature:
    """Wraps a provider so every call uses one temperature, whatever the
    caller asks for."""

    def __init__(self, provider, temperature: float):
        self._p, self.temperature = provider, temperature
        self.name, self.model = provider.name, getattr(provider, "model", None)

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        return self._p.generate(prompt, temperature=self.temperature)

    def chat(self, messages, tools=None, temperature: float = 0.0):
        return self._p.chat(messages, tools=tools, temperature=self.temperature)


def results_path(provider: str, model: str | None, temperature: float) -> Path:
    return MEASUREMENT_DIR / f"repeatability_{_safe(provider, model)}_t{temperature:g}.json"


# ---------------------------------------------------------------- run

def run(provider, runs: int, limit: int | None, temperature: float) -> dict:
    golden = load_golden_dataset()
    items = load_json("labeled_answers.json")[:limit] if limit else load_json("labeled_answers.json")
    wrapped = FixedTemperature(provider, temperature)
    all_runs = []
    for r in range(1, runs + 1):
        rows = []
        for n, it in enumerate(items, 1):
            rows.append(check_row(it, golden[it["qid"]], wrapped))
            print(f"  run {r}/{runs}  [{n:>2}/{len(items)}] {it['id']}", flush=True)
        all_runs.append(rows)
    return {
        "provider": provider.name,
        "model": getattr(provider, "model", None),
        "temperature": temperature,
        "runs": runs,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": all_runs,
    }


# ---------------------------------------------------------------- scoring

def raw_score(row: dict, check: str) -> float | None:
    """The number each check's verdict is taken from: golden similarity,
    judge correctness, or 1/0 for grounded/not grounded."""
    c = row[check]
    if "error" in c:
        return None
    if check == "golden_eval":
        return c["score"]
    if check == "groundedness_check":
        return 1.0 if c["grounded"] else 0.0
    return float(judge_gate_value(c)[1])


def verdicts_by_answer(results: list[list[dict]], golden_floor: float, judge_floor: float) -> dict:
    """Pure function: {answer id: {check: [flag per run]}}; None where the
    check errored in that run."""
    out: dict[str, dict[str, list]] = {}
    for rows in results:
        for row in rows:
            f = flags(row, golden_floor, judge_floor)
            per = out.setdefault(row["id"], {c: [] for c in CHECKS})
            for c in CHECKS:
                per[c].append(f[c])
    return out


def majority(votes: list) -> bool | None:
    """Majority flag over the runs that produced a verdict; a tie counts as
    a flag (the cautious side). None if no run produced a verdict."""
    v = [x for x in votes if x is not None]
    if not v:
        return None
    return sum(v) * 2 >= len(v)


def stability(per_answer: dict) -> dict:
    """Pure function: per check, how many answers got the same verdict in
    every run that produced one, and which ones flipped."""
    out = {}
    for c in CHECKS:
        decided = {aid: [x for x in v[c] if x is not None] for aid, v in per_answer.items()}
        decided = {aid: v for aid, v in decided.items() if v}
        flipped = sorted(aid for aid, v in decided.items() if len(set(v)) > 1)
        n = len(decided)
        out[c] = {
            "answers": n,
            "flipped": len(flipped),
            "flip_rate": round(len(flipped) / n, 3) if n else None,
            "flipped_ids": flipped,
            "errors": sum(x is None for v in per_answer.values() for x in v[c]),
        }
    return out


def score_spread(results: list[list[dict]]) -> dict:
    """Pure function: per LLM check, the mean and max range of the raw score
    across runs for the same answer."""
    out = {}
    for c in ("golden_eval", "groundedness_check", "llm_judge"):
        per: dict[str, list[float]] = {}
        for rows in results:
            for row in rows:
                s = raw_score(row, c)
                if s is not None:
                    per.setdefault(row["id"], []).append(s)
        ranges = [max(v) - min(v) for v in per.values() if len(v) > 1]
        stdevs = [statistics.pstdev(v) for v in per.values() if len(v) > 1]
        out[c] = {
            "mean_range": round(statistics.mean(ranges), 3) if ranges else None,
            "max_range": round(max(ranges), 3) if ranges else None,
            "mean_stdev": round(statistics.mean(stdevs), 3) if stdevs else None,
        }
    return out


def single_vs_majority(per_answer: dict, items: list[dict], runs: int) -> dict:
    """Pure function: per check, recall and false-alarm rate of each single
    run (min/max over runs) and of the majority vote, against human labels."""
    labels = {it["id"]: it.get("human_label") for it in items}
    ids = [aid for aid in per_answer if labels.get(aid) in ("acceptable", "not_acceptable")]
    out = {}
    for c in CHECKS:
        per_run = []
        for r in range(runs):
            pairs = [(per_answer[a][c][r], labels[a] == "not_acceptable") for a in ids if per_answer[a][c][r] is not None]
            per_run.append(metrics(confusion(pairs)))
        maj = [(majority(per_answer[a][c]), labels[a] == "not_acceptable") for a in ids]
        m = metrics(confusion([p for p in maj if p[0] is not None]))
        recalls = [p["recall"] for p in per_run if p["recall"] is not None]
        fas = [p["false_alarm_rate"] for p in per_run if p["false_alarm_rate"] is not None]
        out[c] = {
            "single_run_recall": [min(recalls), max(recalls)] if recalls else None,
            "single_run_false_alarms": [min(fas), max(fas)] if fas else None,
            "majority_recall": m["recall"],
            "majority_false_alarms": m["false_alarm_rate"],
        }
    return out


def escalate_if_any_run(per_answer: dict, items: list[dict], runs: int, check: str = "policy_or") -> list[dict]:
    """Pure function: "a verdict that flips goes to a human". With pass/flag
    verdicts that is the same rule as "escalate if any of k runs flags": if
    every run passes nothing flipped, and if any run flags the answer is
    escalated either way. For k = 1..runs, average recall and false alarms
    over every combination of k recorded runs, against the human labels.
    Cost counts runs actually needed: run j is only made on answers that all
    earlier runs passed, so cost is 1 + the share still passing before each
    extra run."""
    from itertools import combinations

    labels = {it["id"]: it.get("human_label") for it in items}
    ids = [a for a in per_answer if labels.get(a) in ("acceptable", "not_acceptable")]
    out = []
    for k in range(1, runs + 1):
        recalls, fas, costs = [], [], []
        for combo in combinations(range(runs), k):
            pairs, cost = [], 1.0
            for j in range(1, k):
                still_passing = [a for a in ids if not any(per_answer[a][check][r] for r in combo[:j])]
                cost += len(still_passing) / len(ids)
            for a in ids:
                votes = [per_answer[a][check][r] for r in combo if per_answer[a][check][r] is not None]
                if votes:
                    pairs.append((any(votes), labels[a] == "not_acceptable"))
            m = metrics(confusion(pairs))
            recalls.append(m["recall"])
            fas.append(m["false_alarm_rate"])
            costs.append(cost)
        out.append({
            "runs": k,
            "recall": round(statistics.mean(recalls), 3),
            "recall_range": [min(recalls), max(recalls)],
            "false_alarms": round(statistics.mean(fas), 3),
            "false_alarm_range": [min(fas), max(fas)],
            "cost_in_runs": round(statistics.mean(costs), 2),
        })
    return out


def flips_by_type(per_answer: dict, items: list[dict], check: str) -> dict:
    """Which answer types the flips land on, for one check."""
    types = {it["id"]: it["type"] for it in items}
    st = stability({a: v for a, v in per_answer.items()})[check]
    return dict(Counter(types[a] for a in st["flipped_ids"]))


def report(path: Path) -> dict:
    record = json.loads(path.read_text())
    items = load_json("labeled_answers.json")
    gf, jf = CONSERVATIVE["golden_eval_floor"], CONSERVATIVE["judge_score_floor"]
    per_answer = verdicts_by_answer(record["results"], gf, jf)
    return {
        "provider": record["provider"],
        "model": record["model"],
        "temperature": record["temperature"],
        "runs": record["runs"],
        "answers": len(per_answer),
        "floors": {"golden_eval": gf, "llm_judge": jf},
        "stability": stability(per_answer),
        "score_spread": score_spread(record["results"]),
        "single_vs_majority": single_vs_majority(per_answer, items, record["runs"]),
        "policy_or_flips_by_type": flips_by_type(per_answer, items, "policy_or"),
        "escalate_if_any_run": escalate_if_any_run(per_answer, items, record["runs"]),
    }


def _pct(x):
    return "  -  " if x is None else f"{x * 100:.0f}%"


def _range(r):
    return "-" if not r else (f"{r[0] * 100:.0f}%" if r[0] == r[1] else f"{r[0] * 100:.0f}-{r[1] * 100:.0f}%")


def print_report(rep: dict) -> None:
    print(f"Repeatability: {rep['provider']}/{rep['model']}  temperature {rep['temperature']}, "
          f"{rep['runs']} runs x {rep['answers']} answers")
    print(f"\n  {'check':<20}{'flip rate':>10}{'flipped':>9}{'errors':>8}")
    for c, s in rep["stability"].items():
        print(f"  {c:<20}{_pct(s['flip_rate']):>10}{s['flipped']:>5}/{s['answers']:<3}{s['errors']:>8}")
    print("\n  score spread for the same answer across runs (mean range / max range)")
    for c, s in rep["score_spread"].items():
        print(f"  {c:<20}{s['mean_range'] if s['mean_range'] is not None else '-':>8} / {s['max_range'] if s['max_range'] is not None else '-'}")
    print(f"\n  {'check':<20}{'one run: recall':>16}{'false alarms':>14}{'majority: recall':>18}{'false alarms':>14}")
    for c, m in rep["single_vs_majority"].items():
        print(f"  {c:<20}{_range(m['single_run_recall']):>16}{_range(m['single_run_false_alarms']):>14}"
              f"{_pct(m['majority_recall']):>18}{_pct(m['majority_false_alarms']):>14}")
    flips = rep["stability"]["policy_or"]["flipped_ids"]
    print(f"\n  policy (OR) flipped on: {', '.join(flips) or 'none'}"
          + (f"  by type: {rep['policy_or_flips_by_type']}" if flips else ""))
    print("\n  flip = escalate (policy OR): escalate if any of k runs flags, averaged over every set of k runs")
    print(f"  {'k runs':<8}{'recall':>8}{'range':>12}{'false alarms':>14}{'range':>12}{'cost (runs)':>13}")
    for e in rep["escalate_if_any_run"]:
        print(f"  {e['runs']:<8}{_pct(e['recall']):>8}{_range(e['recall_range']):>12}"
              f"{_pct(e['false_alarms']):>14}{_range(e['false_alarm_range']):>12}{e['cost_in_runs']:>13}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["run", "report"])
    parser.add_argument("--runs", type=int, default=5, help="times to run each check on each answer (run step)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N labeled answers (run step), to save quota")
    parser.add_argument("--temperature", type=float, default=0.0, help="override the checks' temperature (default 0, as shipped)")
    args = parser.parse_args(argv)

    import os

    from llm_provider import active_model, get_provider

    if args.step == "run":
        if args.runs < 2:
            print("Repeatability needs at least 2 runs.")
            return 2
        provider = get_provider()
        model = getattr(provider, "model", None)
        path = results_path(provider.name, model, args.temperature)
        print(f"Running every check {args.runs}x per answer: {provider.name}/{model} at temperature {args.temperature:g}")
        record = run(provider, args.runs, args.limit, args.temperature)
        MEASUREMENT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {path.relative_to(Path(__file__).parent.parent)}")
        return 0

    # Scoring reads recorded results only: no model calls, so no API key.
    provider_name = os.environ.get("PROVIDER", "ollama")
    model = active_model(provider_name)
    path = results_path(provider_name, model, args.temperature)
    if not path.exists():
        print(f"No recorded results for {provider_name}/{model} at temperature {args.temperature:g}. "
              f"Run: PROVIDER={provider_name} python3 tools/measure_repeatability.py run")
        return 2
    rep = report(path)
    print_report(rep)
    out = MEASUREMENT_DIR / f"repeatability_report_{_safe(provider_name, model)}_t{args.temperature:g}.json"
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
