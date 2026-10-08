"""
measure_agreement -- how well does each check agree with human labels?

data/labeled_answers.json holds 60 answers to the 15 golden questions. Each
has a human label (acceptable / not_acceptable / unsure), collected blind on
a labeling page, plus the label and failure type proposed when the answer
was written. Two steps, so labels can change without re-running models:

  python3 tools/measure_agreement.py run      # run the 3 LLM checks on every answer, record raw results
  python3 tools/measure_agreement.py report   # score the recorded results against the human labels

A "catch" is a check flagging an answer as not shippable:
  golden_eval         score below the golden floor
  groundedness_check  grounded is false
  llm_judge           correctness below the judge floor (reference-guided)
  policy (OR / AND)   the per-answer verdict sign_off() would give
The positive class is "not acceptable" (what a check should catch), so
recall is the share of bad answers caught and the false-alarm rate is the
share of good answers flagged. Answers labeled "unsure" are left out.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_utils import DATA_DIR, load_golden_dataset, load_json  # noqa: E402
from decision_policy import CONSERVATIVE, judge_gate_value, sign_off  # noqa: E402

MEASUREMENT_DIR = DATA_DIR / "measurements"
CHECKS = ("golden_eval", "groundedness_check", "llm_judge", "policy_or", "policy_and")


def _safe(provider: str, model: str | None) -> str:
    return f"{provider}__{model or 'default'}".replace("/", "_").replace(":", "_")


def results_path(provider: str, model: str | None) -> Path:
    return MEASUREMENT_DIR / f"agreement_{_safe(provider, model)}.json"


# ---------------------------------------------------------------- run

def run(provider) -> dict:
    from tools.golden_eval import golden_eval
    from tools.groundedness_check import groundedness_check
    from tools.llm_judge import llm_judge

    golden = load_golden_dataset()
    rows = []
    items = load_json("labeled_answers.json")
    for n, it in enumerate(items, 1):
        q = golden[it["qid"]]
        row = {"id": it["id"]}
        for name, call in (
            ("golden_eval", lambda: golden_eval(q["question"], q["golden_answer"], it["candidate_answer"], provider)),
            ("groundedness_check", lambda: groundedness_check(q["context"], it["candidate_answer"], provider)),
            ("llm_judge", lambda: llm_judge(q["question"], it["candidate_answer"], provider, reference_answer=q["golden_answer"])),
        ):
            try:
                r = call()
                if name == "golden_eval":
                    row[name] = {"score": r["score"]}
                elif name == "groundedness_check":
                    row[name] = {"grounded": r["grounded"], "unsupported_claims": r.get("unsupported_claims", [])}
                else:
                    row[name] = {"score": r["score"], "breakdown": r["breakdown"]}
            except Exception as e:  # a failed check is recorded, not raised
                row[name] = {"error": f"{type(e).__name__}: {e}"[:200]}
        rows.append(row)
        print(f"  [{n:>2}/{len(items)}] {it['id']} done", flush=True)
    return {
        "provider": provider.name,
        "model": getattr(provider, "model", None),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows,
    }


# ---------------------------------------------------------------- scoring

def flags(row: dict, golden_floor: float, judge_floor: float) -> dict:
    """Pure function: which checks flag this answer. A check that errored
    is None (no evidence), which the policy treats as missing."""
    g, gr, j = row["golden_eval"], row["groundedness_check"], row["llm_judge"]
    out = {
        "golden_eval": None if "error" in g else g["score"] < golden_floor,
        "groundedness_check": None if "error" in gr else not gr["grounded"],
        "llm_judge": None if "error" in j else judge_gate_value(j)[1] < judge_floor,
    }
    for mode in ("or", "and"):
        profile = {**CONSERVATIVE, "golden_eval_floor": golden_floor, "judge_score_floor": judge_floor, "tier2_mode": mode}
        out[f"policy_{mode}"] = policy_flags(row, profile)
    return out


def policy_flags(row: dict, profile: dict) -> bool:
    """True if sign_off() would escalate this answer under `profile`."""
    from decision_policy import GATE_PROFILES

    key = "_agreement_measurement"
    GATE_PROFILES[key] = profile
    try:
        g, gr, j = row["golden_eval"], row["groundedness_check"], row["llm_judge"]
        verdict = sign_off(
            {"score": g["score"]} if "error" not in g else {"score": 0.0},
            None if "error" in gr else {"grounded": gr["grounded"], "unsupported_claims": gr["unsupported_claims"]},
            None if "error" in j else j,
            provider=key,
        )
    finally:
        del GATE_PROFILES[key]
    return verdict["verdict"] == "escalate"


def confusion(pairs: list[tuple[bool, bool]]) -> dict:
    """Pure function: pairs of (flagged, actually_not_acceptable)."""
    tp = sum(1 for f, bad in pairs if f and bad)
    fp = sum(1 for f, bad in pairs if f and not bad)
    fn = sum(1 for f, bad in pairs if not f and bad)
    tn = sum(1 for f, bad in pairs if not f and not bad)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """95% Wilson score interval for k successes out of n."""
    if n == 0:
        return None
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3))


def cohens_kappa(c: dict) -> float | None:
    n = c["tp"] + c["fp"] + c["fn"] + c["tn"]
    if n == 0:
        return None
    po = (c["tp"] + c["tn"]) / n
    p_yes = ((c["tp"] + c["fp"]) / n) * ((c["tp"] + c["fn"]) / n)
    p_no = ((c["fn"] + c["tn"]) / n) * ((c["fp"] + c["tn"]) / n)
    pe = p_yes + p_no
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def metrics(c: dict) -> dict:
    bad, good = c["tp"] + c["fn"], c["fp"] + c["tn"]
    flagged = c["tp"] + c["fp"]
    return {
        **c,
        "recall": round(c["tp"] / bad, 3) if bad else None,
        "recall_ci95": wilson(c["tp"], bad),
        "false_alarm_rate": round(c["fp"] / good, 3) if good else None,
        "false_alarm_ci95": wilson(c["fp"], good),
        "precision": round(c["tp"] / flagged, 3) if flagged else None,
        "kappa": cohens_kappa(c),
    }


def score(rows: list[dict], items: list[dict], golden_floor: float, judge_floor: float) -> dict:
    """Pure function: per-check metrics against human labels, plus recall
    per failure type, for answers with a definite human label."""
    by_id = {it["id"]: it for it in items}
    labeled = [r for r in rows if by_id[r["id"]].get("human_label") in ("acceptable", "not_acceptable")]
    out = {"n": len(labeled), "checks": {}, "by_type": {}}
    for check in CHECKS:
        pairs, errors = [], 0
        per_type: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        for r in labeled:
            f = flags(r, golden_floor, judge_floor)[check]
            if f is None:
                errors += 1
                continue
            bad = by_id[r["id"]]["human_label"] == "not_acceptable"
            pairs.append((f, bad))
            per_type[by_id[r["id"]]["type"]].append((f, bad))
        out["checks"][check] = {**metrics(confusion(pairs)), "errors": errors}
        out["by_type"][check] = {t: {"caught_or_passed": sum(1 for f, bad in p if f == bad), "n": len(p)} for t, p in sorted(per_type.items())}
    return out


def sweep(rows: list[dict], items: list[dict], check: str, floors: list[float]) -> list[dict]:
    """Recall / false-alarm rate across thresholds for one check."""
    out = []
    for floor in floors:
        gf = floor if check == "golden_eval" else CONSERVATIVE["golden_eval_floor"]
        jf = floor if check == "llm_judge" else CONSERVATIVE["judge_score_floor"]
        m = score(rows, items, gf, jf)["checks"][check]
        out.append({"floor": floor, "recall": m["recall"], "false_alarm_rate": m["false_alarm_rate"], "kappa": m["kappa"]})
    return out


def annotator_agreement(items: list[dict]) -> dict:
    """Human labels vs the labels proposed when the answers were written."""
    both = [it for it in items if it.get("human_label") in ("acceptable", "not_acceptable")]
    pairs = [(it["proposed_label"] == "not_acceptable", it["human_label"] == "not_acceptable") for it in both]
    c = confusion(pairs)
    disagreements = [
        {"id": it["id"], "type": it["type"], "proposed": it["proposed_label"], "human": it["human_label"], "note": it.get("human_note", "")}
        for it in both if it["proposed_label"] != it["human_label"]
    ]
    unsure = [it["id"] for it in items if it.get("human_label") == "unsure"]
    return {"n": len(both), "agree": c["tp"] + c["tn"], "kappa": cohens_kappa(c), "disagreements": disagreements, "unsure": unsure}


def report(path: Path) -> dict:
    record = json.loads(path.read_text())
    items = load_json("labeled_answers.json")
    gf, jf = CONSERVATIVE["golden_eval_floor"], CONSERVATIVE["judge_score_floor"]
    return {
        "provider": record["provider"],
        "model": record["model"],
        "floors": {"golden_eval": gf, "llm_judge": jf},
        "annotators": annotator_agreement(items),
        "at_current_floors": score(record["rows"], items, gf, jf),
        "judge_sweep": sweep(record["rows"], items, "llm_judge", [2, 3, 4, 5, 6, 7, 8, 9]),
        "golden_sweep": sweep(record["rows"], items, "golden_eval", [0.1, 0.3, 0.5, 0.7, 0.9]),
    }


def _fmt(x):
    return "  -  " if x is None else f"{x:.2f}"


def print_report(rep: dict) -> None:
    a = rep["annotators"]
    print(f"Agreement report: {rep['provider']}/{rep['model']}  (floors: golden {rep['floors']['golden_eval']}, judge correctness {rep['floors']['llm_judge']})")
    print(f"\nHuman vs proposed labels: {a['agree']}/{a['n']} agree, kappa {a['kappa']}; unsure: {a['unsure'] or 'none'}")
    for d in a["disagreements"]:
        print(f"  {d['id']} ({d['type']}): proposed {d['proposed']}, human {d['human']}  {d['note']}")
    s = rep["at_current_floors"]
    print(f"\nChecks vs human labels (n={s['n']}; positive = not acceptable)")
    print(f"  {'check':<20}{'recall':>7}  {'95% CI':<14}{'false alarms':>13}  {'95% CI':<14}{'precision':>10}{'kappa':>7}  errors")
    for name, m in s["checks"].items():
        ci_r = f"[{m['recall_ci95'][0]:.2f}, {m['recall_ci95'][1]:.2f}]" if m["recall_ci95"] else "-"
        ci_f = f"[{m['false_alarm_ci95'][0]:.2f}, {m['false_alarm_ci95'][1]:.2f}]" if m["false_alarm_ci95"] else "-"
        print(f"  {name:<20}{_fmt(m['recall']):>7}  {ci_r:<14}{_fmt(m['false_alarm_rate']):>13}  {ci_f:<14}{_fmt(m['precision']):>10}{_fmt(m['kappa']):>7}  {m['errors']}")
    print("\nCorrect calls by answer type (caught if bad, passed if good)")
    types = sorted({t for c in s["by_type"].values() for t in c})
    print(f"  {'type':<20}" + "".join(f"{c[:12]:>14}" for c in s["by_type"]))
    for t in types:
        cells = []
        for c in s["by_type"].values():
            v = c.get(t)
            cells.append(f"{v['caught_or_passed']}/{v['n']}" if v else "-")
        print(f"  {t:<20}" + "".join(f"{x:>14}" for x in cells))
    for label, sw in (("judge correctness floor", rep["judge_sweep"]), ("golden_eval floor", rep["golden_sweep"])):
        print(f"\n{label} sweep:  " + "  ".join(f"{p['floor']}: R {_fmt(p['recall'])} / FA {_fmt(p['false_alarm_rate'])}" for p in sw))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["run", "report"])
    args = parser.parse_args(argv)

    import os

    from llm_provider import active_model, get_provider

    if args.step == "run":
        provider = get_provider()
        model = getattr(provider, "model", None)
        path = results_path(provider.name, model)
        print(f"Running all checks on every labeled answer: {provider.name}/{model}")
        record = run(provider)
        MEASUREMENT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {path.relative_to(Path(__file__).parent.parent)}")
        return 0

    # Scoring reads recorded results only: no model calls, so no API key.
    provider_name = os.environ.get("PROVIDER", "ollama")
    model = active_model(provider_name)
    path = results_path(provider_name, model)
    if not path.exists():
        print(f"No recorded results for {provider_name}/{model}. Run: PROVIDER={provider_name} python3 tools/measure_agreement.py run")
        return 2
    items = load_json("labeled_answers.json")
    if not any(it.get("human_label") for it in items):
        print("No human labels in data/labeled_answers.json yet.")
        return 2
    rep = report(path)
    print_report(rep)
    (MEASUREMENT_DIR / f"agreement_report_{_safe(provider_name, model)}.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
