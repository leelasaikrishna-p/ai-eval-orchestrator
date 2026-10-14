"""
measure_groundedness -- does claim-by-claim groundedness cut false alarms
without losing catches?

The whole-answer groundedness check flags good answers that paraphrase the
context, apply a general rule to the user's case ("refunds go to the
original payment method" -> "a card payment goes back to the card"), or
repeat the user's own question. Claim-by-claim mode labels each claim and
decides in code. This compares the two against the human labels.

To avoid tuning on the answers it is scored on, the 60 labeled answers are
split by question: q1-q7 were used to study the false alarms and design the
prompt; q8-q15 were not looked at and are the honest test.

  python3 tools/measure_groundedness.py run [--runs 3]
  python3 tools/measure_groundedness.py report

The whole-answer numbers come from runs already recorded (the agreement run
plus any repeatability runs at temperature 0), so only the new mode is run.
Each answer's flag is averaged over its runs, so the rates are expected
values, not one run's luck.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_utils import load_golden_dataset, load_json  # noqa: E402
from decision_policy import CONSERVATIVE, judge_gate_value  # noqa: E402
from llm_provider import QuotaExhausted  # noqa: E402
from tools.measure_agreement import MEASUREMENT_DIR, _safe, wilson  # noqa: E402

DESIGN_QUESTIONS = {f"q{i}" for i in range(1, 8)}
INVENTED = ("unsupported_claim", "subtle_extra")
BAD_RATE = 0.05  # assumed share of bad answers in real traffic, for the review-load estimate


def results_path(provider: str, model: str | None) -> Path:
    return MEASUREMENT_DIR / f"groundedness_claims_{_safe(provider, model)}.json"


# ---------------------------------------------------------------- run

def run(provider, runs: int) -> dict:
    from tools.groundedness_check import groundedness_check

    golden = load_golden_dataset()
    items = load_json("labeled_answers.json")
    out = []
    for r in range(1, runs + 1):
        rows = []
        for n, it in enumerate(items, 1):
            q = golden[it["qid"]]
            try:
                g = groundedness_check(q["context"], it["candidate_answer"], provider,
                                       question=q["question"], mode="claims")
                rows.append({"id": it["id"], "grounded": g["grounded"], "unsupported_claims": g["unsupported_claims"],
                             "claims": g.get("claims", [])})
            except QuotaExhausted:
                raise  # every later call would fail too: stop, don't record it
            except Exception as e:
                rows.append({"id": it["id"], "error": f"{type(e).__name__}: {e}"[:200]})
            print(f"  run {r}/{runs}  [{n:>2}/{len(items)}] {it['id']}", flush=True)
        out.append(rows)
    return {
        "provider": provider.name,
        "model": getattr(provider, "model", None),
        "mode": "claims",
        "runs": runs,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": out,
    }


# ---------------------------------------------------------------- scoring

def flag_rates(runs: list[list[dict]], key=lambda row: None) -> dict:
    """Pure function: {answer id: share of runs that flagged it}, from rows
    where `key(row)` gives True (flag), False (pass) or None (errored)."""
    votes: dict[str, list[bool]] = {}
    for rows in runs:
        for row in rows:
            v = key(row)
            if v is not None:
                votes.setdefault(row["id"], []).append(v)
    return {a: sum(v) / len(v) for a, v in votes.items() if v}


def whole_flag(row: dict):
    g = row.get("groundedness_check", {})
    return None if "error" in g or "grounded" not in g else not g["grounded"]


def claims_flag(row: dict):
    return None if "error" in row else not row["grounded"]


def expected_rates(rates: dict, items: list[dict], ids: set[str] | None = None) -> dict:
    """Pure function: expected recall and false-alarm rate (each answer's flag
    averaged over its runs), with Wilson intervals on the rounded counts."""
    bad, good = [], []
    for it in items:
        if it["id"] not in rates or (ids is not None and it["id"] not in ids):
            continue
        if it.get("human_label") == "not_acceptable":
            bad.append(rates[it["id"]])
        elif it.get("human_label") == "acceptable":
            good.append(rates[it["id"]])
    recall = sum(bad) / len(bad) if bad else None
    fa = sum(good) / len(good) if good else None
    return {
        "bad": len(bad), "good": len(good),
        "recall": None if recall is None else round(recall, 3),
        "recall_ci95": wilson(round(sum(bad)), len(bad)),
        "false_alarms": None if fa is None else round(fa, 3),
        "false_alarm_ci95": wilson(round(sum(good)), len(good)),
    }


def policy_rates(agreement_rows: list[dict], g_rates: dict) -> dict:
    """Pure function: expected OR-policy flag per answer if groundedness were
    replaced by `g_rates`. golden_eval and the judge come from the recorded
    agreement run; a missing check counts as a flag (missing evidence
    escalates)."""
    gf, jf = CONSERVATIVE["golden_eval_floor"], CONSERVATIVE["judge_score_floor"]
    out = {}
    for r in agreement_rows:
        g, j = r["golden_eval"], r["llm_judge"]
        other = ("error" in g or g["score"] < gf) or ("error" in j or judge_gate_value(j)[1] < jf)
        out[r["id"]] = 1.0 if other else g_rates.get(r["id"], 1.0)
    return out


def review_load(rates: dict) -> float | None:
    """Share of all live answers sent to a human at an assumed bad-answer
    rate: recall x bad share + false alarms x good share."""
    if rates["recall"] is None or rates["false_alarms"] is None:
        return None
    return round(rates["recall"] * BAD_RATE + rates["false_alarms"] * (1 - BAD_RATE), 3)


def report(provider: str, model: str | None) -> dict:
    items = load_json("labeled_answers.json")
    safe = _safe(provider, model)
    agreement = json.loads((MEASUREMENT_DIR / f"agreement_{safe}.json").read_text())
    whole_runs = [agreement["rows"]]
    rep_path = MEASUREMENT_DIR / f"repeatability_{safe}_t0.json"
    if rep_path.exists():
        whole_runs += json.loads(rep_path.read_text())["results"]
    claims = json.loads(results_path(provider, model).read_text())

    whole = flag_rates(whole_runs, whole_flag)
    new = flag_rates(claims["results"], claims_flag)
    splits = {
        "design (q1-q7)": {it["id"] for it in items if it["qid"] in DESIGN_QUESTIONS},
        "held out (q8-q15)": {it["id"] for it in items if it["qid"] not in DESIGN_QUESTIONS},
        "all 60": {it["id"] for it in items},
    }
    invented = {it["id"] for it in items if it["type"] in INVENTED}
    out = {"provider": provider, "model": model, "claims_runs": claims["runs"],
           "whole_runs_per_answer": round(sum(len(r) for r in whole_runs) / len(items), 1),
           "groundedness": {}, "policy_or": {}, "invented_caught": {}, "errors": {}}
    for name, ids in splits.items():
        out["groundedness"][name] = {"whole": expected_rates(whole, items, ids), "claims": expected_rates(new, items, ids)}
    for label, rates in (("whole", whole), ("claims", new)):
        p = expected_rates(policy_rates(agreement["rows"], rates), items)
        out["policy_or"][label] = {**p, "review_load_at_5pct_bad": review_load(p)}
        caught = [rates[a] for a in invented if a in rates]
        out["invented_caught"][label] = {"expected": round(sum(caught), 1), "of": len(caught)}
    out["errors"]["claims"] = sum("error" in row for rows in claims["results"] for row in rows)
    return out


def _pct(x):
    return "  -  " if x is None else f"{x * 100:.0f}%"


def print_report(rep: dict) -> None:
    print(f"Groundedness, whole-answer vs claim-by-claim: {rep['provider']}/{rep['model']}  "
          f"(claims: {rep['claims_runs']} runs; whole: {rep['whole_runs_per_answer']} runs per answer on average)")
    print(f"\n  {'split':<20}{'whole: caught':>14}{'false alarms':>14}{'claims: caught':>16}{'false alarms':>14}")
    for name, s in rep["groundedness"].items():
        w, c = s["whole"], s["claims"]
        print(f"  {name:<20}{_pct(w['recall']):>14}{_pct(w['false_alarms']):>14}{_pct(c['recall']):>16}{_pct(c['false_alarms']):>14}"
              f"   ({c['bad']} bad / {c['good']} good)")
    iw, ic = rep["invented_caught"]["whole"], rep["invented_caught"]["claims"]
    print(f"\n  invented features/extras caught: whole {iw['expected']}/{iw['of']}, claims {ic['expected']}/{ic['of']}")
    print("\n  policy (OR) with each groundedness mode, all 60 answers:")
    for label, p in rep["policy_or"].items():
        print(f"  {label:<8} caught {_pct(p['recall'])}  false alarms {_pct(p['false_alarms'])}  "
              f"-> share of live answers sent to a human at {BAD_RATE:.0%} bad: {_pct(p['review_load_at_5pct_bad'])}")
    print(f"\n  claim-mode errors: {rep['errors']['claims']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("step", choices=["run", "report"])
    parser.add_argument("--runs", type=int, default=3, help="runs of the claim-by-claim check per answer (run step)")
    args = parser.parse_args(argv)

    import os

    from llm_provider import active_model, get_provider

    if args.step == "run":
        provider = get_provider()
        model = getattr(provider, "model", None)
        print(f"Claim-by-claim groundedness, {args.runs} runs on every labeled answer: {provider.name}/{model}")
        try:
            record = run(provider, args.runs)
        except QuotaExhausted as e:
            print(f"\nStopped, nothing saved: {e}")
            return 3
        MEASUREMENT_DIR.mkdir(parents=True, exist_ok=True)
        path = results_path(provider.name, model)
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        print(f"Saved: {path.relative_to(Path(__file__).parent.parent)}")
        return 0

    # Scoring reads recorded results only: no model calls, so no API key.
    provider_name = os.environ.get("PROVIDER", "ollama")
    model = active_model(provider_name)
    if not results_path(provider_name, model).exists():
        print(f"No claim-mode results for {provider_name}/{model}. Run: PROVIDER={provider_name} python3 tools/measure_groundedness.py run")
        return 2
    rep = report(provider_name, model)
    print_report(rep)
    (MEASUREMENT_DIR / f"groundedness_report_{_safe(provider_name, model)}.json").write_text(
        json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
