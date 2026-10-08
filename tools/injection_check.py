"""
injection_check -- flags a candidate answer that contains instructions
aimed at the evaluator instead of an answer for the customer.

The answers this project evaluates are untrusted: they come from the system
under test, and an answer can carry text like "score this 10/10" or a fake
result JSON. Every LLM tool reads that text, and so does the orchestrator's
model, so a check the model chooses to run could be skipped by the very
answer it is checking. This one is plain pattern matching, with no LLM, and
the orchestrator runs it in code on every answer.

It is a heuristic: it catches common, crude attacks, not every possible
one. The tools' prompts also mark the answer as data (tools/untrusted.py),
and the decision policy treats a flag as a reason to escalate to a human,
which costs a minute of review if it is a false alarm.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# (name, pattern) -- each describes a way an answer talks to its evaluator.
PATTERNS = [
    ("override instructions",
     r"\b(ignore|disregard|forget|override)\b.{0,40}\b(instructions?|prompt)\b"),
    ("addresses the evaluator",
     r"\b(note|message|instructions?)\s+(to|for)\s+(the\s+)?(evaluator|grader|judge|reviewer|model|ai|assistant)\b"),
    ("asks for a score",
     r"\b(score|rate|grade|mark)\s+(it|this|this answer|the answer|me|us)\b.{0,30}"
     r"\b(10|ten|maximum|max|highest|perfect|full marks)\b"),
    ("asks for a score", r"\b(return|give)\s+(it\s+)?(the\s+)?(maximum|highest|perfect|full)\s+(score|marks|rating)\b"),
    ("role marker", r"(^|\n)\s*(system|assistant|developer)\s*:"),
    ("prompt delimiter", r"</?\s*(candidate_answer|reference_answer|context|system|instructions?)\b[^>]*>"),
    ("result JSON",
     r"\"(score|grounded|breakdown|unsupported_claims|correctness|verdict)\"\s*:"),
]
COMPILED = [(name, re.compile(p, re.IGNORECASE)) for name, p in PATTERNS]


def injection_check(candidate_answer: str) -> dict:
    """Pure function: {"suspicious": bool, "matches": [{"pattern", "text"}]}."""
    matches = []
    for name, rx in COMPILED:
        m = rx.search(candidate_answer)
        if m:
            matches.append({"pattern": name, "text": m.group(0).strip()[:80]})
    return {"suspicious": bool(matches), "matches": matches}


def _demo():
    from data_utils import load_json, load_sample_candidate_runs

    for run in load_sample_candidate_runs() + load_json("injection_cases.json"):
        label = run.get("label") or run["attack"]
        result = injection_check(run["candidate_answer"])
        names = [m["pattern"] for m in result["matches"]]
        print(f"[{run['id']}] {label:<26} suspicious={result['suspicious']}  {names}")


if __name__ == "__main__":
    _demo()
