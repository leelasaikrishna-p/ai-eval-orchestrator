"""
CI helper: computes a sign-off verdict for one of the three labeled demo
cases and writes it to $GITHUB_OUTPUT, so the workflow can branch on it
(skip the human-approval gate on a clean approve, require it on escalate).

Uses the ACTUAL recorded scores from a real llama3.1:8b run (the same
values in tests/test_decision_policy.py) rather than a live model call --
hosted GitHub runners don't have Ollama installed, and pulling a multi-GB
model on every workflow run would be slow and wasteful for what this step
is demonstrating: that decision_policy's gate structure runs correctly
inside a real CI pipeline and correctly triggers (or skips) the human-
approval environment gate. Judging live model quality is a local exercise
(see README); this proves the pipeline integration, not model behavior.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from decision_policy import sign_off  # noqa: E402

# Verbatim recorded scores from a real Ollama run -- see tests/test_decision_policy.py
CASES = {
    "q1": {  # clean pass -- groundedness false positive, still safe (escalates)
        "golden": {"score": 0.90, "reasoning": "same meaning, minor wording differences"},
        "groundedness": {"grounded": False, "unsupported_claims": ["mostly locked"]},
        "judge": {"score": 8.0, "breakdown": {}},
    },
    "q3": {  # hallucinated -- caught immediately by golden_eval alone
        "golden": {"score": 0.00, "reasoning": "contradicts golden answer"},
        "groundedness": {"grounded": False, "unsupported_claims": ["support can reroute refunds"]},
        "judge": {"score": 4.0, "breakdown": {}},
    },
    "q5": {  # subtly wrong -- only groundedness catches it; judge misses it
        "golden": {"score": 0.80, "reasoning": "mostly matches, extra detail"},
        "groundedness": {"grounded": False, "unsupported_claims": ["can change currency later"]},
        "judge": {"score": 8.0, "breakdown": {}},
    },
    "clean": {  # a genuinely clean case, for a real auto-approve demo
        "golden": {"score": 0.97, "reasoning": "matches"},
        "groundedness": {"grounded": True, "unsupported_claims": []},
        "judge": {"score": 9.5, "breakdown": {}},
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=CASES.keys(), required=True)
    args = parser.parse_args()

    case = CASES[args.case]
    result = sign_off(case["golden"], case["groundedness"], case["judge"])

    print(f"Case: {args.case}")
    print(json.dumps(result, indent=2))

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"verdict={result['verdict']}\n")
            f.write(f"reasons={json.dumps(result['reasons'])}\n")


if __name__ == "__main__":
    main()
