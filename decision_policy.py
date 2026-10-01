"""
Sign-off policy -- combines the four tools' verdicts into one decision:
approve or escalate to a human. Never a silent auto-reject: the two
outcomes are "ship it" or "a person looks at it," so the policy is
deliberately biased toward escalating when unsure -- a false escalation
costs a human a minute of review; a false approval ships a wrong answer.

The gate STRUCTURE is per-provider, not fixed, because a check's
trustworthiness depends on how reliable the model behind it actually is.
Measured against llama3.1:8b on our own test set:
  - golden_eval:         0 false positives, correctly zeroed a hallucination
  - groundedness_check:  1 false positive (flagged a correct paraphrase)
  - llm_judge:           1 false negative (rated a wrong answer 8/10)
  - drift_check:         no LLM involved -- always trustworthy

So for Ollama: golden_eval + drift_check are trusted hard gates (Tier 1);
groundedness_check and llm_judge are demoted to advisory signals (Tier 2)
that can each independently trigger escalation (OR, not AND -- see below)
but never independently trigger approval.

Profiles are keyed by provider AND model ("gemini/gemini-3.5-flash-lite"),
because trust is a property of the model. A model proven reliable enough on
a labeled set can graduate a check from Tier 2 into a Tier 1 hard gate; an
unmeasured model always gets the conservative profile.
"""
from __future__ import annotations

# Trust belongs to a specific MODEL, not a provider: two Gemini models can
# judge very differently. Profiles are keyed "provider/model" and only exist
# for models actually measured against the labeled sample runs.
CONSERVATIVE = {
    "golden_eval_floor": 0.3,   # below this -> Tier 1 auto-escalate
    "judge_score_floor": 5.0,   # below this -> Tier 2 flag (0-10 scale)
    "tier2_mode": "or",         # either groundedness or judge flagging is enough
}

GATE_PROFILES = {
    # Measured: golden_eval 0 FP; groundedness 1 FP (flagged a correct
    # paraphrase); judge 1 FN (rated the subtly wrong answer 8/10).
    "ollama/llama3.1:8b": dict(CONSERVATIVE),
    # Measured: golden_eval 0 FP but missed the subtle error (0.80);
    # groundedness 3/3 correct; judge 2 FN (rated the hallucination 9/10 and
    # the subtly wrong answer 10/10). Keeps OR: AND would approve q5.
    "gemini/gemini-3.5-flash-lite": dict(CONSERVATIVE),
    # "anthropic/<model>": {..., "tier2_mode": "and"},  # once calibrated
}

# Provider-level defaults (used when no model is given) point at the
# measured default model for that provider.
PROVIDER_DEFAULTS = {"ollama": "ollama/llama3.1:8b", "gemini": "gemini/gemini-3.5-flash-lite"}


def resolve_profile(provider: str, model: str | None = None) -> tuple[str, dict]:
    """Return (profile_key, profile). An unmeasured model hasn't earned any
    trust, so it gets the conservative profile rather than a crash or a guess."""
    candidates = [f"{provider}/{model}"] if model else [PROVIDER_DEFAULTS.get(provider), provider]
    for key in candidates:
        if key in GATE_PROFILES:
            return key, GATE_PROFILES[key]
    return "conservative (unmeasured)", CONSERVATIVE


def sign_off(
    golden_result: dict,
    groundedness_result: dict | None = None,
    judge_result: dict | None = None,
    provider: str = "ollama",
    model: str | None = None,
) -> dict:
    """Combine one candidate answer's per-answer checks into a verdict.

    `groundedness_result`/`judge_result` may be None when an orchestrator
    short-circuited and didn't run them -- that's fine for an early ESCALATE
    (a Tier 1 failure alone is sufficient reason), but insufficient to reach
    APPROVE: approving requires full evidence, escalating doesn't. This lets
    an orchestrator stop early on bad news without stopping early on good
    news.

    Returns {"verdict": "approve"|"escalate", "reasons": [str, ...], "flags": [str, ...]}.
    """
    _, profile = resolve_profile(provider, model)
    reasons: list[str] = []
    flags: list[str] = []

    # Tier 1 -- hard gate. A failure here is sufficient to escalate on its
    # own, even if Tier 2 checks were never run.
    if golden_result["score"] < profile["golden_eval_floor"]:
        reasons.append(
            f"golden_eval score {golden_result['score']:.2f} below floor "
            f"{profile['golden_eval_floor']} -- doesn't match known-good behavior"
        )
        return {"verdict": "escalate", "reasons": reasons, "flags": flags}

    # Reaching APPROVE requires full evidence -- an orchestrator that skipped
    # a Tier 2 check hasn't earned the right to approve, only to keep going.
    if groundedness_result is None or judge_result is None:
        return {
            "verdict": "escalate",
            "reasons": ["insufficient evidence: not all checks were run before concluding"],
            "flags": flags,
        }

    # Tier 2 -- advisory, combined per the provider's profile
    groundedness_flagged = not groundedness_result["grounded"]
    judge_flagged = judge_result["score"] < profile["judge_score_floor"]

    if groundedness_flagged:
        flags.append(
            f"groundedness_check flagged unsupported claims: {groundedness_result['unsupported_claims']}"
        )
    if judge_flagged:
        flags.append(f"llm_judge score {judge_result['score']:.1f} below floor {profile['judge_score_floor']}")

    tier2_triggers = (
        (groundedness_flagged or judge_flagged)
        if profile["tier2_mode"] == "or"
        else (groundedness_flagged and judge_flagged)
    )
    if tier2_triggers:
        reasons.extend(flags)

    verdict = "escalate" if reasons else "approve"
    return {"verdict": verdict, "reasons": reasons, "flags": flags if verdict == "approve" else []}


def run_level_check(drift_result: dict) -> dict:
    """Gates the RUN (this model/prompt/deployment), not a single answer."""
    if drift_result["drifted"]:
        flagged_metrics = [m for m, d in drift_result["details"].items() if d["flagged"]]
        return {"verdict": "escalate", "reasons": [f"drift detected in: {', '.join(flagged_metrics)}"]}
    return {"verdict": "approve", "reasons": []}
