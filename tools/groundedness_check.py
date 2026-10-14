"""
groundedness_check -- verifies every factual claim in a RAG-style answer
actually traces back to the retrieved context, instead of being invented.

This is the single most important GenAI-specific quality metric when
retrieval is involved: it's a direct proxy for hallucination rate.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_provider import get_provider, extract_json  # noqa: E402
from data_utils import load_golden_dataset, load_sample_candidate_runs  # noqa: E402
from tools.untrusted import untrusted_block  # noqa: E402

PROMPT_TEMPLATE = """You are checking whether an ANSWER is fully grounded in \
a CONTEXT passage -- i.e. every factual claim the answer makes either appears \
in the context or is a direct, necessary consequence of it. Flag any claim \
that goes beyond what the context actually supports, even if it sounds \
plausible or helpful.

{untrusted_note}

CONTEXT: {context}

ANSWER:
{answer}

Respond with ONLY a JSON object, no other text:
{{"grounded": <true or false>, "unsupported_claims": ["<claim not supported \
by the context>", ...], "reasoning": "<one sentence>"}}
"""


# Claim-by-claim mode: the model labels each claim, and code decides
# "grounded" from the labels. Built to cut false alarms on answers that
# paraphrase the context, apply a general rule to the user's case, or repeat
# the user's own question, without loosening on invented features.
CLAIMS_PROMPT_TEMPLATE = """You are checking whether an ANSWER to a user's \
QUESTION is grounded in a CONTEXT passage. Split the answer into its \
individual factual claims and label each one:
- "stated": the context says it, possibly in other words, or the claim is \
part of what the context says.
- "follows": it follows from the context applied to the user's situation in \
the QUESTION (for example, the context gives a general rule and the claim \
applies it to the user's case).
- "from_question": it only repeats something the user said in the QUESTION.
- "unsupported": it adds something the context does not say (a feature, \
option, number, time limit, condition or exception) or contradicts the \
context. A plausible or helpful addition is still "unsupported".
Leaving something out is not a claim: judge only what the answer says.

{untrusted_note}

QUESTION: {question}

CONTEXT: {context}

ANSWER:
{answer}

Give each claim exactly one label, one of these four words: stated, follows, \
from_question, unsupported. For every claim that is not unsupported, copy \
into "evidence" the exact words from the CONTEXT or QUESTION it rests on.

Respond with ONLY a JSON object, no other text:
{{"claims": [{{"claim": "<one claim>", "label": "<one label>", \
"evidence": "<exact words copied from the context or question>"}}], \
"reasoning": "<one sentence>"}}
"""

CLAIM_LABELS = ("stated", "follows", "from_question", "unsupported")

_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "are", "be", "it",
         "its", "that", "this", "with", "as", "at", "by", "can", "you", "your", "if", "any"}


def evidence_found(evidence: str, sources: str, threshold: float = 0.6) -> bool:
    """Pure function: does the quoted evidence actually come from the context
    or question? Most of its content words must appear there; the model's
    label alone is not trusted."""
    words = [w for w in _WORD.findall((evidence or "").lower()) if w not in _STOP]
    if not words:
        return False
    vocab = set(_WORD.findall(sources.lower()))
    return sum(w in vocab for w in words) / len(words) >= threshold


def evidence_specific(evidence: str, context: str, max_share: float = 0.8) -> bool:
    """Pure function: real support is a specific phrase. Quoting (nearly) the
    whole context is what a model does when it can't point to anything, so
    evidence covering more than `max_share` of a context of 15+ words
    doesn't count."""
    ctx = _WORD.findall(context.lower())
    if len(ctx) < 15:
        return True
    return len(_WORD.findall((evidence or "").lower())) <= max_share * len(ctx)


def grounded_from_claims(claims: list[dict], context: str, question: str = "") -> tuple[bool, list[str]]:
    """Pure function: grounded unless a claim is unsupported. A claim counts
    as unsupported if the model says so, if its label is unknown, or if its
    evidence isn't found in the context or question, or is just the whole
    context."""
    unsupported = []
    for c in claims:
        label = str(c.get("label", "")).strip().lower()
        ev = c.get("evidence", "")
        ok = (label in CLAIM_LABELS[:3] and evidence_found(ev, f"{context}\n{question}")
              and evidence_specific(ev, context))
        if not ok:
            unsupported.append(c.get("claim", ""))
    return not unsupported, unsupported


def mode_for(provider) -> str:
    """The groundedness mode measured for this exact model ("whole" unless a
    model has earned "claims"; see GATE_PROFILES in decision_policy.py)."""
    from decision_policy import resolve_profile

    _, profile = resolve_profile(provider.name, getattr(provider, "model", None))
    return profile.get("groundedness_mode", "whole")


def groundedness_check(context: str, answer: str, provider=None, question: str | None = None,
                       mode: str | None = None) -> dict:
    provider = provider or get_provider()
    mode = mode or mode_for(provider)
    note, block = untrusted_block(answer)
    if mode == "claims":
        prompt = CLAIMS_PROMPT_TEMPLATE.format(untrusted_note=note, question=question or "(not given)",
                                               context=context, answer=block)
        result = extract_json(provider.generate(prompt))
        claims = result.get("claims")
        if not isinstance(claims, list):
            raise ValueError("claims mode: no claims list in the response")
        result["grounded"], result["unsupported_claims"] = grounded_from_claims(claims, context, question or "")
        return result
    prompt = PROMPT_TEMPLATE.format(untrusted_note=note, context=context, answer=block)
    raw = provider.generate(prompt)
    result = extract_json(raw)
    result["grounded"] = bool(result["grounded"])
    result.setdefault("unsupported_claims", [])
    return result


def _demo():
    golden = load_golden_dataset()
    provider = get_provider()
    print(f"Provider: {provider.__class__.__name__}\n")
    for run in load_sample_candidate_runs():
        item = golden[run["id"]]
        result = groundedness_check(item["context"], run["candidate_answer"], provider)
        print(f"[{run['id']}] {run['label']}")
        print(f"  answer: {run['candidate_answer']}")
        print(f"  -> grounded={result['grounded']}  unsupported={result['unsupported_claims']}")
        print(f"  {result['reasoning']}\n")


if __name__ == "__main__":
    _demo()
