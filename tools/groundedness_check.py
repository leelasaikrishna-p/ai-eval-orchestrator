"""
groundedness_check -- verifies every factual claim in a RAG-style answer
actually traces back to the retrieved context, instead of being invented.

This is the single most important GenAI-specific quality metric when
retrieval is involved: it's a direct proxy for hallucination rate.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_provider import get_provider, extract_json  # noqa: E402
from data_utils import load_golden_dataset, load_sample_candidate_runs  # noqa: E402

PROMPT_TEMPLATE = """You are checking whether an ANSWER is fully grounded in \
a CONTEXT passage -- i.e. every factual claim the answer makes either appears \
in the context or is a direct, necessary consequence of it. Flag any claim \
that goes beyond what the context actually supports, even if it sounds \
plausible or helpful.

CONTEXT: {context}

ANSWER: {answer}

Respond with ONLY a JSON object, no other text:
{{"grounded": <true or false>, "unsupported_claims": ["<claim not supported \
by the context>", ...], "reasoning": "<one sentence>"}}
"""


def groundedness_check(context: str, answer: str, provider=None) -> dict:
    provider = provider or get_provider()
    prompt = PROMPT_TEMPLATE.format(context=context, answer=answer)
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
