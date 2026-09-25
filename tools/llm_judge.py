"""
llm_judge -- rubric-based grading of an answer's overall quality
(correctness, clarity, tone), independent of exact wording.

This is the "LLM-as-judge" pattern: scalable, subjective quality scoring
that a deterministic assertion can't express.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_provider import get_provider, extract_json  # noqa: E402
from data_utils import load_golden_dataset, load_sample_candidate_runs  # noqa: E402

PROMPT_TEMPLATE = """You are grading a customer-support ANSWER to a QUESTION \
on a 0-10 rubric across three dimensions:
- correctness: does it accurately answer what was asked?
- clarity: is it easy to understand, free of jargon or ambiguity?
- tone: is it professional and appropriately concise?

QUESTION: {question}

ANSWER: {answer}

Respond with ONLY a JSON object, no other text:
{{"score": <overall float 0-10>, "breakdown": {{"correctness": <0-10>, \
"clarity": <0-10>, "tone": <0-10>}}, "reasoning": "<one sentence>"}}
"""


def llm_judge(question: str, answer: str, provider=None) -> dict:
    provider = provider or get_provider()
    prompt = PROMPT_TEMPLATE.format(question=question, answer=answer)
    raw = provider.generate(prompt)
    result = extract_json(raw)
    result["score"] = float(result["score"])
    return result


def _demo():
    golden = load_golden_dataset()
    provider = get_provider()
    print(f"Provider: {provider.__class__.__name__}\n")
    for run in load_sample_candidate_runs():
        item = golden[run["id"]]
        result = llm_judge(item["question"], run["candidate_answer"], provider)
        print(f"[{run['id']}] {run['label']}")
        print(f"  -> score={result['score']:.1f}/10  breakdown={result['breakdown']}")
        print(f"  {result['reasoning']}\n")


if __name__ == "__main__":
    _demo()
