"""
golden_eval -- scores a candidate answer against a golden reference answer.

This is the "regression testing against known-good behavior" gate: not an
exact-match check (too strict for generative text), but a semantic-similarity
judgment from the LLM itself, since no embedding model/vector DB is used in
this project (see README for why).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_provider import get_provider, extract_json  # noqa: E402
from data_utils import load_golden_dataset, load_sample_candidate_runs  # noqa: E402

PROMPT_TEMPLATE = """You are grading whether a CANDIDATE answer means the same \
thing as a GOLDEN (reference, known-correct) answer to the same question. \
Minor wording differences don't matter -- only whether the substance matches.

QUESTION: {question}

GOLDEN ANSWER: {golden_answer}

CANDIDATE ANSWER: {candidate_answer}

Respond with ONLY a JSON object, no other text:
{{"score": <float 0.0-1.0, where 1.0 means fully equivalent in meaning>, \
"reasoning": "<one sentence>"}}
"""


def golden_eval(question: str, golden_answer: str, candidate_answer: str, provider=None) -> dict:
    provider = provider or get_provider()
    prompt = PROMPT_TEMPLATE.format(
        question=question, golden_answer=golden_answer, candidate_answer=candidate_answer
    )
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
        result = golden_eval(item["question"], item["golden_answer"], run["candidate_answer"], provider)
        print(f"[{run['id']}] {run['label']}")
        print(f"  question:   {item['question']}")
        print(f"  candidate:  {run['candidate_answer']}")
        print(f"  -> score={result['score']:.2f}  {result['reasoning']}\n")


if __name__ == "__main__":
    _demo()
