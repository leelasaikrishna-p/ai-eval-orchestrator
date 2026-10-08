"""
llm_judge -- rubric-based grading of an answer's overall quality
(correctness, clarity, tone), independent of exact wording.

This is the "LLM-as-judge" pattern: scalable, subjective quality scoring
that a deterministic assertion can't express.

Two modes:
  - no reference (the original): the judge sees only the question and the
    answer, so "correctness" is the model's own belief about the product.
    Measured, it rated fluent wrong answers 8-10/10 -- it can't know the
    facts of a product it has never seen.
  - reference-guided: the judge also gets the known-correct answer and
    grades correctness against it. Pass `reference_answer` to use it.
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
{{"breakdown": {{"correctness": <0-10>, "clarity": <0-10>, "tone": <0-10>}}, \
"reasoning": "<one sentence>"}}
"""

REFERENCE_PROMPT_TEMPLATE = """You are grading a customer-support ANSWER to a QUESTION \
on a 0-10 rubric across three dimensions. A REFERENCE ANSWER, known to be \
correct, is provided.
- correctness: judge ONLY against the REFERENCE ANSWER, not your own \
knowledge of the product.
  9-10: consistent with the reference; nothing contradicts it.
  4-8: consistent but missing or blurring part of the reference.
  0-3: contradicts the reference, or adds a factual claim the reference \
does not support.
- clarity: is it easy to understand, free of jargon or ambiguity?
- tone: is it professional and appropriately concise?

QUESTION: {question}

REFERENCE ANSWER: {reference_answer}

ANSWER: {answer}

Respond with ONLY a JSON object, no other text:
{{"breakdown": {{"correctness": <0-10>, "clarity": <0-10>, "tone": <0-10>}}, \
"reasoning": "<one sentence>"}}
"""

JUDGE_DIMENSIONS = ("correctness", "clarity", "tone")


def llm_judge(question: str, answer: str, provider=None, reference_answer: str | None = None) -> dict:
    """The model scores each dimension; the overall score is computed HERE
    (an equal-weight mean), not taken from the model, so it is always
    consistent with the breakdown and explainable.

    With `reference_answer`, correctness is graded against it instead of
    the model's own beliefs. `mode` in the result records which was used."""
    provider = provider or get_provider()
    if reference_answer:
        prompt = REFERENCE_PROMPT_TEMPLATE.format(
            question=question, reference_answer=reference_answer, answer=answer
        )
    else:
        prompt = PROMPT_TEMPLATE.format(question=question, answer=answer)
    raw = provider.generate(prompt)
    result = extract_json(raw)
    breakdown = result.get("breakdown") or {}
    try:
        scores = [float(breakdown[d]) for d in JUDGE_DIMENSIONS]
    except (KeyError, TypeError, ValueError) as e:
        # A judgment without a usable breakdown is a failed check, not a score.
        raise ValueError(f"llm_judge needs numeric {JUDGE_DIMENSIONS} in 'breakdown'; got {breakdown!r}") from e
    if "score" in result:
        result["model_reported_score"] = result.pop("score")
    result["score"] = round(sum(scores) / len(scores), 1)
    result["mode"] = "reference" if reference_answer else "no_reference"
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
