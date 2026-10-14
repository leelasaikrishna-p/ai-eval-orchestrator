"""
Offline tests for claim-by-claim groundedness: the code that decides
"grounded" from the model's claim labels and evidence, and the measurement
helpers. No model calls.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_provider import MockProvider, QuotaExhausted  # noqa: E402
from tools.groundedness_check import (  # noqa: E402
    evidence_found, evidence_specific, grounded_from_claims, groundedness_check, mode_for,
)
from tools.measure_agreement import check_row  # noqa: E402
from tools.measure_groundedness import expected_rates, flag_rates, review_load, run  # noqa: E402

CONTEXT = ("Refunded amounts are returned to the original payment method and typically post within "
           "5-10 business days. Ledgerly does not support refunding a payment to a different payment "
           "method than the one used originally.")
QUESTION = "If a customer paid by credit card, can I refund them to their bank account instead?"


def test_evidence_must_come_from_the_context_or_question():
    assert evidence_found("returned to the original payment method", CONTEXT)
    assert evidence_found("paid by credit card", CONTEXT + QUESTION)
    assert not evidence_found("the refund is sent as a check", CONTEXT)
    assert not evidence_found("", CONTEXT)


def test_quoting_the_whole_context_is_not_specific_evidence():
    assert evidence_specific("returned to the original payment method", CONTEXT)
    assert not evidence_specific(CONTEXT, CONTEXT)
    assert evidence_specific("Net 30", "Default payment terms are Net 30.")  # short context: always ok


def test_a_rule_applied_to_the_users_case_is_grounded():
    claims = [
        {"claim": "a card payment is refunded to the card", "label": "follows",
         "evidence": "returned to the original payment method"},
        {"claim": "the customer paid by card", "label": "from_question", "evidence": "paid by credit card"},
    ]
    assert grounded_from_claims(claims, CONTEXT, QUESTION) == (True, [])


def test_an_invented_detail_is_unsupported_whatever_label_the_model_gives():
    claims = [
        {"claim": "it goes back to the card", "label": "stated", "evidence": "returned to the original payment method"},
        # The model calls this "stated" but can only point at the whole passage.
        {"claim": "if the card expired, a check is sent", "label": "stated", "evidence": CONTEXT},
        {"claim": "refunds take one hour", "label": "stated|follows", "evidence": "typically post"},
    ]
    grounded, unsupported = grounded_from_claims(claims, CONTEXT, QUESTION)
    assert not grounded
    assert unsupported == ["if the card expired, a check is sent", "refunds take one hour"]


def test_claims_mode_decides_grounded_in_code_from_the_models_labels():
    reply = json.dumps({"claims": [
        {"claim": "refunds go to the card", "label": "follows", "evidence": "original payment method"},
        {"claim": "a check is sent instead", "label": "unsupported", "evidence": ""},
    ], "reasoning": "x"})
    r = groundedness_check(CONTEXT, "answer", MockProvider(reply), question=QUESTION, mode="claims")
    assert r["grounded"] is False and r["unsupported_claims"] == ["a check is sent instead"]


def test_expected_rates_average_each_answers_flags_over_its_runs():
    items = [
        {"id": "B", "human_label": "not_acceptable"},
        {"id": "G1", "human_label": "acceptable"},
        {"id": "G2", "human_label": "acceptable"},
    ]
    runs = [
        [{"id": "B", "grounded": False}, {"id": "G1", "grounded": False}, {"id": "G2", "grounded": True}],
        [{"id": "B", "grounded": True}, {"id": "G1", "grounded": True}, {"id": "G2", "error": "x"}],
    ]
    rates = flag_rates(runs, lambda row: None if "error" in row else not row["grounded"])
    assert rates == {"B": 0.5, "G1": 0.5, "G2": 0.0}
    r = expected_rates(rates, items)
    assert r["recall"] == 0.5 and r["false_alarms"] == 0.25
    # At 5% bad answers: 0.5 x 0.05 + 0.25 x 0.95
    assert review_load(r) == round(0.5 * 0.05 + 0.25 * 0.95, 3)


def test_a_used_up_daily_quota_stops_a_measurement_instead_of_recording_errors():
    class OutOfQuota(MockProvider):
        def generate(self, prompt, temperature=0.0):
            raise QuotaExhausted("daily quota exhausted")

    for call in (lambda: run(OutOfQuota(), runs=1),
                 lambda: check_row({"id": "L01", "candidate_answer": "x"},
                                   {"question": "q", "golden_answer": "g", "context": "c"}, OutOfQuota())):
        try:
            call()
        except QuotaExhausted:
            continue
        raise AssertionError("a used-up quota was recorded as an ordinary error")


def test_claim_mode_is_used_only_by_the_model_that_earned_it():
    class Model(MockProvider):
        def __init__(self, name, model):
            super().__init__()
            self.name, self.model = name, model

    assert mode_for(Model("gemini", "gemini-3.5-flash-lite")) == "claims"  # measured on held-out questions
    assert mode_for(Model("ollama", "llama3.1:8b")) == "whole"  # measured: claim mode hurt
    assert mode_for(Model("gemini", "some-unmeasured-model")) == "whole"  # no trust until measured
    assert mode_for(MockProvider()) == "whole"


def test_the_active_models_mode_is_used_unless_a_measurement_asks_for_one():
    reply = json.dumps({"claims": [{"claim": "x", "label": "stated", "evidence": "original payment method"}]})

    class Gemini(MockProvider):
        name, model = "gemini", "gemini-3.5-flash-lite"

    r = groundedness_check(CONTEXT, "answer", Gemini(reply), question=QUESTION)
    assert "claims" in r and r["grounded"] is True
    whole = groundedness_check(CONTEXT, "answer", Gemini('{"grounded": false, "unsupported_claims": ["x"]}'),
                               question=QUESTION, mode="whole")
    assert whole["grounded"] is False and "claims" not in whole


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} claim-groundedness tests passed")
