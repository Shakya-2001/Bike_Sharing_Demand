from __future__ import annotations

import pandas as pd

from app.training.evaluation import (
    decide_promotion,
    evaluate_model,
    measure_inference_ms,
)
from tests.training_fixtures import FixedModel, make_dataset


def test_first_model_is_promoted_when_there_is_no_champion() -> None:
    decision = decide_promotion(50.0, None, 500.0, 100.0)

    assert decision.promote is True


def test_better_and_fast_candidate_is_promoted() -> None:
    decision = decide_promotion(40.0, 50.0, 5.0, 100.0)

    assert decision.promote is True


def test_equal_candidate_is_promoted() -> None:
    decision = decide_promotion(50.0, 50.0, 5.0, 100.0)

    assert decision.promote is True


def test_worse_candidate_is_rejected_with_reason() -> None:
    decision = decide_promotion(60.0, 50.0, 5.0, 100.0)

    assert decision.promote is False
    assert "worse" in decision.reasons[0]


def test_slow_candidate_is_rejected_with_reason() -> None:
    decision = decide_promotion(40.0, 50.0, 150.0, 100.0)

    assert decision.promote is False
    assert "Inference time" in decision.reasons[0]


def test_candidate_failing_both_rules_gets_both_reasons() -> None:
    decision = decide_promotion(60.0, 50.0, 150.0, 100.0)

    assert decision.promote is False
    assert len(decision.reasons) == 2


def test_evaluate_model_scores_a_perfect_and_a_bad_model() -> None:
    test_df = pd.DataFrame(make_dataset(20)).assign(cnt=100)

    perfect = evaluate_model(FixedModel(100), test_df)
    off_by_ten = evaluate_model(FixedModel(110), test_df)

    assert perfect["rmse"] == 0.0
    assert perfect["mae"] == 0.0
    assert off_by_ten["rmse"] == 10.0
    assert off_by_ten["mae"] == 10.0


def test_measure_inference_returns_a_positive_time() -> None:
    test_df = make_dataset(20)

    elapsed = measure_inference_ms(FixedModel(1), test_df, repeats=5)

    assert elapsed >= 0.0
