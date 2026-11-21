import numpy as np
import pytest

from ml.training.phase2_utils import evaluate_predictions


def test_evaluation_selects_threshold_on_validation_and_scores_test_once():
    validation_labels = np.array([1, 0, 1, 0])
    validation_scores = np.array([0.9, 0.1, 0.8, 0.2])
    validation_sources = np.array(["sampoq", "sampoq", "ic_splc", "ic_splc"])
    test_labels = np.array([1, 0, 1, 0])
    test_scores = np.array([0.95, 0.05, 0.75, 0.25])
    test_sources = np.array(["sampoq", "sampoq", "ic_splc", "ic_splc"])

    result = evaluate_predictions(
        validation_labels,
        validation_scores,
        validation_sources,
        None,
        test_labels,
        test_scores,
        test_sources,
        None,
        include_curves=False,
    )

    assert result["threshold_selection"]["source"] == "validation"
    assert result["validation"]["roc_auc"] == pytest.approx(1.0)
    assert result["test"]["roc_auc"] == pytest.approx(1.0)
    assert result["test"]["average_precision"] == pytest.approx(1.0)
    assert result["test"]["counts"]["positive"] == 2
    assert result["test"]["counts"]["negative"] == 2
    assert result["test"]["per_source"]["sampoq"]["roc_auc"] == pytest.approx(1.0)
    assert result["test"]["per_source"]["ic_splc"]["roc_auc"] == pytest.approx(1.0)
    assert result["calibration"]["score_type"] == "score_platt_scaling"
    assert result["calibration"]["test"]["positive_count"] == 2
    assert result["calibration"]["test"]["negative_count"] == 2


def test_evaluation_handles_single_class_source_without_fabricating_auc():
    validation_labels = np.array([1, 0, 1, 0])
    validation_scores = np.array([0.9, 0.1, 0.8, 0.2])
    validation_sources = np.array(["a", "a", "b", "b"])
    test_labels = np.array([1, 0, 1, 1])
    test_scores = np.array([0.9, 0.1, 0.8, 0.7])
    test_sources = np.array(["a", "a", "b", "b"])

    result = evaluate_predictions(
        validation_labels,
        validation_scores,
        validation_sources,
        None,
        test_labels,
        test_scores,
        test_sources,
        None,
        include_curves=False,
    )

    assert result["test"]["per_source"]["b"]["roc_auc"] is None
    assert result["test"]["per_source"]["b"]["average_precision"] is None
