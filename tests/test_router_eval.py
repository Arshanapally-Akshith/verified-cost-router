"""Unit tests for verified_cost_router.eval.router_eval."""

from __future__ import annotations

import pytest
from fakes import ScriptedClassifier

from verified_cost_router.data_prep.adversarial_eval import ComplexityItem
from verified_cost_router.eval.router_eval import evaluate_complexity_items


def _item(id_: str, query: str) -> ComplexityItem:
    return ComplexityItem(id=id_, category="complexity_mislabeled", query=query, true_complexity="complex", rationale="r")


def test_correct_and_incorrect_classifications_are_recorded():
    items = [_item("cx1", "q1"), _item("cx2", "q2")]
    classifier = ScriptedClassifier({"q1": "complex", "q2": "simple"})

    result = evaluate_complexity_items(items, classifier)

    outcomes = {o.item_id: o for o in result.outcomes}
    assert outcomes["cx1"].correct is True
    assert outcomes["cx2"].correct is False


def test_complex_recall_is_fraction_correctly_labeled_complex():
    items = [_item("cx1", "q1"), _item("cx2", "q2"), _item("cx3", "q3")]
    classifier = ScriptedClassifier({"q1": "complex", "q2": "complex", "q3": "simple"})

    result = evaluate_complexity_items(items, classifier)

    assert result.complex_recall == pytest.approx(2 / 3)


def test_complex_recall_is_vacuously_one_for_empty_input():
    result = evaluate_complexity_items([], ScriptedClassifier({}))
    assert result.complex_recall == 1.0


def test_all_items_are_classified():
    items = [_item("cx1", "q1"), _item("cx2", "q2")]
    classifier = ScriptedClassifier({"q1": "complex", "q2": "complex"})

    evaluate_complexity_items(items, classifier)

    assert classifier.calls == ["q1", "q2"]
