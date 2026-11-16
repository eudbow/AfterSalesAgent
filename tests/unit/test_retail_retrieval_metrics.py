"""Hand-checkable ranking examples and failure cases for the offline evaluator."""

import importlib.util
import math
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "retail_metrics",
    Path(__file__).resolve().parents[2] / "scripts/evaluate_retail_rag_v2_retrieval.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_known_rank_example():
    result = module.rank_metrics(["a", "b", "d", "c"], {"b", "d"}, 2)
    assert result["recall@2"] == 0.5
    assert result["mrr@2"] == 0.5
    assert result["ndcg@2"] == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))
    assert module.rank_metrics(["b", "d"], {"b", "d"}, 3)["ndcg@3"] == 1


def test_absent_relevant_and_invalid_input():
    assert all(v == 0 for v in module.rank_metrics(["a"], {"b"}, 3).values())
    with pytest.raises(ValueError):
        module.rank_metrics(["b", "b"], {"b"}, 3)
    with pytest.raises(ValueError):
        module.rank_metrics(["b"], set(), 3)
