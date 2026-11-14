"""Paid-call wrapper tests with fixed responses, never a real API."""

import asyncio

import pytest

from aftersalesagent.evaluation.e4.budget import GenerationBudget
from aftersalesagent.evaluation.e4.execution import BudgetedBackend
from aftersalesagent.models.errors import ModelBackendFault
from aftersalesagent.schemas.contracts import ModelRequest, ModelResponse


class FixedBackend:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    async def generate(self, request):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response

    async def aclose(self):
        pass


def response(model="test-model", known=True):
    return ModelResponse(
        content="answer",
        input_tokens=10 if known else None,
        output_tokens=5 if known else None,
        provider_metadata={"model": model, "system_fingerprint": "stable"},
    )


def test_wrapper_records_known_usage_and_refuses_identity_drift(tmp_path):
    async def run():
        budget = GenerationBudget(tmp_path / "budget.sqlite")
        raw = FixedBackend([response(), response("different-model")])
        backend = BudgetedBackend(raw, budget, "run1", tmp_path / "identity.json")
        backend.call_key = "one"
        assert await backend.generate(ModelRequest(messages=[], tools=[]))
        backend.call_key = "two"
        with pytest.raises(ModelBackendFault):
            await backend.generate(ModelRequest(messages=[], tools=[]))
        assert raw.calls == 2
        assert budget.report() == {"limit": 300, "reserved_calls": 2, "states": {"known": 2}}
        budget.close()

    asyncio.run(run())


def test_wrapper_unknown_usage_prevents_next_network_call(tmp_path):
    async def run():
        budget = GenerationBudget(tmp_path / "budget.sqlite")
        raw = FixedBackend([response(known=False), response()])
        backend = BudgetedBackend(raw, budget, "run1", tmp_path / "identity.json")
        backend.call_key = "one"
        await backend.generate(ModelRequest(messages=[], tools=[]))
        backend.call_key = "two"
        with pytest.raises(ValueError, match="Unresolved"):
            await backend.generate(ModelRequest(messages=[], tools=[]))
        assert raw.calls == 1 and budget.report()["states"] == {"unknown": 1}
        budget.close()

    asyncio.run(run())


def test_provider_failure_keeps_observed_usage_and_missing_key_is_not_a_call(tmp_path):
    async def run():
        budget = GenerationBudget(tmp_path / "budget.sqlite")
        raw = FixedBackend([ModelBackendFault("model_protocol_error", observation=response())])
        backend = BudgetedBackend(raw, budget, "run1", tmp_path / "identity.json")
        with pytest.raises(ValueError, match="key"):
            await backend.generate(ModelRequest(messages=[], tools=[]))
        assert budget.report()["reserved_calls"] == 0
        backend.call_key = "one"
        with pytest.raises(ModelBackendFault):
            await backend.generate(ModelRequest(messages=[], tools=[]))
        assert budget.report()["states"] == {"known": 1}
        budget.close()

    asyncio.run(run())
