"""Scripted model for offline plumbing tests; it does not understand natural language."""

from collections import deque
from collections.abc import Iterable

from aftersalesagent.schemas.contracts import ModelRequest, ModelResponse


class MockBackend:
    def __init__(self, responses: Iterable[ModelResponse]):
        self._responses = deque(responses)
        self.requests: list[ModelRequest] = []

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request.model_copy(deep=True))
        if not self._responses:
            raise RuntimeError("Mock response script exhausted")
        return self._responses.popleft().model_copy(deep=True)
