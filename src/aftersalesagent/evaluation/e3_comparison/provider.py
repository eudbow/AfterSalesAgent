"""逐请求账本与跨阶段模型/指纹检查；原始提供方错误不含Key或HTTP正文。"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from aftersalesagent.models.errors import ModelBackendFault
from aftersalesagent.schemas.contracts import ModelRequest, ModelResponse


class MonitoredBackend:
    def __init__(self, backend, ledger: Path, episode_id: str, model: str):
        self.backend, self.ledger, self.episode_id, self.model = backend, ledger, episode_id, model

    async def generate(self, request: ModelRequest) -> ModelResponse:
        rows = (
            [json.loads(s) for s in self.ledger.read_text("utf-8").splitlines()]
            if self.ledger.exists()
            else []
        )
        error = None
        try:
            response = await self.backend.generate(request)
        except ModelBackendFault as exc:
            response = exc.observation
            error = exc.code
        entry = {
            "episode_id": self.episode_id,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "request_model": request.model_name,
            "request_temperature": request.temperature,
            "error": error,
            "response": response.model_dump(mode="json") if response else None,
        }
        if response is not None and error is None:
            meta = response.provider_metadata
            old = [row["response"] for row in rows if row.get("response")]
            ids = {row["provider_metadata"].get("id") for row in old}
            fingerprints = {
                row["provider_metadata"]["system_fingerprint"]
                for row in old
                if row["provider_metadata"].get("system_fingerprint")
            }
            actual_fingerprint = meta.get("system_fingerprint")
            if (
                request.model_name != self.model
                or meta.get("model") != self.model
                or request.temperature != 0
            ):
                error = "model_configuration_or_provider_model_drift"
            elif fingerprints and actual_fingerprint and actual_fingerprint not in fingerprints:
                error = "model_provider_fingerprint_drift"
            elif not meta.get("id") or meta["id"] in ids:
                error = "model_provider_response_id_missing_or_duplicate"
            elif response.input_tokens is None or response.output_tokens is None:
                error = "model_usage_unknown"
        entry["error"] = error
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("ab") as stream:
            stream.write((json.dumps(entry, ensure_ascii=False) + "\n").encode())
        if error:
            raise ModelBackendFault(error, observation=response)
        assert response is not None
        return response

    async def aclose(self):
        await self.backend.aclose()
