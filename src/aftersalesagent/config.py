"""应用配置读取。只读取本进程环境变量，不自动打开或打印密钥文件。"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from aftersalesagent.schemas.contracts import RunLimits


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runtime: str = "langgraph"
    tool_transport: Literal["native", "mcp"] = "native"
    knowledge_transport: Literal["native", "mcp"] = "native"
    knowledge_config: str | None = None
    retail_rag: bool = False
    retail_rag_v2: bool = False
    retail_retrieval: Literal["bm25", "dense", "hybrid"] = "bm25"
    skill_id: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9-]{0,63}$")
    skill_version: str = "none"
    backend: str = "mock"
    model_name: str | None = None
    model_base_url: str = "https://api.deepseek.com"
    max_model_calls: int = Field(default=12, ge=1)
    max_tool_calls: int = Field(default=20, ge=1)
    max_active_seconds: float = Field(default=120, gt=0)
    max_tokens: int = Field(default=24000, ge=1)
    model_timeout_seconds: float = Field(default=25, gt=0, le=120)
    model_max_output_tokens: int = Field(default=1024, ge=1, le=8192)
    model_temperature: float = Field(default=0, ge=0, le=2)

    max_turn_model_calls: int = Field(default=6, ge=1)
    max_turn_tool_calls: int = Field(default=10, ge=1)
    max_turn_active_seconds: float = Field(default=30, gt=0)
    max_turn_tokens: int = Field(default=12000, ge=1)

    @model_validator(mode="after")
    def coherent_skill_selection(self) -> Settings:
        from aftersalesagent.schemas.contracts import RunContext

        if self.retail_rag_v2:
            self.retail_rag = True
            self.retail_retrieval = "hybrid"
        RunContext(
            environment_id="settings-validation",
            skill_id=self.skill_id,
            skill_version=self.skill_version,
        )
        return self

    def run_limits(self) -> RunLimits:
        return RunLimits.model_validate(self.model_dump(include=set(RunLimits.model_fields)))

    @classmethod
    def from_environment(cls, values: Mapping[str, str] | None = None) -> Settings:
        values = os.environ if values is None else values
        return cls(
            runtime=values.get("AFTERSALES_RUNTIME", "langgraph"),
            tool_transport=TypeAdapter(Literal["native", "mcp"]).validate_python(
                values.get("AFTERSALES_TOOL_TRANSPORT", "native")
            ),
            knowledge_transport=TypeAdapter(Literal["native", "mcp"]).validate_python(
                values.get("AFTERSALES_KNOWLEDGE_TRANSPORT", "native")
            ),
            retail_retrieval=TypeAdapter(Literal["bm25", "dense", "hybrid"]).validate_python(
                values.get("AFTERSALES_RETAIL_RETRIEVAL", "bm25")
            ),
            retail_rag=TypeAdapter(bool).validate_python(
                values.get("AFTERSALES_RETAIL_RAG", "false")
            ),
            retail_rag_v2=TypeAdapter(bool).validate_python(
                values.get("AFTERSALES_RETAIL_RAG_V2", "false")
            ),
            knowledge_config=values.get("AFTERSALES_KNOWLEDGE_CONFIG") or None,
            skill_id=values.get("AFTERSALES_SKILL_ID") or None,
            skill_version=values.get("AFTERSALES_SKILL_VERSION", "none"),
            backend=values.get("AFTERSALES_BACKEND", "mock"),
            model_name=values.get("AFTERSALES_MODEL_NAME") or None,
            model_base_url=values.get("AFTERSALES_MODEL_BASE_URL", "https://api.deepseek.com"),
            max_model_calls=int(values.get("AFTERSALES_MAX_MODEL_CALLS", "12")),
            max_tool_calls=int(values.get("AFTERSALES_MAX_TOOL_CALLS", "20")),
            max_active_seconds=float(values.get("AFTERSALES_MAX_ACTIVE_SECONDS", "120")),
            max_tokens=int(values.get("AFTERSALES_MAX_TOKENS", "24000")),
            model_timeout_seconds=float(values.get("AFTERSALES_MODEL_TIMEOUT_SECONDS", "25")),
            model_max_output_tokens=int(values.get("AFTERSALES_MODEL_MAX_OUTPUT_TOKENS", "1024")),
            model_temperature=float(values.get("AFTERSALES_MODEL_TEMPERATURE", "0")),
            max_turn_model_calls=int(values.get("AFTERSALES_MAX_TURN_MODEL_CALLS", "6")),
            max_turn_tool_calls=int(values.get("AFTERSALES_MAX_TURN_TOOL_CALLS", "10")),
            max_turn_active_seconds=float(values.get("AFTERSALES_MAX_TURN_ACTIVE_SECONDS", "30")),
            max_turn_tokens=int(values.get("AFTERSALES_MAX_TURN_TOKENS", "12000")),
        )
