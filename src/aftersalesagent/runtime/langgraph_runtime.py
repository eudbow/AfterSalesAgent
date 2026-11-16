"""单Agent图：文件Checkpoint + 请求账本 + 独立业务事务，保留框架无关应用接口。"""

from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, model_validator

from aftersalesagent.business.service import RetailService
from aftersalesagent.harness.execution import invoke_safely, run_sync
from aftersalesagent.harness.journal import Journal, redact
from aftersalesagent.models.base import ModelBackend
from aftersalesagent.models.errors import ModelBackendFault
from aftersalesagent.schemas.contracts import (
    ConfirmationEvent,
    MessageEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    SessionView,
    ToolCall,
    ToolContext,
    ToolResult,
    TurnResult,
    UserEvent,
)
from aftersalesagent.skills.base import SkillContent, SkillProvider, text_hash
from aftersalesagent.tools.registry import ToolRegistry


def policy_citation_requested(message: str) -> bool:
    """Conservative guard for explicit Retail policy questions, not a semantic judge."""
    text = message.lower()
    question = re.search(
        r"政策|规定|依据|时效|到账|赔偿|能否|可不可以|是否|还能|还可以|能不能|"
        r"policy|eligible|can i|may i|refund time",
        text,
    )
    topic = re.search(
        r"取消|退货|退款|改址|修改|换货|支付|赔偿|"
        r"cancel|return|refund|exchange|address|payment",
        text,
    )
    return bool(question and topic)


def policy_claim_present(reply: str) -> bool:
    return bool(
        re.search(
            r"不能|可以|只能|必须|无需|规定|未规定|不支持|不允许|不符合|无法办理|"
            r"eligible|cannot|can only|must|policy|not allowed|not supported",
            reply.lower(),
        )
    )


class Frame(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    context: RunContext
    messages: list[ModelMessage]
    skill: SkillContent | None = None
    response: ModelResponse | None = None
    event: UserEvent | None = None
    model_index: int = 0
    status: str = "waiting_user"
    reply: str = ""
    evidence: dict[str, dict] = Field(default_factory=dict)
    policy_citation_required: bool = False
    policy_repairs: int = 0
    sources: list[dict] = Field(default_factory=list)
    reason: str | None = None
    stop: bool = False
    retryable: bool = False
    resume_node: str = "model"
    usage: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def coherent_skill_snapshot(self) -> Frame:
        if self.skill is None:
            if self.context.skill_id is not None or self.context.skill_version != "none":
                raise ValueError("Session Skill snapshot is missing")
        elif (
            self.context.skill_id != self.skill.skill_id
            or self.context.skill_version != self.skill.version
            or self.context.skill_sha256 != self.skill.content_sha256
            or self.context.skill_snapshot_sha256 != text_hash(self.skill.model_dump_json())
            or len(self.messages) < 2
            or self.messages[0].role != "system"
            or text_hash(self.messages[0].content) != self.skill.policy_sha256
            or self.messages[1] != ModelMessage(role="system", content=self.skill.prompt())
        ):
            raise ValueError("Session Skill snapshot or prompt mismatch")
        return self


class GraphState(TypedDict):
    # Checkpoint只存JSON值；不依赖任意Python对象反序列化。
    frame: dict[str, Any]


def checked_retail_reply(frame: Frame, content: str) -> tuple[str, list[dict], str | None]:
    if frame.context.knowledge_version != "retail_sim_v1":
        return content, [], None
    ids = re.findall(r"\[cite:([^\]\s]+)\]", content)
    invalid = any(cid not in frame.evidence for cid in ids)
    invalid = invalid or content.count("[cite:") != len(ids)
    if invalid:
        return "回答引用未通过资料校验，暂时无法提供可靠的政策结论。", [], "invalid_policy_citation"
    if frame.policy_citation_required and not ids and policy_claim_present(content):
        return (
            "回答缺少可核实的政策引用，暂时无法提供可靠的政策结论。",
            [],
            "missing_policy_citation",
        )
    return content, [frame.evidence[cid] for cid in dict.fromkeys(ids)], None


def safe_pending_explanation(content: str) -> tuple[str, bool]:
    """Omit paragraphs promising side effects the simulated app does not perform."""
    unsupported = re.compile(
        r"(?:会|将|已).{0,20}(?:收到|发送|寄出).{0,25}(?:邮件|email)|"
        r"(?:will|have).{0,30}(?:receive|send|sent).{0,20}(?:email|mail)|"
        r"(?:退款|refund).{0,15}(?:已到账|已完成|completed|received)",
        re.IGNORECASE,
    )
    paragraphs = content.split("\n\n")
    retained = [paragraph for paragraph in paragraphs if not unsupported.search(paragraph)]
    return "\n\n".join(retained).strip(), len(retained) != len(paragraphs)


class LangGraphRuntime:
    def __init__(
        self,
        backend: ModelBackend,
        service: RetailService,
        *,
        policy_text: str,
        checkpointer: BaseCheckpointSaver | None = None,
        journal: Journal | None = None,
        trace_directory: Path | None = None,
        registry: ToolRegistry | None = None,
        skill_provider: SkillProvider | None = None,
    ):
        self.backend, self.service, self.policy_text = backend, service, policy_text
        self.registry = registry or ToolRegistry(service)
        self.skill_provider = skill_provider
        self.journal = journal or Journal()
        self.trace_directory = trace_directory
        self._persistent = checkpointer is not None
        self._frames: dict[str, Frame] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._deadlines: dict[str, float] = {}
        builder = StateGraph(GraphState)
        builder.add_node("model", self._model)
        builder.add_node("tools", self._tools)
        builder.add_node("execute", self._execute)
        builder.add_conditional_edges(START, lambda state: state["frame"]["resume_node"])
        builder.add_conditional_edges("model", self._after_model)
        builder.add_conditional_edges("tools", self._after_tools)
        builder.add_edge("execute", END)
        self.graph = builder.compile(checkpointer=checkpointer)

    def _config(self, sid: str, limits=None) -> RunnableConfig:
        return {
            "configurable": {"thread_id": sid},
            "recursion_limit": 2 * (limits.max_model_calls if limits else 12) + 8,
        }

    async def _snapshot(self, sid: str) -> tuple[Frame, tuple[str, ...]]:
        if self._persistent:
            snapshot = await self.graph.aget_state(self._config(sid))
            if not snapshot.values:
                raise ValueError("Unknown runtime session")
            frame = Frame.model_validate(snapshot.values["frame"])
            if frame.context.environment_id != self.service.environment_id:
                raise ValueError("Checkpoint environment mismatch")
            self._check_skill(frame)
            self._check_knowledge(frame)
            return frame, snapshot.next
        if sid not in self._frames:
            raise ValueError("Unknown runtime session")
        frame = Frame.model_validate(self._frames[sid].model_dump(mode="json"))
        self._check_skill(frame)
        self._check_knowledge(frame)
        return frame, ()

    def _check_skill(self, frame: Frame) -> None:
        # 恢复只读会话快照，不重新解析 released，也不要求原始文件/版本库仍在。
        if frame.skill is not None:
            frame.skill.check_compatibility(
                frame.context.policy_version, self.policy_text, self.registry.specs()
            )

    def _check_knowledge(self, frame: Frame) -> None:
        if frame.context.knowledge_version != "retail_sim_v1":
            return
        index = self.registry.retail_index
        if index is None or frame.context.knowledge_fingerprint != index.fingerprint:
            raise ValueError("Retail session knowledge snapshot mismatch")
        index.validate_snapshot(frame.evidence)

    async def _save(self, frame: Frame) -> None:
        if self._persistent:
            # execute节点的后继为END：保存应用返回边界，不留下意外待执行节点。
            await self.graph.aupdate_state(
                self._config(frame.session_id),
                {"frame": frame.model_dump(mode="json")},
                as_node="execute",
            )
        else:
            self._frames[frame.session_id] = frame.model_copy(deep=True)

    async def start(self, context: RunContext) -> SessionView:
        if context.knowledge_version == "retail_sim_v1":
            index = self.registry.retail_index
            if index is None or context.knowledge_fingerprint != index.fingerprint:
                raise ValueError("Retail session knowledge snapshot mismatch")
        skill = None
        messages = [ModelMessage(role="system", content=self.policy_text)]
        if context.skill_id is not None:
            if self.skill_provider is None:
                raise ValueError("Selected Skill requires a provider")
            skill = await self.skill_provider.load(context.skill_id, context.skill_version)
            if skill.skill_id != context.skill_id or (
                context.skill_version != "released" and skill.version != context.skill_version
            ):
                raise ValueError("Skill provider returned a different version")
            skill.check_compatibility(
                context.policy_version, self.policy_text, self.registry.specs()
            )
            context = RunContext.model_validate(
                context.model_dump()
                | {
                    "skill_version": skill.version,
                    "skill_sha256": skill.content_sha256,
                    "skill_snapshot_sha256": text_hash(skill.model_dump_json()),
                }
            )
            messages.append(ModelMessage(role="system", content=skill.prompt()))
        # 所有加载/兼容性检查先于业务会话创建，失败时不能悄悄降级成无 Skill。
        view = await run_sync(lambda: self.service.start(context))
        await run_sync(lambda: self.journal.register(view.session_id))
        frame = Frame(
            session_id=view.session_id,
            context=context,
            messages=messages,
            skill=skill,
        )
        index = self.registry.retail_index
        if index is not None and hasattr(index, "fixed_evidence"):
            from aftersalesagent.policies.retail_v2 import capability_prompt, validate_tools

            validate_tools(self.registry.specs())
            frame.evidence.update(index.fixed_evidence())
            frame.messages.append(ModelMessage(role="system", content=capability_prompt()))
        await self._save(frame)
        return view

    async def get_state(self, session_id: str) -> SessionView:
        await self._snapshot(session_id)
        return await run_sync(lambda: self.service.view(session_id))

    async def _emit(self, frame: Frame, kind: str, payload: dict) -> None:
        payload = dict(payload)
        payload["request_event_id"] = frame.event.event_id if frame.event else None
        await run_sync(
            lambda: self.journal.append(frame.session_id, kind, payload, self.service.clock())
        )

    def _truncate(self, frame: Frame, reason: str) -> None:
        frame.stop, frame.status, frame.reason = True, "truncated", reason
        frame.reply = "Execution budget reached; no further tools were run."
        # 已返回工具提议但未执行时补齐对应观察，避免下一轮出现悬空tool_call_id。
        pending = {}
        for message in frame.messages:
            for call in message.tool_calls:
                pending[call.call_id] = call
            if message.role == "tool":
                pending.pop(message.tool_call_id, None)
        for call in pending.values():
            frame.messages.append(
                ModelMessage(
                    role="tool",
                    tool_call_id=call.call_id,
                    content='{"status":"error","error":{"code":"policy_denied",'
                    '"message":"Budget reached; this tool call was not executed."}}',
                )
            )

    async def _budget(self, frame: Frame, kind: str) -> bool:
        assert frame.event is not None
        event_id = frame.event.event_id
        limits = frame.context.limits
        usage = await run_sync(lambda: self.journal.usage(frame.session_id))
        turn = await run_sync(lambda: self.journal.request(frame.session_id, event_id))
        if time.monotonic() >= self._deadlines[frame.session_id]:
            self._truncate(frame, "active_time_limit")
        elif usage["reported_tokens"] >= limits.max_tokens:
            self._truncate(frame, "reported_token_limit")
        elif turn["tokens"] >= limits.max_turn_tokens:
            self._truncate(frame, "turn_token_limit")
        elif kind == "model" and usage["model_calls"] >= limits.max_model_calls:
            self._truncate(frame, "model_call_limit")
        elif kind == "model" and turn["model_calls"] >= limits.max_turn_model_calls:
            self._truncate(frame, "turn_model_call_limit")
        elif kind == "tool" and usage["tool_calls"] >= limits.max_tool_calls:
            self._truncate(frame, "tool_call_limit")
        elif kind == "tool" and turn["tool_calls"] >= limits.max_turn_tool_calls:
            self._truncate(frame, "turn_tool_call_limit")
        return not frame.stop

    async def _charge(self, frame: Frame, kind: str, count: int = 1) -> None:
        assert frame.event is not None
        event_id = frame.event.event_id
        await run_sync(lambda: self.journal.charge(frame.session_id, event_id, kind, count))

    async def _model(self, state: GraphState) -> GraphState:
        frame = Frame.model_validate(state["frame"])
        if not await self._budget(frame, "model"):
            return {"frame": frame.model_dump(mode="json")}
        assert frame.event is not None
        event_id = frame.event.event_id
        usage_before = await run_sync(lambda: self.journal.usage(frame.session_id))
        turn_before = await run_sync(lambda: self.journal.request(frame.session_id, event_id))
        output_cap = min(
            frame.context.max_output_tokens,
            frame.context.limits.max_tokens - usage_before["reported_tokens"],
            frame.context.limits.max_turn_tokens - turn_before["tokens"],
        )
        request = ModelRequest(
            messages=frame.messages,
            tools=self.registry.specs(),
            model_name=frame.context.model_name,
            temperature=frame.context.temperature,
            max_output_tokens=max(1, output_cap),
        )
        await self._charge(frame, "model_calls")
        # 发出请求就存在未知消耗；只有收到完整usage后才能解除未知标记。
        await self._charge(frame, "unknown_usage_calls")
        await self._emit(
            frame,
            "model_request",
            {
                "model_call_index": frame.model_index,
                **request.model_dump(mode="json"),
                "policy_version": frame.context.policy_version,
                "skill_version": frame.context.skill_version,
                "skill_id": frame.context.skill_id,
                "skill_sha256": frame.context.skill_sha256,
                "skill_snapshot_sha256": frame.context.skill_snapshot_sha256,
                "knowledge_version": frame.context.knowledge_version,
            },
        )
        try:
            response = await asyncio.wait_for(
                self.backend.generate(request),
                timeout=max(0.0, self._deadlines[frame.session_id] - time.monotonic()),
            )
        except TimeoutError:
            self._truncate(frame, "active_time_limit")
            return {"frame": frame.model_dump(mode="json")}
        except ModelBackendFault as exc:
            if exc.observation is not None:
                await self._observe_usage(frame, exc.observation)
            await self._emit(
                frame,
                "model_error",
                {
                    **exc.public(),
                    "usage": exc.observation.model_dump(mode="json") if exc.observation else None,
                },
            )
            frame.stop, frame.status, frame.reason = True, "failed", exc.code
            frame.retryable, frame.resume_node = exc.retryable, "model"
            frame.reply = "模型请求未完成：" + exc.code + "。请查看配置或重试原请求。"
            return {"frame": frame.model_dump(mode="json")}
        except Exception:
            frame.stop, frame.status, frame.reason = True, "failed", "model_backend_error"
            frame.reply = "The model backend could not complete the request."
            return {"frame": frame.model_dump(mode="json")}
        response = ModelResponse.model_validate(redact(response.model_dump(mode="json")))
        frame.response = response
        frame.model_index += 1
        await self._observe_usage(frame, response)
        await self._emit(frame, "model_response", response.model_dump(mode="json"))
        frame.messages.append(
            ModelMessage(role="assistant", content=response.content, tool_calls=response.tool_calls)
        )
        usage = await run_sync(lambda: self.journal.usage(frame.session_id))
        assert frame.event is not None
        event_id = frame.event.event_id
        turn = await run_sync(lambda: self.journal.request(frame.session_id, event_id))
        if response.finish_reason == "length":
            self._truncate(frame, "model_output_limit")
        elif usage["reported_tokens"] > frame.context.limits.max_tokens:
            self._truncate(frame, "reported_token_limit")
        elif turn["tokens"] > frame.context.limits.max_turn_tokens:
            self._truncate(frame, "turn_token_limit")
        elif len(response.tool_calls) > 1:
            read_names = {s.name for s in self.registry.specs() if s.effect == "read"}
            if frame.context.knowledge_version == "retail_sim_v1" and all(
                c.name in read_names for c in response.tool_calls
            ):
                # The provider emitted a batch despite the one-call protocol. Execute only
                # the first read; the raw response remains in the journal for diagnosis.
                frame.response = response.model_copy(update={"tool_calls": response.tool_calls[:1]})
                frame.messages[-1] = ModelMessage(
                    role="assistant", content=response.content, tool_calls=response.tool_calls[:1]
                )
                await self._emit(
                    frame,
                    "model_error",
                    {
                        "code": "extra_read_calls_ignored",
                        "ignored_names": [c.name for c in response.tool_calls[1:]],
                    },
                )
            else:
                frame.stop, frame.status, frame.reason = True, "failed", "multiple_tool_calls"
                frame.reply = "This Retail runtime permits one tool call at a time."
                for call in response.tool_calls:
                    frame.messages.append(
                        ModelMessage(
                            role="tool",
                            tool_call_id=call.call_id,
                            content='{"status":"error","error":{"code":"invalid_arguments",'
                            '"message":"One tool call allowed; none executed."}}',
                        )
                    )
        elif not response.tool_calls:
            frame.stop, frame.status, frame.reply = True, "completed", response.content
            frame.reply, frame.sources, frame.reason = checked_retail_reply(frame, response.content)
            index = self.registry.retail_index
            v2_issue = None
            if index is not None and hasattr(index, "fixed_evidence") and frame.reason is None:
                from aftersalesagent.policies.retail_v2 import answer_issue

                v2_issue = answer_issue(response.content)
                if v2_issue:
                    frame.reason = "unsupported_policy_claim"
            if (
                index is not None
                and hasattr(index, "fixed_evidence")
                and frame.reason
                in {
                    "missing_policy_citation",
                    "invalid_policy_citation",
                    "unsupported_policy_claim",
                }
                and frame.policy_repairs == 0
                and isinstance(frame.event, MessageEvent)
            ):
                frame.policy_repairs += 1
                query = (
                    "。".join(
                        part
                        for part in frame.event.content.split("。")
                        if part and "我是" not in part and "#W" not in part
                    )
                    or "售后政策条件与应用能力"
                )
                call = ToolCall(
                    call_id=f"policy-repair-{frame.model_index}",
                    name="search_retail_policy",
                    arguments={"query": query},
                )
                await self._emit(
                    frame,
                    "model_error",
                    {"code": "policy_repair", "reason": frame.reason, "query": query},
                )
                frame.messages.append(
                    ModelMessage(
                        role="system",
                        content="上一回答依据不足。根据补充检索修订回答，"
                        "每个重要政策结论引用适用证据，能力引用 APP-CAP-V2；只允许这一次修订。"
                        + (v2_issue or ""),
                    )
                )
                frame.response = ModelResponse(tool_calls=[call])
                frame.messages.append(ModelMessage(role="assistant", content="", tool_calls=[call]))
                frame.stop, frame.status, frame.reason = False, "running", None
            elif v2_issue:
                frame.reply = "部分结论未能得到资料支持，暂时无法核实这项政策解释。"
                frame.sources = []
        return {"frame": frame.model_dump(mode="json")}

    async def _observe_usage(self, frame: Frame, response: ModelResponse) -> None:
        if response.input_tokens is not None and response.output_tokens is not None:
            await self._charge(frame, "unknown_usage_calls", -1)
        await self._charge(
            frame, "tokens", (response.input_tokens or 0) + (response.output_tokens or 0)
        )

    def _after_model(self, state: GraphState) -> str:
        return END if state["frame"]["stop"] else "tools"

    async def _tools(self, state: GraphState) -> GraphState:
        frame = Frame.model_validate(state["frame"])
        if not await self._budget(frame, "tool"):
            return {"frame": frame.model_dump(mode="json")}
        assert frame.response is not None and len(frame.response.tool_calls) == 1
        call = frame.response.tool_calls[0]
        view = await run_sync(lambda: self.service.view(frame.session_id))
        assert frame.event is not None
        event_id = frame.event.event_id
        context = ToolContext(
            session_id=view.session_id,
            environment_id=view.environment_id,
            authenticated_user_id=view.authenticated_user_id,
            idempotency_key=f"{event_id}:{frame.model_index}",
        )
        await self._charge(frame, "tool_calls")
        await self._emit(
            frame,
            "tool_call",
            {**call.model_dump(mode="json"), "transport": self.registry.transport},
        )
        if (
            hasattr(self.registry.retail_index, "fixed_evidence")
            and call.name == "cancel_pending_order"
        ):
            from aftersalesagent.policies.retail_v2 import cancellation_reason_supported

            user_text = "\n".join(m.content for m in frame.messages if m.role == "user")
            if not cancellation_reason_supported(call.arguments.get("reason"), user_text):
                result = ToolResult.model_validate(
                    {
                        "call_id": call.call_id,
                        "status": "error",
                        "error": {
                            "code": "policy_denied",
                            "message": (
                                "取消原因未在用户表达中得到核实。不得改写为允许原因；"
                                "请解释允许条件或追问。"
                            ),
                        },
                    }
                )
                await self._emit(frame, "tool_result", result.model_dump(mode="json"))
                frame.messages.append(
                    ModelMessage(
                        role="tool", content=result.model_dump_json(), tool_call_id=call.call_id
                    )
                )
                return {"frame": frame.model_dump(mode="json")}
        result = await invoke_safely(
            lambda: self.registry.invoke(call, context),
            call.call_id,
            on_retry=lambda code: self._emit(
                frame,
                "retry",
                {
                    "call_id": call.call_id,
                    "error_code": code,
                    "idempotency_key": context.idempotency_key,
                },
            ),
        )
        await self._emit(frame, "tool_result", result.model_dump(mode="json"))
        if result.error and result.error.code in {"infrastructure_error", "timeout"}:
            frame.stop, frame.status, frame.reason = True, "failed", result.error.code
            frame.retryable, frame.resume_node = result.error.retryable, "tools"
            frame.reply = result.error.message
            if not frame.retryable:
                frame.messages.append(
                    ModelMessage(
                        role="tool", content=result.model_dump_json(), tool_call_id=call.call_id
                    )
                )
            return {"frame": frame.model_dump(mode="json")}
        frame.messages.append(
            ModelMessage(role="tool", content=result.model_dump_json(), tool_call_id=call.call_id)
        )
        if call.name == "search_retail_policy" and result.status == "ok":
            frame.policy_citation_required = True
            for record in result.data["evidence"]:
                frame.evidence[record["chunk_id"]] = record
        latest = await run_sync(lambda: self.service.view(frame.session_id))
        if latest.pending_action is not None:
            await self._emit(frame, "pending_action", latest.pending_action.model_dump(mode="json"))
            frame.stop, frame.status = True, "waiting_confirmation"
            explanation = frame.response.content.strip() if frame.response else ""
            if explanation:
                explanation, filtered = safe_pending_explanation(explanation)
                if filtered:
                    await self._emit(
                        frame, "model_error", {"code": "unsupported_app_promise_filtered"}
                    )
                    frame.messages[-2] = frame.messages[-2].model_copy(
                        update={"content": explanation}
                    )
            if explanation:
                frame.reply, frame.sources, frame.reason = checked_retail_reply(frame, explanation)
                frame.reply += "\n\n待确认操作：" + latest.pending_action.summary
            else:
                frame.reply = latest.pending_action.summary
        elif result.status == "ok" and isinstance(result.data, dict):
            if result.data.get("status") == "handoff_requested":
                frame.stop, frame.status = True, "handoff"
                frame.reply = "A simulated human handoff was requested."
        return {"frame": frame.model_dump(mode="json")}

    def _after_tools(self, state: GraphState) -> str:
        return END if state["frame"]["stop"] else "model"

    async def _recorded(self, frame: Frame) -> ToolResult | None:
        frame.sources = []
        assert isinstance(frame.event, ConfirmationEvent)
        event = frame.event
        # 即使预算已耗尽，也允许查回已经提交的结果；这里不会执行新写入。
        from sqlalchemy import select

        from aftersalesagent.business.database import operations

        def read():
            with self.service.store.transaction() as connection:
                row = connection.execute(
                    select(operations.c.result_json).where(
                        operations.c.action_id == event.action_id,
                        operations.c.session_id == frame.session_id,
                    )
                ).first()
                return None if row is None else ToolResult.model_validate(row[0])

        return await run_sync(read)

    async def _execute(self, state: GraphState) -> GraphState:
        frame = Frame.model_validate(state["frame"])
        frame.sources = []
        assert isinstance(frame.event, ConfirmationEvent)
        event = frame.event
        event = frame.event
        recorded = await self._recorded(frame)
        if recorded is None and event.approved:
            if not await self._budget(frame, "tool"):
                return {"frame": frame.model_dump(mode="json")}
            await self._charge(frame, "tool_calls")
        await self._emit(frame, "confirmation", event.model_dump(mode="json"))
        result = await invoke_safely(
            lambda: run_sync(
                lambda: self.service.confirm(frame.session_id, event.action_id, event.approved)
            ),
            event.action_id,
            on_retry=lambda code: self._emit(
                frame,
                "retry",
                {
                    "action_id": event.action_id,
                    "error_code": code,
                },
            ),
        )
        await self._emit(frame, "tool_result", result.model_dump(mode="json"))
        frame.messages.append(
            ModelMessage(
                role="system", content="Server confirmation result: " + result.model_dump_json()
            )
        )
        if result.error:
            frame.status = (
                "failed" if result.error.code in {"infrastructure_error", "timeout"} else "rejected"
            )
            frame.reason, frame.retryable = result.error.code, result.error.retryable
            frame.reply = result.error.message
        elif isinstance(result.data, dict) and result.data.get("executed"):
            history = await run_sync(lambda: self.journal.events(frame.session_id))
            if not any(
                e.kind == "state_change" and e.payload.get("action_id") == event.action_id
                for e in history
            ):
                await self._emit(
                    frame,
                    "state_change",
                    {"action_id": event.action_id, "order": result.data.get("order")},
                )
            order = result.data.get("order")
            status = order.get("status") if isinstance(order, dict) else None
            frame.status = "completed"
            frame.reply = {
                "cancelled": "订单已取消，模拟退款记录已保存。",
                "return requested": "退货申请已提交；当前尚未退款。",
            }.get(str(status), "已完成本次确认的模拟操作，请查看最新订单快照。")
        else:
            frame.status, frame.reply = "waiting_user", "已拒绝操作，订单保持原样。"
        frame.stop = True
        return {"frame": frame.model_dump(mode="json")}

    async def advance(self, session_id: str, user_event: UserEvent) -> TurnResult:
        lock = self._locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            frame, next_nodes = await self._snapshot(session_id)
            cached = await run_sync(lambda: self.journal.claim(session_id, user_event))
            if cached is not None:
                if trace_path := self.trace_directory:
                    await run_sync(lambda: self.journal.export(session_id, trace_path))
                return cached
            request = await run_sync(lambda: self.journal.request(session_id, user_event.event_id))
            same_event = frame.event is not None and frame.event.event_id == user_event.event_id
            already_finished = same_event and frame.stop and not frame.retryable and not next_nodes
            started = time.monotonic()
            reservation, remaining = await run_sync(
                lambda: self.journal.reserve(session_id, user_event.event_id, frame.context.limits)
            )
            self._deadlines[session_id] = started + remaining
            if not same_event:
                frame.event, frame.response, frame.model_index = user_event, None, 0
                frame.policy_repairs = 0
                frame.policy_citation_required = (
                    frame.context.knowledge_version == "retail_sim_v1"
                    and isinstance(user_event, MessageEvent)
                    and policy_citation_requested(user_event.content)
                )
                frame.stop, frame.retryable, frame.reason = False, False, None
                frame.status, frame.reply = "running", ""
                frame.sources = []
                if isinstance(user_event, ConfirmationEvent):
                    frame.resume_node = "execute"
                else:
                    previous = await run_sync(lambda: self.service.view(session_id))
                    await run_sync(lambda: self.service.invalidate_pending(session_id))
                    frame.resume_node = "model"
                    frame.messages.append(ModelMessage(role="user", content=user_event.content))
                    if previous.pending_action is not None:
                        # 新消息使旧卡片失效，必须把服务端事实同步给模型，避免它仍引用历史提议。
                        frame.messages.append(
                            ModelMessage(
                                role="system",
                                content="Application state update: "
                                + json.dumps(
                                    {
                                        "superseded_action_id": previous.pending_action.action_id,
                                        "active_pending_action": None,
                                        "instruction": (
                                            "The previous confirmation card was invalidated "
                                            "by this new user message. Textual approval does "
                                            "not commit an action. If the customer still wants "
                                            "a valid operation, call the matching prepare tool "
                                            "again with the latest details to create a fresh "
                                            "card. Do not ask the customer to click the "
                                            "invalidated card."
                                        ),
                                    }
                                ),
                            )
                        )
                    await self._emit(frame, "user_message", user_event.model_dump(mode="json"))
            try:
                if not already_finished:
                    can_resume = same_event and bool(next_nodes)
                    if not can_resume:
                        frame.stop, frame.retryable, frame.reason = False, False, None
                        frame.status = "running"
                    value = await self.graph.ainvoke(
                        None if can_resume else {"frame": frame.model_dump(mode="json")},
                        self._config(session_id, frame.context.limits),
                        durability="sync" if self._persistent else None,
                    )
                    frame = Frame.model_validate(value["frame"])
            except asyncio.CancelledError:
                # 账本保留running，恢复时仍用原event_id；不伪造完成结果。
                raise
            except Exception:
                # 最近完成的节点已经Checkpoint；保留相同事件才能安全恢复。
                frame, pending_nodes = await self._snapshot(session_id)
                if pending_nodes and pending_nodes[0] in {"model", "tools", "execute"}:
                    frame.resume_node = pending_nodes[0]
                frame.stop = True
                frame.status, frame.reason, frame.retryable = (
                    "failed",
                    "infrastructure_error",
                    True,
                )
                frame.reply = "Execution was interrupted; retry the same event."
            elapsed = time.monotonic() - started
            if frame.reason == "active_time_limit":
                # Windows定时器可能稍早唤醒。超时已发生就消耗本次完整预留，
                # 防止几毫秒余量允许下一条消息再次发出模型请求。
                elapsed = max(elapsed, remaining)
            await run_sync(lambda: self.journal.settle(reservation, elapsed))
            frame.usage = await run_sync(lambda: self.journal.usage(session_id))
            await self._save(frame)
            trace_history = await run_sync(lambda: self.journal.events(session_id))
            ended = any(
                item.kind == "turn_end"
                and item.payload.get("request_event_id") == user_event.event_id
                for item in trace_history
            )
            if not already_finished or not ended:
                await self._emit(
                    frame,
                    "turn_end",
                    {
                        "status": frame.status,
                        "termination_reason": frame.reason,
                        "retryable": frame.retryable,
                        "usage": frame.usage,
                        "active_seconds": time.monotonic() - started,
                        "policy_version": frame.context.policy_version,
                        "skill_version": frame.context.skill_version,
                        "skill_id": frame.context.skill_id,
                        "skill_sha256": frame.context.skill_sha256,
                        "skill_snapshot_sha256": frame.context.skill_snapshot_sha256,
                        "knowledge_version": frame.context.knowledge_version,
                    },
                )
            events = await run_sync(lambda: self.journal.events(session_id, request["trace_start"]))
            result = TurnResult.model_validate(
                {
                    "session": await run_sync(lambda: self.service.view(session_id)),
                    "reply": frame.reply,
                    "sources": frame.sources,
                    "status": frame.status,
                    "events": events,
                    "termination_reason": frame.reason,
                }
            )
            await run_sync(
                lambda: self.journal.finish(
                    session_id, user_event.event_id, result, retryable=frame.retryable
                )
            )
            if trace_path := self.trace_directory:
                await run_sync(lambda: self.journal.export(session_id, trace_path))
            return result
