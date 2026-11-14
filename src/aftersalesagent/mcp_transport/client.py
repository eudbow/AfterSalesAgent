"""一个工作任务持有 SDK 上下文，跨 HTTP 请求通过队列调用，避免跨任务退出取消域。"""

import asyncio
import json
import os
import secrets
import sys
import time
from pathlib import Path
from typing import Any

from mcp import Client, StdioServerParameters

from aftersalesagent.schemas.contracts import ToolContext

ROOT = Path(__file__).resolve().parents[3]


class MCPConnection:
    def __init__(
        self,
        *,
        database: Path | None = None,
        session_id: str | None = None,
        allow_proposals=False,
        timeout=15.0,
        trace_path: Path | None = None,
        fault_after_commit: Path | None = None,
        fault_delay_seconds: float = 0,
        knowledge_config: Path | None = None,
    ):
        self.database, self.session_id = database, session_id
        self.allow_proposals, self.timeout = allow_proposals, timeout
        self.trace_path, self.fault_after_commit = trace_path, fault_after_commit
        self.fault_delay_seconds = fault_delay_seconds
        self.knowledge_config = knowledge_config
        self.capability = secrets.token_urlsafe(32)
        self.queue: asyncio.Queue = asyncio.Queue()
        self.task: asyncio.Task | None = None
        self.lock = asyncio.Lock()
        self.ready: asyncio.Future | None = None
        self.protocol_version: str | None = None
        self.discovery: list = []
        self.closed = False

    def parameters(self):
        args = ["-m", "aftersalesagent.mcp_transport.server"]
        if self.knowledge_config:
            args += ["--knowledge-config", str(self.knowledge_config.resolve())]
        if self.database:
            args += [
                "--database",
                str(self.database.resolve()),
                "--session-id",
                self.session_id or "",
            ]
        if self.allow_proposals:
            args.append("--allow-proposals")
        if self.fault_after_commit:
            args += ["--fault-after-commit", str(self.fault_after_commit.resolve())]
            if self.fault_delay_seconds:
                args += ["--fault-delay-seconds", str(self.fault_delay_seconds)]
        # 不继承用户模型Key；只保留启动Python/Windows所需的最小环境。
        env = {
            k: os.environ[k]
            for k in ("PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "COMSPEC", "PATHEXT")
            if k in os.environ
        }
        env.update(
            {
                "AFTERSALES_MCP_CAPABILITY": self.capability,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
            }
        )
        return StdioServerParameters(
            command=sys.executable, args=args, cwd=ROOT, env=env, encoding="utf-8"
        )

    async def _worker(self):
        try:
            async with Client(self.parameters(), read_timeout_seconds=self.timeout) as client:
                self.protocol_version = client.protocol_version
                self.discovery = (await client.list_tools()).tools
                assert self.ready is not None
                self.ready.set_result(None)
                while True:
                    item = await self.queue.get()
                    if item is None:
                        break
                    name, arguments, meta, future = item
                    try:
                        result = await client.call_tool(name, arguments, meta=meta)
                        if not future.done():
                            future.set_result(result)
                    except Exception as exc:
                        if not future.done():
                            future.set_exception(ConnectionError("MCP response unavailable"))
                        raise ConnectionError("MCP connection interrupted") from exc
        except Exception:
            if self.ready is not None and not self.ready.done():
                self.ready.set_exception(ConnectionError("MCP startup/discovery failed"))
        finally:
            while not self.queue.empty():
                item = self.queue.get_nowait()
                if item is not None and not item[-1].done():
                    item[-1].set_exception(ConnectionError("MCP connection closed"))

    async def connect(self):
        if self.closed:
            raise ConnectionError("MCP connection was closed")
        async with self.lock:
            if self.task is None or self.task.done():
                self.ready = asyncio.get_running_loop().create_future()
                self.task = asyncio.create_task(self._worker())
            assert self.ready is not None
            try:
                await asyncio.wait_for(asyncio.shield(self.ready), timeout=self.timeout + 10)
            except Exception:
                # 失败连接先完全退出，下一次同键重试才能启动新进程。
                if self.task:
                    self.queue.put_nowait(None)
                    await self.task
                raise

    async def call(
        self,
        name: str,
        arguments: dict,
        *,
        context: ToolContext | None = None,
        call_id="knowledge",
    ) -> dict:
        started = time.perf_counter()
        result = None
        future = None
        transport_error = None
        try:
            await self.connect()
            meta: Any = {"aftersales/call_id": call_id}
            if context is not None:
                meta.update(
                    {
                        "aftersales/context": context.model_dump(mode="json"),
                        "aftersales/capability": self.capability,
                    }
                )
            future = asyncio.get_running_loop().create_future()
            self.queue.put_nowait((name, arguments, meta, future))
            response = await asyncio.wait_for(asyncio.shield(future), timeout=self.timeout + 1)
            result = response.structured_content
            if not isinstance(result, dict):
                raise ValueError("MCP response must have structured content")
            return result
        except asyncio.CancelledError:
            transport_error = "cancelled_result_unknown"
            await self._settle(future)
            raise
        except TimeoutError:
            transport_error = "timeout_result_unknown"
            await self._settle(future)
            raise
        except Exception:
            transport_error = "infrastructure_result_unknown"
            await self._settle(future)
            raise ConnectionError("MCP call failed; keep the original operation key") from None
        finally:
            if self.trace_path:
                self.trace_path.parent.mkdir(parents=True, exist_ok=True)
                row = {
                    "transport": "mcp",
                    "tool": name,
                    "call_id": call_id,
                    "context": context.model_dump(mode="json") if context else None,
                    "arguments": arguments,
                    "result": result,
                    "error": transport_error,
                    "elapsed_seconds": time.perf_counter() - started,
                    "protocol_version": self.protocol_version,
                }
                # capability 和进程环境绝不进入轨迹。
                with self.trace_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    async def _settle(self, future):
        if future is None:
            return
        try:
            # 取消调用方不能绕过仍在执行的业务事务；SDK超时后退出子进程再释放锁。
            await asyncio.shield(future)
        except Exception:
            if self.task:
                await self.task

    async def aclose(self):
        self.closed = True
        if self.task and not self.task.done():
            self.queue.put_nowait(None)
            await self.task
