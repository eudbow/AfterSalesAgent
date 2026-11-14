"""数据库工作放到线程；超时或取消不会让同会话绕过仍在执行的事务。"""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable

from sqlalchemy.exc import OperationalError

from aftersalesagent.schemas.contracts import ToolError, ToolResult


async def run_sync[T](work: Callable[[], T]) -> T:
    task = asyncio.create_task(asyncio.to_thread(work))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        # to_thread不能停止已运行的事务。先等事务退出，再释放会话锁。
        # 若提交后响应丢失，下次使用原幂等键查询结果。
        try:
            await task
        finally:
            raise


async def invoke_safely(
    work: Callable[[], Awaitable[ToolResult]],
    call_id: str,
    *,
    on_retry: Callable[[str], Awaitable[None]] | None = None,
) -> ToolResult:
    for attempt in range(2):
        try:
            return await work()
        except (TimeoutError, ConnectionError, sqlite3.OperationalError, OperationalError) as exc:
            if attempt == 0:
                # 调用方必须提供相同的持久化操作键。最多重试一次，不另建操作。
                if on_retry:
                    await on_retry(
                        "timeout" if isinstance(exc, TimeoutError) else "infrastructure_error"
                    )
                continue
            return ToolResult(
                call_id=call_id,
                status="error",
                error=ToolError(
                    code="timeout" if isinstance(exc, TimeoutError) else "infrastructure_error",
                    message="The operation result is unavailable; retry the same event.",
                    retryable=True,
                ),
            )
        except Exception:
            return ToolResult(
                call_id=call_id,
                status="error",
                error=ToolError(
                    code="infrastructure_error",
                    message="The operation could not complete; inspect the local trace.",
                    retryable=False,
                ),
            )
    raise AssertionError("Unreachable")
