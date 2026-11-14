"""E2：MCP创建提议，宿主提交后崩溃，独立进程恢复确认；无模型API。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from aftersalesagent.business.database import RetailStore, clone_database, import_pinned_database
from aftersalesagent.business.service import RetailService
from aftersalesagent.harness.journal import Journal
from aftersalesagent.mcp_transport.adapters import MCPRetailProvider
from aftersalesagent.models.scenarios import SCENARIOS, ScenarioMockBackend
from aftersalesagent.runtime.langgraph_runtime import LangGraphRuntime
from aftersalesagent.schemas.contracts import ConfirmationEvent, MessageEvent, RunContext
from aftersalesagent.tools.registry import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]


async def phase(name: str, directory: Path) -> None:
    database = directory / "business.sqlite"
    if name == "prepare":
        template = ROOT / ".local/databases/retail_template.sqlite"
        if not template.exists():
            initial = import_pinned_database(ROOT, template)
            initial.close()
        store = clone_database(template, database)
    else:
        store = RetailStore(database)
    journal = Journal(directory / "application.sqlite")
    service = RetailService(store)
    async with AsyncSqliteSaver.from_conn_string(str(directory / "checkpoints.sqlite")) as saver:
        runtime = LangGraphRuntime(
            ScenarioMockBackend("cancel"),
            service,
            registry=ToolRegistry(
                service,
                provider=MCPRetailProvider(service, trace_path=directory / "traces/mcp.jsonl"),
            ),
            policy_text=(
                ROOT / "data/upstream/tau2-bench/data/tau2/domains/retail/policy.md"
            ).read_text("utf-8"),
            checkpointer=saver,
            journal=journal,
            trace_directory=directory / "traces",
        )
        state_path = directory / "state.json"
        if name == "prepare":
            session = await runtime.start(
                RunContext(environment_id=store.path.stem, tool_transport="mcp")
            )
            result = await runtime.advance(
                session.session_id,
                MessageEvent(event_id="message-1", content=SCENARIOS["cancel"]["prompt"]),
            )
            assert result.status == "waiting_confirmation"
            assert result.session.pending_action is not None
            state_path.write_text(
                json.dumps(
                    {
                        "session_id": session.session_id,
                        "action_id": result.session.pending_action.action_id,
                    }
                ),
                encoding="utf-8",
            )
        else:
            state = json.loads(state_path.read_text("utf-8"))
            sid, aid = state["session_id"], state["action_id"]
            if name == "crash":
                original = service.confirm

                def commit_then_exit(*args):
                    result = original(*args)
                    assert result.status == "ok"
                    # 故障注入仅发生在本脚本新建的子进程中；SQLite事务已提交。
                    os._exit(73)

                service.confirm = commit_then_exit
            event = ConfirmationEvent(event_id="confirm-1", action_id=aid, approved=True)
            before = journal.usage(sid)
            result = await runtime.advance(sid, event)
            assert name == "recover" and result.status == "completed"
            replay = await runtime.advance(sid, event)
            assert result == replay
            with store.transaction() as connection:
                order = store.order(connection, "#W5056519")
            assert order is not None
            assert order["status"] == "cancelled" and len(order["payment_history"]) == 2
            assert journal.usage(sid)["model_calls"] == before["model_calls"] == 3
            events = journal.events(sid)
            assert sum(e.kind == "state_change" for e in events) == 1
            output = {
                "mode": "offline_mcp_proposal_host_confirmation_crash_recovery",
                "session_id": sid,
                "action_id": aid,
                "order_id": "#W5056519",
                "status": order["status"],
                "payment_history_count": len(order["payment_history"]),
                "state_change_events": 1,
                "model_calls": before["model_calls"],
                "reserved_active_seconds_after_crash": before["active_seconds"],
                "same_event_replay_equal": True,
                "trace": str(directory / "traces" / (sid + ".jsonl")),
                "agent_task_success_rate": None,
            }
            (directory / "result.json").write_text(
                json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        await runtime.registry.aclose()
    journal.close()
    store.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["prepare", "crash", "recover"])
    parser.add_argument("--directory", type=Path)
    args = parser.parse_args()
    if args.phase:
        if args.directory is None:
            parser.error("--phase requires --directory")
        asyncio.run(phase(args.phase, args.directory.resolve()))
        return
    run_id = "e2_recovery_" + uuid.uuid4().hex[:10]
    directory = ROOT / ".local/recovery" / run_id
    directory.mkdir(parents=True, exist_ok=False)
    codes = {}
    for name, expected in [("prepare", 0), ("crash", 73), ("recover", 0)]:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--phase",
                name,
                "--directory",
                str(directory),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=45,
        )
        codes[name] = completed.returncode
        if completed.returncode != expected:
            raise RuntimeError(
                f"Phase {name}: expected {expected}, got {completed.returncode}\n"
                + completed.stdout
                + completed.stderr
            )
    output = json.loads((directory / "result.json").read_text("utf-8"))
    output.update(run_id=run_id, process_exit_codes=codes, directory=str(directory))
    target = ROOT / ".local/results" / (run_id + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print("Result:", target)


if __name__ == "__main__":
    main()
