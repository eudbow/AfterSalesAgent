"""E3 第一步的离线工程验收：场景 Mock 固定脚本，不用于比较模型能力。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from aftersalesagent.api.app import create_app
from aftersalesagent.config import Settings
from aftersalesagent.models.scenarios import SCENARIOS
from aftersalesagent.policies.application import load_application_policy
from aftersalesagent.skills.base import text_hash

ROOT = Path(__file__).resolve().parents[1]


def save(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".local/e3_step1_runs/offline_v1")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / ".local").resolve()):
        raise ValueError("Acceptance output must stay under the project's .local")
    if output.exists():
        raise ValueError("Refusing to overwrite an acceptance run; choose a fresh output directory")
    rows, comparable = [], {}
    for transport in ("native", "mcp"):
        for with_skill in (False, True):
            group = output / (transport + ("_s1" if with_skill else "_s0"))
            settings = Settings(
                tool_transport=transport,
                skill_id="after-sales-workflow" if with_skill else None,
                skill_version="1.0.0" if with_skill else "none",
            )
            app = create_app(data_dir=group / "app", settings=settings)
            with TestClient(app) as client:
                for scenario in ("query", "cancel", "address", "return"):
                    folder = group / scenario
                    created = client.post("/sessions", json={"scenario": scenario})
                    assert created.status_code == 201
                    session = created.json()
                    sid = session["session"]["session_id"]
                    headers = {"Authorization": "Bearer " + session["access_token"]}
                    # 不落盘 access_token；持久轨迹只含已鉴权业务上下文。
                    response = client.post(
                        f"/sessions/{sid}/messages",
                        headers=headers,
                        json={"event_id": "message-1", "content": SCENARIOS[scenario]["prompt"]},
                    )
                    assert response.status_code == 200, response.text
                    first = response.json()
                    before = client.get(f"/sessions/{sid}/orders", headers=headers).json()
                    first_snapshot = before
                    if scenario == "query":
                        assert first["status"] == "completed"
                        final = first
                    else:
                        assert first["status"] == "waiting_confirmation"
                        action = first["session"]["pending_action"]
                        original = next(
                            x for x in before["orders"] if x["order_id"] == action["order_id"]
                        )
                        assert original["status"] == (
                            "delivered" if scenario == "return" else "pending"
                        )
                        url = f"/sessions/{sid}/actions/{action['action_id']}/confirm"
                        confirm_body = {"event_id": "confirm-1", "approved": True}
                        committed = client.post(url, headers=headers, json=confirm_body)
                        assert committed.status_code == 200
                        final = committed.json()
                        assert final["status"] == "completed"
                        assert client.post(url, headers=headers, json=confirm_body).json() == final
                    trace = client.get(f"/sessions/{sid}/trace", headers=headers).json()["events"]
                    requests = [e["payload"] for e in trace if e["kind"] == "model_request"]
                    assert requests
                    for request in requests:
                        assert request["messages"][0]["content"] == load_application_policy(ROOT)
                        assert request["skill_version"] == ("1.0.0" if with_skill else "none")
                        if with_skill:
                            assert request["skill_sha256"] == text_hash(
                                (ROOT / "skills/after-sales-workflow/1.0.0/SKILL.md")
                                .read_bytes()
                                .decode("utf-8")
                            )
                            assert request["messages"][1]["role"] == "system"
                            assert request["messages"][1]["content"].startswith(
                                "Supplemental workflow guidance"
                            )
                    changes = sum(e["kind"] == "state_change" for e in trace)
                    assert changes == (0 if scenario == "query" else 1)
                    signatures = [
                        {"name": e["payload"]["name"], "arguments": e["payload"]["arguments"]}
                        for e in trace
                        if e["kind"] == "tool_call"
                    ]
                    key = (transport, scenario)
                    fingerprint = {
                        "core": text_hash(requests[0]["messages"][0]["content"]),
                        "tools": requests[0]["tools"],
                        "calls": signatures,
                    }
                    if key in comparable:
                        assert comparable[key] == fingerprint
                    comparable[key] = fingerprint
                    orders = client.get(f"/sessions/{sid}/orders", headers=headers).json()
                    save(folder / "first_result.json", first)
                    save(folder / "final_result.json", final)
                    save(folder / "before_confirmation_orders.json", first_snapshot)
                    save(folder / "final_orders.json", orders)
                    save(folder / "trace.json", trace)
                    rows.append(
                        {
                            "transport": transport,
                            "skill_version": "1.0.0" if with_skill else "none",
                            "scenario": scenario,
                            "status": final["status"],
                            "passed": True,
                            "state_change_events": changes,
                            "model_calls": len(requests),
                            "trace": str((folder / "trace.json").relative_to(ROOT)),
                        }
                    )
    summary = {
        "stage": "E3-step1",
        "backend": "ScenarioMockBackend",
        "meaning": "工程机制验收，不是自然语言能力或 Skill 收益实验",
        "episodes": len(rows),
        "passed": sum(r["passed"] for r in rows),
        "actual_paid_model_calls": 0,
        "local_gpu_experiments": 0,
        "s0_s1_core_tools_and_scripted_calls_identical": True,
        "rows": rows,
    }
    save(output / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
