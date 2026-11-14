"""M2：真实ASGI生命周期、SQLite Checkpoint、权限隔离和重复请求。"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from aftersalesagent.api.app import create_app


def application(tmp_path: Path, retail_template: Path, **kwargs):
    return create_app(data_dir=tmp_path / "app", template=retail_template, **kwargs)


def create(client: TestClient, scenario="cancel", **extra):
    response = client.post("/sessions", json={"scenario": scenario, **extra})
    assert response.status_code == 201, response.text
    data = response.json()
    return data, {"Authorization": "Bearer " + data["access_token"]}


def send(client: TestClient, data: dict, headers: dict, event_id="message-1"):
    sid = data["session"]["session_id"]
    response = client.post(
        f"/sessions/{sid}/messages",
        headers=headers,
        json={
            "event_id": event_id,
            "content": data["example_message"],
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def order(client, sid, headers, oid="#W5056519"):
    data = client.get(f"/sessions/{sid}/orders", headers=headers).json()
    return next(o for o in data["orders"] if o["order_id"] == oid)


@pytest.mark.parametrize("scenario", ["query", "cancel", "address", "return"])
def test_four_http_scenarios(tmp_path: Path, retail_template: Path, scenario: str):
    with TestClient(application(tmp_path, retail_template)) as client:
        assert client.get("/health").json()["stage"] == "M3"
        assert "AfterSalesAgent" in client.get("/").text
        data, headers = create(client, scenario)
        sid = data["session"]["session_id"]
        assert client.get(f"/sessions/{sid}/orders", headers=headers).json() == {"orders": []}
        result = send(client, data, headers)
        if scenario == "query":
            assert result["status"] == "completed"
            assert result["session"]["pending_action"] is None
            return
        assert result["status"] == "waiting_confirmation"
        action = result["session"]["pending_action"]
        oid = action["order_id"]
        before = order(client, sid, headers, oid)
        result = client.post(
            f"/sessions/{sid}/actions/{action['action_id']}/confirm",
            headers=headers,
            json={"event_id": "confirm-1", "approved": True},
        )
        assert result.status_code == 200 and result.json()["status"] == "completed"
        after = order(client, sid, headers, oid)
        if scenario == "cancel":
            assert before["status"] == "pending" and after["status"] == "cancelled"
            assert len(after["payment_history"]) == len(before["payment_history"]) + 1
        elif scenario == "address":
            assert after["address"]["zip"] == "98101"
            assert after["status"] == before["status"]
        else:
            assert after["status"] == "return requested"
            assert after["payment_history"] == before["payment_history"]
        assert after["version"] == before["version"] + 1


def test_restart_restores_pending_history_budget_and_idempotent_result(
    tmp_path: Path,
    retail_template: Path,
):
    data_dir = tmp_path / "app"
    with TestClient(create_app(data_dir=data_dir, template=retail_template)) as client:
        data, headers = create(client)
        sid = data["session"]["session_id"]
        first = send(client, data, headers)
        usage = client.get(f"/sessions/{sid}", headers=headers).json()["usage"]
        assert usage["model_calls"] == 3
        action = first["session"]["pending_action"]
    # 新app、新Hub、新Runtime、新SQLite连接，不能借用旧Python对象。
    new_app = create_app(data_dir=data_dir, template=retail_template)
    with TestClient(new_app) as reopened:
        state = reopened.get(f"/sessions/{sid}", headers=headers).json()
        assert state["session"]["pending_action"] == action
        assert state["usage"] == usage
        replay = send(reopened, data, headers)
        assert replay == first
        url = f"/sessions/{sid}/actions/{action['action_id']}/confirm"
        result = reopened.post(
            url, headers=headers, json={"event_id": "confirm-1", "approved": True}
        )
        assert result.json()["status"] == "completed"
        assert (
            reopened.post(
                url, headers=headers, json={"event_id": "confirm-1", "approved": True}
            ).json()
            == result.json()
        )
        again = reopened.post(
            url, headers=headers, json={"event_id": "confirm-2", "approved": True}
        ).json()
        assert again["status"] == "completed"
        assert len(order(reopened, sid, headers)["payment_history"]) == 2
        trace = reopened.get(f"/sessions/{sid}/trace", headers=headers).json()["events"]
        assert sum(e["kind"] == "state_change" for e in trace) == 1
        assert [e["sequence"] for e in trace] == list(range(len(trace)))
        assert reopened.portal is not None
        snapshot = reopened.portal.call(
            new_app.state.hub._runtimes[sid].graph.aget_state,
            {"configurable": {"thread_id": sid}},
        )
        assert any(m["role"] == "user" for m in snapshot.values["frame"]["messages"])
        assert (
            snapshot.values["frame"]["context"]["environment_id"]
            == data["session"]["environment_id"]
        )


def test_event_conflict_and_session_tokens(tmp_path: Path, retail_template: Path):
    with TestClient(application(tmp_path, retail_template)) as client:
        first, headers = create(client)
        second, other = create(client)
        sid = first["session"]["session_id"]
        for suffix in ["", "/orders", "/trace"]:
            assert client.get(f"/sessions/{sid}{suffix}").status_code == 404
            assert client.get(f"/sessions/{sid}{suffix}", headers=other).status_code == 404
        result = send(client, first, headers)
        conflict = client.post(
            f"/sessions/{sid}/messages",
            headers=headers,
            json={
                "event_id": "message-1",
                "content": "different",
            },
        )
        assert conflict.status_code == 409
        state = client.get(f"/sessions/{sid}", headers=headers).json()
        assert state["session"]["pending_action"] == result["session"]["pending_action"]
        foreign = client.post(
            f"/sessions/{second['session']['session_id']}/actions/"
            f"{result['session']['pending_action']['action_id']}/confirm",
            headers=other,
            json={"event_id": "foreign", "approved": True},
        )
        assert foreign.json()["status"] == "rejected"
        assert (
            client.post("/sessions", json={"database_path": "C:/arbitrary.sqlite"}).status_code
            == 422
        )
        assert (
            client.post(
                f"/sessions/{sid}/messages",
                headers=headers,
                json={
                    "event_id": "bad",
                    "content": "hello",
                    "authenticated_user_id": "someone",
                },
            ).status_code
            == 422
        )


def test_environments_are_independent_and_decline_survives_replay(
    tmp_path: Path,
    retail_template: Path,
):
    with TestClient(application(tmp_path, retail_template)) as client:
        one, h1 = create(client)
        two, h2 = create(client)
        r1, r2 = send(client, one, h1), send(client, two, h2)
        sid1, sid2 = one["session"]["session_id"], two["session"]["session_id"]
        for sid, headers, result, approved in [
            (sid1, h1, r1, True),
            (sid2, h2, r2, False),
        ]:
            aid = result["session"]["pending_action"]["action_id"]
            url = f"/sessions/{sid}/actions/{aid}/confirm"
            first = client.post(
                url, headers=headers, json={"event_id": "decision", "approved": approved}
            )
            assert first.status_code == 200
            assert (
                client.post(
                    url,
                    headers=headers,
                    json={
                        "event_id": "decision-again",
                        "approved": approved,
                    },
                ).json()["status"]
                == first.json()["status"]
            )
            opposite = client.post(
                url,
                headers=headers,
                json={
                    "event_id": "opposite",
                    "approved": not approved,
                },
            )
            assert opposite.json()["status"] == "rejected"
        assert order(client, sid1, h1)["status"] == "cancelled"
        assert order(client, sid2, h2)["status"] == "pending"


def test_trace_projection_rebuild_does_not_execute_again(tmp_path: Path, retail_template: Path):
    app = application(tmp_path, retail_template)
    with TestClient(app) as client:
        data, headers = create(client)
        sid = data["session"]["session_id"]
        result = send(client, data, headers)
        runtime = app.state.hub._runtimes[sid]
        before = runtime.journal.usage(sid)
        path = tmp_path / "app" / "traces" / (sid + ".jsonl")
        path.write_text('{"broken":', encoding="utf-8")
        trace = client.get(f"/sessions/{sid}/trace?output=jsonl", headers=headers)
        assert trace.status_code == 200
        events = [json.loads(line) for line in trace.text.splitlines()]
        assert events == [json.loads(line) for line in path.read_text("utf-8").splitlines()]
        assert runtime.journal.usage(sid) == before
        assert (
            runtime.service.view(sid).pending_action.action_id
            == result["session"]["pending_action"]["action_id"]
        )
        assert data["access_token"] not in trace.text


def test_commit_then_timeout_recovers_with_same_operation_key(
    tmp_path: Path,
    retail_template: Path,
    monkeypatch,
):
    app = application(tmp_path, retail_template)
    with TestClient(app) as client:
        data, headers = create(client)
        sid = data["session"]["session_id"]
        pending = send(client, data, headers)["session"]["pending_action"]
        service = app.state.hub._runtimes[sid].service
        original = service.confirm
        attempts = []

        def lose_first_response(*args):
            result = original(*args)
            attempts.append(args[1])
            if len(attempts) == 1:
                raise TimeoutError("Injected response loss after commit")
            return result

        monkeypatch.setattr(service, "confirm", lose_first_response)
        response = client.post(
            f"/sessions/{sid}/actions/{pending['action_id']}/confirm",
            headers=headers,
            json={"event_id": "timeout-confirm", "approved": True},
        )
        assert response.json()["status"] == "completed"
        assert attempts == [pending["action_id"]] * 2
        assert len(order(client, sid, headers)["payment_history"]) == 2


def test_unresolved_commit_timeout_can_recover_after_app_restart(
    tmp_path: Path,
    retail_template: Path,
    monkeypatch,
):
    app = application(tmp_path, retail_template)
    with TestClient(app) as client:
        data, headers = create(client)
        sid = data["session"]["session_id"]
        action = send(client, data, headers)["session"]["pending_action"]["action_id"]
        service = app.state.hub._runtimes[sid].service
        original = service.confirm

        def always_lose_response(*args):
            original(*args)
            raise TimeoutError("Injected response loss")

        monkeypatch.setattr(service, "confirm", always_lose_response)
        url = f"/sessions/{sid}/actions/{action}/confirm"
        result = client.post(
            url, headers=headers, json={"event_id": "lost", "approved": True}
        ).json()
        assert result["status"] == "failed" and result["termination_reason"] == "timeout"
        assert order(client, sid, headers)["status"] == "cancelled"
        blocked = client.post(
            f"/sessions/{sid}/messages",
            headers=headers,
            json={
                "event_id": "new",
                "content": "new request",
            },
        )
        assert blocked.status_code == 409
    with TestClient(application(tmp_path, retail_template)) as reopened:
        result = reopened.post(
            url, headers=headers, json={"event_id": "lost", "approved": True}
        ).json()
        assert result["status"] == "completed"
        assert len(order(reopened, sid, headers)["payment_history"]) == 2


def test_slow_transaction_does_not_block_health(
    tmp_path: Path,
    retail_template: Path,
    monkeypatch,
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    app = application(tmp_path, retail_template)
    with TestClient(app) as client:
        data, headers = create(client)
        sid = data["session"]["session_id"]
        action = send(client, data, headers)["session"]["pending_action"]["action_id"]
        service = app.state.hub._runtimes[sid].service
        original = service.confirm
        entered, release = Event(), Event()

        def blocked(*args):
            entered.set()
            assert release.wait(timeout=5)
            return original(*args)

        monkeypatch.setattr(service, "confirm", blocked)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future = pool.submit(
                client.post,
                f"/sessions/{sid}/actions/{action}/confirm",
                headers=headers,
                json={"event_id": "slow", "approved": True},
            )
            try:
                assert entered.wait(timeout=3)
                health = pool.submit(client.get, "/health").result(timeout=2)
                assert health.status_code == 200
            finally:
                release.set()
            assert future.result(timeout=3).json()["status"] == "completed"


def test_http_infrastructure_error_is_structured_and_does_not_leak_details(
    tmp_path: Path,
    retail_template: Path,
    monkeypatch,
):
    app = application(tmp_path, retail_template)
    with TestClient(app, raise_server_exceptions=False) as client:
        data, headers = create(client)

        async def fail(_sid):
            raise OSError("private database path and credential must not leak")

        monkeypatch.setattr(app.state.hub, "get", fail)
        response = client.get("/sessions/" + data["session"]["session_id"], headers=headers)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "infrastructure_error"
        assert "private database" not in response.text
