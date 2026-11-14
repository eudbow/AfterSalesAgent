"""依赖冒烟：验证 StateGraph + 文件型 Checkpointer，而非售后 Agent。"""

from pathlib import Path
from typing import TypedDict

import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    counter: int


def increment(state: State) -> State:
    return {"counter": state["counter"] + 1}


def test_graph_state_survives_reopening_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    path = str(tmp_path / "checkpoint.sqlite")
    config: RunnableConfig = {"configurable": {"thread_id": "offline-smoke"}}
    builder = StateGraph(State)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    with SqliteSaver.from_conn_string(path) as saver:
        graph = builder.compile(checkpointer=saver)
        assert graph.invoke({"counter": 0}, config)["counter"] == 1
    with SqliteSaver.from_conn_string(path) as reopened:
        graph = builder.compile(checkpointer=reopened)
        assert graph.get_state(config).values["counter"] == 1
