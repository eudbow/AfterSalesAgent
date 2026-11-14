"""Isolated database copies; the pinned source and template are never modified by cases."""

from pathlib import Path

import pytest

from aftersalesagent.business.database import clone_database, import_pinned_database

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def retail_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("retail_template") / "template.sqlite"
    store = import_pinned_database(ROOT, path)
    store.close()
    return path


@pytest.fixture
def store(retail_template: Path, tmp_path: Path):
    path = tmp_path / "episode.sqlite"
    result = clone_database(retail_template, path)
    try:
        yield result
    finally:
        result.close()


@pytest.fixture(autouse=True)
def block_real_http_transport(monkeypatch: pytest.MonkeyPatch):
    """所有默认pytest封锁真实HTTPX传输；MockTransport和ASGI测试不受影响。"""
    import httpx

    async def no_async_network(*args, **kwargs):
        raise AssertionError("Real HTTP is forbidden in offline pytest")

    def no_sync_network(*args, **kwargs):
        raise AssertionError("Real HTTP is forbidden in offline pytest")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", no_async_network)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", no_sync_network)
