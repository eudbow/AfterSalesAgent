"""网页和评测共用模型/提示/工具定义；不创建可执行业务工具的知识通路。"""

import hashlib
import json
import sqlite3
from pathlib import Path

from aftersalesagent.business.database import RetailStore
from aftersalesagent.business.service import RetailService
from aftersalesagent.knowledge.base import PolicyEvidenceProvider
from aftersalesagent.knowledge.corpus import load_corpus
from aftersalesagent.knowledge.e4 import E4Config, build_index
from aftersalesagent.knowledge.retrieval import SQLiteKnowledgeIndex
from aftersalesagent.knowledge.service import PROTOCOL, KnowledgeService, OfflineKnowledgeBackend
from aftersalesagent.local_config import options_from_settings
from aftersalesagent.mcp_transport.adapters import MCPKnowledgeProvider
from aftersalesagent.models.deepseek import DeepSeekBackend
from aftersalesagent.policies.application import load_application_policy


def make_service(
    root: Path,
    settings,
    secret,
    *,
    template: Path,
    database: str | Path = ":memory:",
    tokenizer="char",
    top_k=5,
    enable_paid_api=False,
    backend_override=None,
    retrieval_config=None,
):
    config_path = retrieval_config or settings.knowledge_config
    e4_config = None
    if config_path:
        config_path = (root / config_path).resolve()
        if not config_path.is_relative_to(root.resolve()):
            raise ValueError("Knowledge configuration escapes project")
        e4_config = E4Config.model_validate_json(config_path.read_text("utf-8"))
        top_k = e4_config.top_k
    chunks, manifest = load_corpus(root, e4_config.corpus_version if e4_config else "e1_v1")
    store = RetailStore(template)
    try:
        tools = RetailService(store).specs()
    finally:
        store.close()
    core = load_application_policy(root)
    options = options_from_settings(settings) if settings.backend == "deepseek" else None
    if backend_override is not None:
        backend = backend_override
        label = "offline_test"
    elif options is not None:
        if not enable_paid_api or secret is None or not secret.get_secret_value():
            raise ValueError("Knowledge live mode requires explicit paid opt-in and local key")
        backend = DeepSeekBackend(options, secret, allow_network=True)
        label = "deepseek"
    else:
        backend = OfflineKnowledgeBackend()
        label = "mock_demo"
    if str(database) != ":memory:":
        Path(database).parent.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(
        json.dumps(
            {
                "corpus": manifest["files"],
                "options": options.model_dump(mode="json") if options else None,
                "tools": [t.model_dump(mode="json") for t in tools],
                "core": core,
                "protocol": PROTOCOL,
                "backend": label,
                "transport": settings.knowledge_transport,
                "tokenizer": tokenizer,
                "top_k": top_k,
                **({"retrieval_config": e4_config.model_dump()} if e4_config else {}),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    index: PolicyEvidenceProvider = (
        build_index(root, e4_config)
        if e4_config
        else SQLiteKnowledgeIndex(chunks, tokenizer=tokenizer, path=str(database))
    )
    ledger = sqlite3.connect(str(database), check_same_thread=False) if e4_config else None
    if settings.knowledge_transport == "mcp":
        if not e4_config and tokenizer != "char":
            index.close()
            raise ValueError("E2 MCP uses the frozen char tokenizer only")
        index = MCPKnowledgeProvider(index, knowledge_config=config_path if e4_config else None)
    return KnowledgeService(
        index,
        backend,
        core_policy=core,
        tools=tools,
        model_name=settings.model_name,
        temperature=settings.model_temperature,
        max_output_tokens=settings.model_max_output_tokens,
        top_k=top_k,
        backend_label=label,
        signature=signature,
        timeout_seconds=settings.model_timeout_seconds,
        ledger=ledger,
        close_ledger=e4_config is not None,
    )
