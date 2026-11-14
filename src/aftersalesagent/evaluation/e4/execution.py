"""Budgeted real-model calls and immutable execution fingerprints, without secret serialization."""

import hashlib
import json
from pathlib import Path

from aftersalesagent.evaluation.e4.budget import GenerationBudget
from aftersalesagent.models.errors import ModelBackendFault


def sha(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprints(root: Path):
    paths = [
        *root.joinpath("src").rglob("*.py"),
        *root.joinpath("tests").rglob("*.py"),
        *root.joinpath("scripts").glob("*e4*.py"),
        root / "pyproject.toml",
        root / "uv.lock",
        root / "configs/e4/requirements-models.txt",
        root / ".local/e4_artifacts/model-requirements.lock.txt",
        root / "data/upstream/tau2-bench/data/tau2/domains/retail/policy.md",
    ]
    return {path.relative_to(root).as_posix(): sha(path) for path in sorted(set(paths))}


def verify_fingerprints(root: Path, files: dict):
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError("Frozen E4 execution file mismatch: " + relative)


class BudgetedBackend:
    """Reserve only when generate actually starts, settle usage even on provider failure."""

    def __init__(self, backend, budget: GenerationBudget, run_id: str, identity_path: Path):
        self.backend, self.budget, self.run_id = backend, budget, run_id
        self.identity_path = identity_path
        self.call_key: str | None = None
        self.identity = (
            json.loads(identity_path.read_text("utf-8")) if identity_path.exists() else None
        )

    async def generate(self, request):
        if not self.call_key:
            raise ValueError("Missing explicit E4 paid-call key")
        self.budget.reserve(self.call_key, self.run_id)
        response = None
        error = None
        try:
            response = await self.backend.generate(request)
            identity = {
                k: response.provider_metadata.get(k) for k in ("model", "system_fingerprint")
            }
            if not identity["model"]:
                error = "missing_model_identity"
            elif self.identity is None:
                self.identity = identity
                self.identity_path.write_text(json.dumps(identity, indent=2) + "\n", "utf-8")
            elif identity != self.identity:
                error = "model_identity_drift"
            if error:
                raise ModelBackendFault(error, observation=response)
            return response
        except ModelBackendFault as exc:
            response = exc.observation
            error = exc.code
            raise
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            self.budget.finish(
                self.call_key,
                input_tokens=response.input_tokens if response else None,
                output_tokens=response.output_tokens if response else None,
                error=error,
                model=response.provider_metadata.get("model") if response else None,
            )

    async def aclose(self):
        await self.backend.aclose()
