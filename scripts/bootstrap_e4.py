"""Explicit E4 preparation; never runs as part of tests or downloads extra challengers."""

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / ".local/e4_artifacts"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", "utf-8")


def baseline():
    dest = ARTIFACTS / "pre_e4"
    manifest = dest / "manifest.json"
    if manifest.exists():
        print("Pre-E4 baseline already exists; not overwritten.")
        return
    rows = {}
    for folder in ("src", "tests", "configs", "docs", "data", "policies", "skills"):
        base = ROOT / folder
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            rel = path.relative_to(ROOT).as_posix()
            rows[rel] = sha(path)
            if folder in {"src", "configs", "docs"}:
                copy = dest / "files" / rel
                copy.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, copy)
    for name in ("pyproject.toml", "uv.lock"):
        rows[name] = sha(ROOT / name)
        shutil.copy2(ROOT / name, dest / "files" / name)
    write_json(manifest, {"created_at": datetime.now(UTC).isoformat(), "files": rows})
    print(json.dumps({"baseline_files": len(rows)}))


def download_models():
    models = ("BAAI/bge-small-zh-v1.5", "BAAI/bge-reranker-v2-m3")
    with httpx.Client(timeout=90, follow_redirects=True, trust_env=False) as client:
        for model in models:
            lock = ARTIFACTS / (model.split("/")[1] + ".json")
            if lock.exists():
                metadata = json.loads(lock.read_text("utf-8"))
            else:
                response = client.get(f"https://huggingface.co/api/models/{model}?blobs=true")
                response.raise_for_status()
                body = response.json()
                siblings = body["siblings"]
                has_safe = any(s["rfilename"] == "model.safetensors" for s in siblings)
                wanted = []
                for item in siblings:
                    name = item["rfilename"]
                    if name.startswith(("onnx/", "openvino/")):
                        continue
                    if name.endswith((".json", ".txt", ".model")) or name in {
                        "README.md",
                        "LICENSE",
                        "model.safetensors" if has_safe else "pytorch_model.bin",
                    }:
                        wanted.append(
                            {"name": name, "upstream_sha256": item.get("lfs", {}).get("sha256")}
                        )
                metadata = {
                    "model_id": model,
                    "license": body.get("cardData", {}).get("license"),
                    "revision": body["sha"],
                    "files": wanted,
                    "created_at": datetime.now(UTC).isoformat(),
                }
                write_json(lock, metadata)
            directory = ROOT / ".local/models" / model.split("/")[1] / metadata["revision"]
            directory.mkdir(parents=True, exist_ok=True)
            for item in metadata["files"]:
                path = directory / item["name"]
                if path.exists() and item.get("sha256") == sha(path):
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                url = (
                    f"https://huggingface.co/{model}/resolve/{metadata['revision']}/{item['name']}"
                )
                partial = path.with_name(path.name + ".partial")
                print(f"Downloading {model}/{item['name']}", flush=True)
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with partial.open("wb") as stream:
                        for block in response.iter_bytes(1024 * 1024):
                            stream.write(block)
                actual = sha(partial)
                if item.get("upstream_sha256") and item["upstream_sha256"] != actual:
                    raise ValueError("Model download checksum mismatch")
                partial.replace(path)
                item["sha256"], item["bytes"] = actual, path.stat().st_size
                metadata["local_path"] = directory.relative_to(ROOT).as_posix()
                write_json(lock, metadata)
            metadata["complete"] = True
            write_json(lock, metadata)
            print(
                json.dumps({"model": model, "revision": metadata["revision"], "complete": True}),
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("baseline", "models"))
    args = parser.parse_args()
    baseline() if args.action == "baseline" else download_models()
