"""Download the pinned Retail source files. Explicit network step; never used by tests."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/sierra-research/tau2-bench"
TAG = "v1.0.1"
COMMIT = "fc0055dc4e0a316c3f83133267fbd6faaa770992"
FILES = (
    "README.md",
    "LICENSE",
    "pyproject.toml",
    "docs/evaluation.md",
    "data/tau2/domains/retail/db.json",
    "data/tau2/domains/retail/policy.md",
    "data/tau2/domains/retail/tasks.json",
    "data/tau2/domains/retail/split_tasks.json",
    "src/tau2/domains/retail/data_model.py",
    "src/tau2/domains/retail/tools.py",
    "src/tau2/domains/retail/utils.py",
    "src/tau2/data_model/tasks.py",
    "src/tau2/environment/db.py",
    "src/tau2/environment/tool.py",
)


def fetch(relative: str) -> dict[str, object]:
    # 标签只用于说明；实际 URL 使用完整 commit，避免标签/分支漂移。
    url = f"https://raw.githubusercontent.com/sierra-research/tau2-bench/{COMMIT}/{relative}"
    request = urllib.request.Request(url, headers={"User-Agent": "AfterSalesAgent-M0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        content = response.read()
    target = ROOT / "data/upstream/tau2-bench" / relative
    if target.exists() and target.read_bytes() != content:
        raise RuntimeError(f"Refusing to overwrite changed source: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return {
        "upstream_path": relative,
        "local_path": target.relative_to(ROOT).as_posix(),
        "url": url,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size_bytes": len(content),
    }


def main() -> None:
    with ThreadPoolExecutor(max_workers=6) as pool:
        records = list(pool.map(fetch, FILES))
    manifest_path = ROOT / "data/manifests/source_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["commit"] == COMMIT and old["files"] == records:
            print(f"Verified existing pinned source: {len(records)} files")
            return
        raise RuntimeError("Source manifest already exists with a different snapshot.")
    manifest = {
        "manifest_version": "1.0",
        "dataset_kind": "public_simulated_retail",
        "repository": REPOSITORY,
        "tag": TAG,
        "commit": COMMIT,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "license_file": "data/upstream/tau2-bench/LICENSE",
        "files": records,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Downloaded {len(records)} files; pinned commit {COMMIT}")


if __name__ == "__main__":
    main()
