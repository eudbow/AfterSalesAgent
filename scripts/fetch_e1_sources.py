"""显式下载两份官方HTML快照；不覆盖既有版本，默认测试不联网。"""

import hashlib
import json
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "ikea_returns": "https://www.ikea.cn/cn/zh/customer-service/returns-claims/return-policy/",
    "ikea_guarantees": "https://www.ikea.cn/cn/zh/customer-service/returns-claims/guarantee/",
}


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "br", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"}:
            self.hidden -= 1
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "div"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)

    def text(self):
        return "\n".join(
            " ".join(line.split()) for line in "".join(self.parts).splitlines() if line.strip()
        )


def main():
    directory = ROOT / "data/knowledge/e1_v1/sources"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text("utf-8"))
        for doc in manifest["documents"]:
            for kind in ("html", "text"):
                assert (
                    hashlib.sha256((ROOT / doc[kind + "_path"]).read_bytes()).hexdigest()
                    == doc[kind + "_sha256"]
                )
        print("Existing snapshots verified; no network requests.")
        return
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    with httpx.Client(timeout=40, follow_redirects=True, trust_env=False) as client:
        for document_id, url in SOURCES.items():
            response = client.get(url, headers={"User-Agent": "AfterSalesAgent-E1-Research/1.0"})
            response.raise_for_status()
            raw = response.content
            parser = TextParser()
            parser.feed(raw.decode("utf-8"))
            text = parser.text().encode()
            assert len(text) > 1000
            paths = {}
            for kind, data, suffix in [("html", raw, "html"), ("text", text, "txt")]:
                p = directory / f"{document_id}.{suffix}"
                p.write_bytes(data)
                paths[kind + "_path"] = p.relative_to(ROOT).as_posix()
                paths[kind + "_sha256"] = hashlib.sha256(data).hexdigest()
            rows.append(
                {
                    "document_id": document_id,
                    "source_url": url,
                    "final_url": str(response.url),
                    "fetched_at": datetime.now(UTC).isoformat(),
                    "http_status": response.status_code,
                    "effective_from": None,
                    "effective_until": None,
                    "scope": "ikea_cn",
                    "origin": "official",
                    "version": "snapshot-2026-10-24-" + hashlib.sha256(raw).hexdigest()[:12],
                    "date_note": (
                        "No policy effective date ident"
                        "ified; fetch timestamp is not "
                        "an effective date"
                    ),
                    **paths,
                }
            )
    manifest_path.write_text(
        json.dumps({"snapshot_date": "2026-10-24", "documents": rows}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"documents": len(rows), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
