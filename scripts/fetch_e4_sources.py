"Fetch ten official mainland-China HTML sources into a new E4 snapshot, without E1 writes."

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "data/knowledge/e4_v1/sources"
BASE = "https://www.ikea.cn/cn/zh/customer-service/returns-claims/guarantee/"
SOURCES = {
    "mattress": "chuangdianpinzhibaozheng-pub47442310",
    "wardrobe": "pax-komplement-pub43414e20",
    "sliding": "-si-ji-ta-hua-men-pin-zhi-bao-zheng-pube3d3221e",
    "bathroom": "yushi-pub84374700",
    "seating": "keting-pub34d338d0",
    "kitchen": "chufang-pubf2540d3b",
    "knives": "ikea-pubf5ce25c6",
    "cookware": "ikea-pubc57cd6da",
    "office": "bangong-pub5a4ae7cf",
    "chairs": "ikea-pub1fd94ced",
}


class MainText(HTMLParser):
    "Keep visible blocks and headings; retain full extracted text for source offsets."

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0
        self.seen_h1 = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "head"}:
            self.hidden += 1
        if tag == "h1" and not self.hidden:
            self.seen_h1 = True
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "br", "div", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "head"}:
            self.hidden -= 1
        if tag in {"p", "li", "h1", "h2", "h3", "h4", "div", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden and self.seen_h1:
            self.parts.append(data)

    def text(self):
        text = "\n".join(" ".join(s.split()) for s in "".join(self.parts).splitlines() if s.strip())
        # The exact club footer heading marks the end of substantive policy content.
        return text.split("\n加入宜家俱乐部\n", 1)[0].strip() + "\n"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch(item):
    name, tail = item
    url = BASE + tail
    meta = DIRECTORY / (name + ".json")
    if meta.exists():
        record = json.loads(meta.read_text("utf-8"))
        for kind in ("html", "text"):
            if digest((ROOT / record[kind + "_path"]).read_bytes()) != record[kind + "_sha256"]:
                raise ValueError("Existing E4 source changed; use a new snapshot version")
        return record
    with httpx.Client(timeout=60, follow_redirects=True, trust_env=False) as client:
        response = client.get(url, headers={"User-Agent": "AfterSalesAgent-E4-Research/1.0"})
        response.raise_for_status()
    if response.url.host != "www.ikea.cn":
        raise ValueError("Unexpected source redirect")
    parser = MainText()
    parser.feed(response.text)
    text = parser.text()
    if len(text) < 500 or "品质保证" not in text:
        raise ValueError("Incomplete official policy body")
    record = {
        "source_id": "e4_" + name,
        "document_id": "e4_" + name,
        "title": text.splitlines()[0],
        "source_url": url,
        "final_url": str(response.url),
        "fetched_at": datetime.now(UTC).isoformat(),
        "http_status": response.status_code,
        "scope": "ikea_cn",
        "origin": "official",
        "category": name,
        "version": "e4-snapshot-" + digest(response.content)[:12],
        "effective_from": None,
        "effective_until": None,
        "date_evidence": None,
        "date_note": (
            "Publication/effective dates unverified; fetch time is not an effective date"
        ),
        "extractor_version": "e4-html-main-h1-v1",
        "status": "captured_pending_review",
        "cleaning": (
            "Exclude head/scripts/styles; visible blocks from first h1; remove exact club footer"
        ),
    }
    for kind, data, suffix in (("html", response.content, "html"), ("text", text.encode(), "txt")):
        path = DIRECTORY / (name + "." + suffix)
        path.write_bytes(data)
        record[kind + "_path"] = path.relative_to(ROOT).as_posix()
        record[kind + "_sha256"] = digest(data)
    meta.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(json.dumps({"source": name, "characters": len(text)}, ensure_ascii=False), flush=True)
    return record


if __name__ == "__main__":
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, SOURCES.items()))
    manifest = DIRECTORY / "manifest.json"
    if not manifest.exists():
        manifest.write_text(
            json.dumps({"documents": records}, ensure_ascii=False, indent=2) + "\n", "utf-8"
        )
    print(
        json.dumps(
            {
                "documents": len(records),
                "body_characters": sum(
                    len((ROOT / r["text_path"]).read_text("utf-8")) for r in records
                ),
            }
        )
    )
