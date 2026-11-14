"""E2 浏览器验收：仅本机 Mock，不读取密钥、不运行 GPU。"""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8767")
    args = parser.parse_args()
    output = ROOT / ".local/screenshots"
    output.mkdir(parents=True, exist_ok=True)
    errors, checks = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True, args=["--disable-gpu"])
        version = browser.version
        page = browser.new_page(viewport={"width": 1360, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        health = page.request.get(args.url + "/health").json()
        assert health["tool_transport"] == health["knowledge_transport"] == "mcp"
        checks.append("native_mcp_configuration_visible")
        expect(page.locator("#knowledge-mode-status")).to_contain_text("离线资料演示")
        page.locator("#knowledge-submit").click()
        expect(page.locator("#knowledge-answer")).to_contain_text("离线演示资料卡片")
        expect(page.locator("#knowledge-sources")).to_contain_text("R02")
        expect(page.locator("#knowledge-sources")).to_contain_text("生效日期：未知")
        expect(page.locator("#knowledge-usage")).to_contain_text("Token：未知")
        checks.append("official_sources_unknown_effective_date_and_usage")
        page.locator("#knowledge-mode").select_option("full")
        page.locator("#knowledge-submit").click()
        expect(page.locator("#knowledge-usage")).to_contain_text("资料片段：17")
        checks.append("full_context_17_curated_cards")

        page.locator("#knowledge-scope").select_option("demo_store")
        expect(page.locator("#knowledge-date")).to_have_value("2026-07-01")
        page.locator("#knowledge-mode").select_option("rag")
        page.locator("#knowledge-submit").click()
        expect(page.locator("#knowledge-sources")).to_contain_text("D02")
        expect(page.locator("#knowledge-sources")).to_contain_text("项目模拟条款")
        expect(page.locator("#knowledge-sources")).not_to_contain_text("D01")
        checks.append("simulated_scope_and_july_boundary")

        page.locator("#knowledge-scope").select_option("ikea_cn")
        page.locator("#knowledge-date").fill("2020-01-01")
        page.locator("#knowledge-submit").click()
        expect(page.locator("#knowledge-answer")).to_contain_text("没有可用证据")
        expect(page.locator("#knowledge-sources")).to_be_empty()
        checks.append("unknown_official_historical_applicability_refused")

        page.locator("#knowledge-date").fill("")
        page.locator("#knowledge-submit").click()
        expect(page.locator("#knowledge-sources")).to_contain_text("R02")
        page.screenshot(path=str(output / "e2_desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(output / "e2_mobile.png"), full_page=True)
        checks.append("responsive_layout_no_horizontal_overflow")

        page.set_viewport_size({"width": 1360, "height": 1000})
        page.locator("#create").click()
        expect(page.locator("#send")).to_be_enabled()
        page.locator("#send").click()
        expect(page.locator("#pending")).to_be_visible()
        page.locator("#decline").click()
        expect(page.locator("#pending")).to_be_hidden()
        checks.append("existing_business_confirmation_flow_preserved")
        assert not errors, errors
        browser.close()
    report = {
        "passed": True,
        "mode": "headless_browser_local_mock_over_mcp",
        "browser_version": version,
        "checks": checks,
        "page_errors": errors,
        "real_model_calls": 0,
        "screenshots": [str(output / "e2_desktop.png"), str(output / "e2_mobile.png")],
    }
    target = ROOT / ".local/results/e2_ui_acceptance.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
