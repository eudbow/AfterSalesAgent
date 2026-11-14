"""E4 local browser acceptance with Mock answers; no real API calls or model downloads."""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main(args):
    output = ROOT / ".local/e4_artifacts/ui_mock"
    output.mkdir(parents=True, exist_ok=True)
    errors, checks = [], []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            channel="msedge", headless=True, args=["--disable-gpu"]
        )
        page = browser.new_page(viewport={"width": 1360, "height": 1000})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(args.url)
        status = page.request.get(args.url + "/knowledge/status").json()
        assert status["stage"] == "E4" and status["backend"] == "mock_demo"
        expect(page.locator("#knowledge-question")).to_have_value(
            "拆掉 SKYTTA 滑门地轨胶带后地板留印，属于品质保证吗？"
        )
        page.locator("#knowledge-submit").click()
        expect(page.locator("#knowledge-answer")).to_contain_text("离线演示资料卡片")
        expect(page.locator("#knowledge-sources")).to_contain_text("E4-sliding-")
        expect(page.locator("#knowledge-sources")).to_contain_text("生效日期：未知")
        checks.append("e4_corpus_and_traceable_official_sources")
        page.locator("#knowledge-scope").select_option("demo_store")
        page.locator("#knowledge-submit").click()
        expect(page.locator("#knowledge-sources")).to_contain_text("E4-SIM-D2")
        expect(page.locator("#knowledge-sources")).not_to_contain_text("E4-SIM-D1")
        expect(page.locator("#knowledge-sources")).to_contain_text("项目模拟条款")
        checks.append("simulation_date_boundary_and_scope")
        page.locator("#knowledge-scope").select_option("ikea_cn")
        page.locator("#knowledge-date").fill("2010-01-01")
        page.locator("#knowledge-submit").click()
        expect(page.locator("#knowledge-answer")).to_contain_text("没有可用证据")
        expect(page.locator("#knowledge-sources")).to_be_empty()
        checks.append("unknown_historical_applicability_refused")
        page.locator("#knowledge-date").fill("")
        page.locator("#knowledge-submit").click()
        expect(page.locator("#knowledge-sources")).to_contain_text("E4-sliding-")
        page.screenshot(path=str(output / "desktop.png"), full_page=True)
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        checks.append("responsive_layout")
        page.set_viewport_size({"width": 1360, "height": 1000})
        page.locator("#create").click()
        expect(page.locator("#send")).to_be_enabled()
        page.locator("#send").click()
        expect(page.locator("#pending")).to_be_visible()
        page.locator("#decline").click()
        expect(page.locator("#pending")).to_be_hidden()
        checks.append("old_business_confirmation_chain_preserved")
        assert not errors, errors
        browser.close()
    report = {
        "passed": True,
        "mode": "local_headless_mock",
        "checks": checks,
        "page_errors": errors,
        "real_generation_calls": 0,
        "screenshots": [str(output / "desktop.png"), str(output / "mobile.png")],
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8784")
    main(parser.parse_args())
