"""可选的独立浏览器验收；使用新浏览器上下文，不读取用户日常浏览器资料。"""

import argparse
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--channel", default="msedge")
    parser.add_argument("--artifact-prefix", choices=["m2", "m3"], default="m2")
    args = parser.parse_args()
    prefix = args.artifact_prefix
    output = ROOT / ".local/screenshots"
    output.mkdir(parents=True, exist_ok=True)
    errors = []
    completed = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.channel, headless=True, args=["--disable-gpu"])
        version = browser.version
        context = browser.new_context(viewport={"width": 1360, "height": 1000})
        page = context.new_page()
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(args.url)
        expect(page.get_by_role("heading", name="每一次售后，都有据可查。")).to_be_visible()
        page.get_by_role("button", name="新建会话").click()
        expect(page.locator("#message")).to_be_enabled()
        expect(page.locator("#send")).to_be_enabled()
        page.get_by_role("button", name="发送消息").click()
        expect(page.locator("#pending")).to_be_visible()
        expect(page.locator("#approve")).to_be_enabled()
        page.screenshot(path=str(output / f"{prefix}_desktop_pending.png"), full_page=True)
        # 浏览器刷新后从本地会话凭据与服务端Checkpoint恢复。
        page.reload()
        expect(page.locator("#pending")).to_be_visible()
        expect(page.locator("#approve")).to_be_enabled()
        completed.append("pending_action_survives_browser_reload")
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
        page.screenshot(path=str(output / f"{prefix}_mobile_pending.png"), full_page=True)
        page.set_viewport_size({"width": 1360, "height": 1000})

        lost = []

        def lose_response(route):
            if not lost:
                # 让服务端真正提交，但不把响应交给页面，模拟客户端超时。
                response = route.fetch()
                assert response.status == 200
                lost.append(True)
                route.abort("failed")
            else:
                route.continue_()

        page.route("**/actions/*/confirm", lose_response)
        page.get_by_role("button", name="确认执行").click()
        expect(page.locator("#retry")).to_be_visible()
        expect(page.locator("#retry")).to_be_enabled()
        page.get_by_role("button", name="重试原请求").click()
        expect(page.locator("#connection")).to_have_text("本轮完成")
        expect(page.locator("#pending")).to_be_hidden()
        expect(page.locator("#orders")).to_contain_text("cancelled")
        expect(page.locator("#orders")).to_contain_text("付款记录：2 条")
        completed.append("lost_confirmation_response_retried_without_duplicate_refund")
        page.unroute("**/actions/*/confirm", lose_response)
        page.get_by_role("button", name="查看", exact=True).click()
        expect(page.locator("#trace details").first).to_be_visible()
        page.screenshot(path=str(output / f"{prefix}_desktop_completed.png"), full_page=True)

        for scenario, expected, approve in [
            ("query", "本轮完成", None),
            ("address", "等待消息", False),
            ("return", "本轮完成", True),
        ]:
            page.locator("#scenario").select_option(scenario)
            page.get_by_role("button", name="新建会话").click()
            expect(page.locator("#send")).to_be_enabled()
            page.get_by_role("button", name="发送消息").click()
            if approve is not None:
                expect(page.locator("#pending")).to_be_visible()
                expect(page.locator("#approve")).to_be_enabled()
                page.get_by_role("button", name="确认执行" if approve else "拒绝操作").click()
            expect(page.locator("#connection")).to_have_text(expected)
            expect(page.locator("#send")).to_be_enabled()
            completed.append(scenario + "_interaction")
        assert not errors, errors
        browser.close()
    report = {
        "mode": "headless_browser_local_mock",
        "browser_channel": args.channel,
        "browser_version": version,
        "passed": True,
        "checks": completed,
        "page_errors": errors,
        "screenshots": [
            str(output / name)
            for name in [
                f"{prefix}_desktop_pending.png",
                f"{prefix}_mobile_pending.png",
                f"{prefix}_desktop_completed.png",
            ]
        ],
        "real_model_calls": 0,
    }
    target = ROOT / ".local/results" / f"{prefix}_ui_acceptance.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
