"use strict";
(() => {
  const el = (id) => document.getElementById(id);
  let busy = false, lastRequest = null, stage = "E1";
  const initialQuestion = el("knowledge-question").value;
  function defaultQuestion(scope) {
    if (stage === "E4") {
      return scope === "ikea_cn"
        ? "拆掉 SKYTTA 滑门地轨胶带后地板留印，属于品质保证吗？"
        : scope === "demo_store" ? "仅按所选日期的项目模拟条款，配件补寄运费由谁承担？"
        : "项目模拟商店两份文件对包装破损商品的退货期限是否存在冲突？";
    }
    return scope === "ikea_cn"
      ? "宜家用银行卡支付后退货，需要带哪些货品和凭证？"
      : scope === "demo_store" ? "仅按模拟商店所选日期的版本，收货后多少天内可以退货？"
      : "模拟商店两份文件的退货审核天数是否存在冲突？";
  }
  function show(result) {
    el("knowledge-answer").textContent = result.answer?.answer || result.error?.code || "未得到可用回答";
    el("knowledge-sources").replaceChildren();
    for (const source of result.sources || []) {
      const box = document.createElement("div");
      box.className = "knowledge-source";
      const heading = document.createElement("strong");
      heading.textContent = source.chunk_id + " · " + source.section;
      const detail = document.createElement("p");
      detail.textContent = source.version + " · " + (source.origin === "simulated" ? "项目模拟条款" : "官方资料快照")
        + " · 生效日期：" + (source.effective_from || "未知");
      const excerpt = document.createElement("p"); excerpt.textContent = source.text;
      box.append(heading, detail, excerpt);
      if (source.source.startsWith("https://www.ikea.cn/")) {
        const link = document.createElement("a");
        link.href = source.source; link.textContent = "查看官方来源";
        link.target = "_blank"; link.rel = "noopener noreferrer"; box.append(link);
      }
      el("knowledge-sources").append(box);
    }
    const usage = result.usage || {};
    el("knowledge-usage").textContent =
      (result.backend === "mock_demo" ? "离线演示 · " : "模型回答 · ") +
      "Token：" + (usage.reported_tokens ?? "未知") +
      " · 资料片段：" + (result.context_ids || []).length;
  }
  async function perform(body) {
    if (busy) return;
    busy = true; el("knowledge-submit").disabled = true; el("knowledge-retry").disabled = true;
    el("knowledge-error").textContent = "";
    try {
      const response = await fetch("/knowledge/ask", {
        method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)
      });
      const data = await response.json();
      if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : "资料咨询未完成");
      show(data); el("knowledge-retry").hidden = true;
    } catch (error) {
      el("knowledge-error").textContent = error.message;
      el("knowledge-retry").hidden = false;
    } finally {
      busy = false; el("knowledge-submit").disabled = false; el("knowledge-retry").disabled = false;
    }
  }
  el("knowledge-form").addEventListener("submit", (event) => {
    event.preventDefault();
    lastRequest = {
      request_id: crypto.randomUUID(), question: el("knowledge-question").value.trim(),
      scope: el("knowledge-scope").value, mode: el("knowledge-mode").value,
      as_of: el("knowledge-date").value || null
    };
    if (lastRequest.question) perform(lastRequest);
  });
  el("knowledge-retry").addEventListener("click", () => { if (lastRequest) perform(lastRequest); });
  el("knowledge-scope").addEventListener("change", () => {
    const scope = el("knowledge-scope").value;
    el("knowledge-date").value = scope === "ikea_cn" ? "" : "2026-07-01";
    el("knowledge-question").value = defaultQuestion(scope);
  });
  fetch("/knowledge/status").then(r => r.json()).then(data => {
    stage = data.stage || "E1";
    if (stage === "E4" && el("knowledge-question").value === initialQuestion) {
      el("knowledge-question").value = defaultQuestion(el("knowledge-scope").value);
    }
    el("knowledge-mode-status").textContent = data.backend === "deepseek"
      ? "已选择真实模型，提问时调用。资料仅用于咨询，不提交订单操作。"
      : "当前为离线资料演示，展示检索来源，不代表真实模型回答。";
  }).catch(() => { el("knowledge-mode-status").textContent = "无法读取资料咨询状态。"; });
})();
