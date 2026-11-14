"use strict";
const $ = (id) => document.getElementById(id);
let saved;
try { saved = JSON.parse(localStorage.getItem("aftersales-session") || "null"); }
catch { saved = null; }
let session = saved, pending = null, lastRequest = null, busy = false;
const labels = {waiting_confirmation:"等待确认",completed:"本轮完成",waiting_user:"等待消息",ready:"就绪",failed:"执行失败",truncated:"预算已耗尽",rejected:"已拒绝",handoff:"已转人工"};
function error(text="") { $("error").textContent = text; }
function setBusy(value) {
  busy = value;
  for (const id of ["create","send","approve","decline","refresh","retry","load-trace"]) $(id).disabled = value || (id !== "create" && !session);
}
function save() { localStorage.setItem("aftersales-session", JSON.stringify(session)); }
async function api(path, options={}) {
  const headers = {"Content-Type":"application/json"};
  if(session) headers.Authorization = "Bearer " + session.token;
  const response = await fetch(path, {...options, headers});
  const data = await response.json();
  if(!response.ok) {
    const message = data.error?.message || (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
    throw new Error(message || "请求未完成");
  }
  return data;
}
function bubble(role, text) {
  const item = document.createElement("div"); item.className = "bubble " + role;
  const label = document.createElement("small"); label.textContent = role === "user" ? "你" : "售后 Agent";
  const body = document.createElement("span"); body.textContent = text;
  item.append(label, body);
  $("chat").querySelector(".empty")?.remove(); $("chat").append(item);
  $("chat").scrollTop = $("chat").scrollHeight;
}
function displayPending(action) {
  pending = action; $("pending").hidden = !action;
  if(action) {
    $("summary").textContent = action.summary;
    $("arguments").textContent = JSON.stringify(action.arguments,null,2);
  }
}
function row(list,key,value) {
  const dt=document.createElement("dt"),dd=document.createElement("dd");
  dt.textContent=key;dd.textContent=String(value ?? "—");list.append(dt,dd);
}
async function refresh() {
  if(!session)return;
  const [state, orders] = await Promise.all([
    api("/sessions/"+session.id), api("/sessions/"+session.id+"/orders")
  ]);
  displayPending(state.session.pending_action);
  const latest=state.latest_request;
  const status=latest?.result?.status || state.session.status;
  $("connection").textContent = labels[status] || status;
  $("status").replaceChildren();
  row($("status"),"会话",session.id.slice(0,12)+"…");
  row($("status"),"模拟用户",state.session.authenticated_user_id);
  row($("status"),"执行状态",labels[status]||status);
  row($("status"),"模型 / 工具",state.usage.model_calls+" / "+state.usage.tool_calls);
  row($("status"),"Token",state.usage.total_tokens === null ? "未知（Mock未报告）" : state.usage.total_tokens);
  row($("status"),"活跃时间",state.usage.active_seconds.toFixed(2)+" 秒");
  $("orders").replaceChildren();
  if(!orders.orders.length) $("orders").textContent="定位用户后显示该用户的订单。";
  for(const order of orders.orders){
    const box=document.createElement("div");box.className="order";
    const name=document.createElement("strong");name.textContent=order.order_id;
    const badge=document.createElement("span");badge.className="state";badge.textContent=order.status;
    const address=document.createElement("p");address.textContent="地址："+order.address.address1+" · "+order.address.city+" "+order.address.zip;
    const payments=document.createElement("p");payments.textContent="付款记录："+order.payment_history.length+" 条 · 订单版本 "+order.version;
    box.append(name,badge,address,payments);$("orders").append(box);
  }
  if(latest && latest.status !== "done") {
    const event=latest.event;
    lastRequest={path:event.kind === "message" ? "/sessions/"+session.id+"/messages" :
      "/sessions/"+session.id+"/actions/"+event.action_id+"/confirm",
      body:event.kind === "message" ? {event_id:event.event_id,content:event.content} :
        {event_id:event.event_id,approved:event.approved}};
    session.inflight=lastRequest;save();$("retry").hidden=false;
  } else {
    session.inflight=null;save();lastRequest=null;$("retry").hidden=true;
  }
  $("message").disabled=false;$("send").disabled=busy;
  return state;
}
async function perform(request) {
  lastRequest=request;session.inflight=request;save();setBusy(true);error();
  try{
    const result=await api(request.path,{method:"POST",body:JSON.stringify(request.body)});
    bubble("assistant",result.reply);
    if(result.status === "failed" && ["timeout","infrastructure_error"].includes(result.termination_reason)){
      $("retry").hidden=false;
    } else {lastRequest=null;session.inflight=null;save();$("retry").hidden=true;}
    await refresh();
  }catch(e){error("请求未完成："+e.message+"。重试会沿用原事件ID。");$("retry").hidden=false;}
  finally{setBusy(false);}
}
$("create").addEventListener("click",async()=>{
  setBusy(true);error();
  try{
    const result=await api("/sessions",{method:"POST",body:JSON.stringify({scenario:$("scenario").value})});
    session={id:result.session.session_id,token:result.access_token,scenario:result.scenario};save();
    lastRequest=null;displayPending(null);$("chat").replaceChildren();
    $("message").value=result.example_message;
    $("trace").textContent="暂无记录";
    bubble("assistant","已创建独立模拟订单环境。请发送消息开始处理。");
    await refresh();
  }catch(e){error(e.message);}
  finally{setBusy(false);}
});
$("message-form").addEventListener("submit",async(event)=>{
  event.preventDefault();if(!session||busy||!$("message").value.trim())return;
  const content=$("message").value.trim();bubble("user",content);
  await perform({path:"/sessions/"+session.id+"/messages",body:{event_id:crypto.randomUUID(),content}});
});
function confirm(approved){
  if(!pending||busy)return;
  return perform({path:"/sessions/"+session.id+"/actions/"+pending.action_id+"/confirm",
    body:{event_id:crypto.randomUUID(),approved}});
}
$("approve").addEventListener("click",()=>confirm(true));
$("decline").addEventListener("click",()=>confirm(false));
$("retry").addEventListener("click",()=>{if(lastRequest)perform(lastRequest);});
$("refresh").addEventListener("click",async()=>{setBusy(true);error();try{await refresh();}catch(e){error(e.message);}finally{setBusy(false);}});
$("load-trace").addEventListener("click",async()=>{
  if(!session)return;setBusy(true);error();
  try{
    const data=await api("/sessions/"+session.id+"/trace");$("trace").replaceChildren();
    for(const event of data.events){
      const details=document.createElement("details"),title=document.createElement("summary"),body=document.createElement("pre");
      title.textContent=event.sequence+" · "+event.kind;
      body.textContent=JSON.stringify(event.payload,null,2);details.append(title,body);$("trace").append(details);
    }
  }catch(e){error(e.message);}finally{setBusy(false);}
});
(async()=>{
  setBusy(false);
  try {
    const health = await api("/health");
    if(health.backend === "deepseek") {
      document.querySelector(".badge").textContent="M3 · DeepSeek 模式";
      $("mode-notice").textContent="当前使用真实模型API。场景仅用于填入示例消息，你可以直接输入需求；订单操作仍仅影响本地模拟数据。";
    }
  } catch { error("服务暂不可用，请检查本地服务是否启动。"); }
  if(session){
    $("scenario").value=session.scenario;lastRequest=session.inflight;
    try{const state=await refresh();if(state.latest_request?.result)bubble("assistant",state.latest_request.result.reply);
      $("message").value="继续所选场景的演示";}
    catch(e){error("无法恢复会话："+e.message);if(lastRequest)$("retry").hidden=false;}
  }
})();
