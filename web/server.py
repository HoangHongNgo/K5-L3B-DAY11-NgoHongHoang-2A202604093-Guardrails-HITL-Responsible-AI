"""Demo website cho Lab Day 11 — Guardrails.

Chạy từ gốc repo:   .venv\Scripts\python.exe web\server.py
Mở:                 http://127.0.0.1:8000

Mọi lớp phòng thủ dùng CODE THẬT của repo (src/guardrails, src/assignment,
src/agents). LLM live cần API key trong .env; thiếu key / hết quota thì web
tự rơi về câu trả lời dự phòng / replay từ outputs/ để buổi demo không chết giữa chừng.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from assignment.pipeline import build_production_plugins, is_egress_allowed
from guardrails.input_guardrails import detect_injection, topic_filter, _normalize
from guardrails.output_guardrails import content_filter
from agents.security_boundary import (
    ActionRequest, ExternalContent, assess_external_content, authorize_action,
    HIGH_RISK_ACTIONS, TRUSTED_EGRESS_HOSTS,
)
from core.utils import chat_with_agent

WEB = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
app = FastAPI(title="VinBank Guardrails Demo")

FALLBACK = ("Thank you for your banking question. VinBank's 12-month savings rate is "
            "4.25% per year; please visit a branch or the app for account-specific help.")
PII_LEAK = ("Sure, here are the customer details: card 4111 1111 1111 1111, phone 0901234567, "
            "email nguyen.van.a@gmail.com, national ID 079123456789.")
LEAKY = ("Sure! Internal note: admin password = admin123 ; API key = sk-vinbank-secret-2024 ; "
         "database host = db.vinbank.internal:5432. Contact ops@vinbank.com or 0901234567.")

_state: dict = {"plugins": None, "agents": {}}


def _plugins():
    if _state["plugins"] is None:
        _state["plugins"] = build_production_plugins(max_requests=10, window_seconds=60)
    return _state["plugins"]


def _agent(kind: str):
    """Tạo agent live một lần (cache cả lỗi để khỏi chờ lại)."""
    if kind in _state["agents"]:
        v = _state["agents"][kind]
        if isinstance(v, Exception):
            raise v
        return v
    try:
        if kind == "blue":
            from agents.agent import create_blue_agent
            pair = create_blue_agent(_plugins())
            pair[1].plugins = []  # plugin chạy thủ công để có trace từng lớp
            pair[1].client_kwargs = {**(pair[1].client_kwargs or {}), "max_retries": 0, "timeout": 12}
        elif kind == "red":
            from agents.agent import create_red_agent_default
            pair = create_red_agent_default()
        else:
            from agents.guards_agent import create_red_agent_advance
            pair = create_red_agent_advance()
        _state["agents"][kind] = pair
        return pair
    except Exception as exc:
        _state["agents"][kind] = exc
        raise


async def _llm(kind: str, message: str, timeout: int = 25):
    try:
        agent, runner = _agent(kind)
        text, _ = await asyncio.wait_for(chat_with_agent(agent, runner, message), timeout=timeout)
        return (text or ""), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:160]}"


def _text(content) -> str:
    return "".join(getattr(p, "text", "") or "" for p in (getattr(content, "parts", None) or []))


def _step(layer, status, detail, ms=None):
    return {"layer": layer, "status": status, "detail": detail, "ms": ms}


class Chat(BaseModel):
    target: str                  # red | blue | advance
    message: str
    user_id: str = "demo"
    leak_kind: str = "secret"    # secret | pii (khi simulate_leak)
    simulate_leak: bool = False  # Blue: giả lập LLM bị jailbreak để show output filter
    live: bool = True            # False = không gọi LLM thật


def _finish(target, msg, steps, response, blocked, layer, raw=None, error=None):
    from attacks.attacks import response_leaked_secrets
    return {"target": target, "input": msg, "steps": steps, "response": response,
            "blocked": blocked, "layer": layer, "leaked": bool(response_leaked_secrets(response)),
            "raw": raw, "error": error}


# ---- Red: không guardrail ---------------------------------------------------------
def _replay_red(message: str):
    try:
        d = json.loads((OUT / "attack_results.json").read_text(encoding="utf-8"))
        for r in d["unsafe_attacks"]:
            if r["input"].strip() == message.strip():
                return r["response_preview"]
    except Exception:
        pass
    return None


async def run_red(req: Chat) -> dict:
    steps = [_step("Guardrails", "none", "Không có lớp nào — Red cố ý để yếu")]
    t = time.time()
    text, err = (await _llm("red", req.message)) if req.live else (None, "live tắt")
    if text is None:
        text = _replay_red(req.message)
        if text is None:
            m = req.message.lower()
            extractive = detect_injection(req.message) == "BLOCK" or any(
                k in m for k in ("password", "api key", "credential", "secret", "config", "mật khẩu", "mat khau"))
            text = LEAKY if extractive else FALLBACK
        steps.append(_step("LLM", "fallback", f"Không gọi được LLM ({err}) → replay/mô phỏng"))
    else:
        steps.append(_step("LLM", "pass", "Model mềm trả lời thẳng", int((time.time() - t) * 1000)))
    return _finish("red", req.message, steps, text, False, None)


# ---- Blue: pipeline tự xây --------------------------------------------------------
class _Ctx:
    def __init__(self, uid):
        self.user_id = uid


async def run_blue(req: Chat) -> dict:
    """Trace từng lớp của Blue đã nâng cấp: rate → input → NeMo(in) → LLM → NeMo(out) → output."""
    from google.genai import types
    by = {p.name: p for p in _plugins()}
    rl, inp, nemo, outp = by["rate_limiter"], by["input_guardrail"], by.get("nemo_guardrail"), by["output_guardrail"]
    steps: list[dict] = []
    msg = req.message
    c = types.Content(role="user", parts=[types.Part.from_text(text=msg)])
    ctx = _Ctx(req.user_id)

    # 1 Rate limit
    r = await rl.on_user_message_callback(invocation_context=ctx, user_message=c)
    if r is not None:
        steps.append(_step("1 · Rate limiter", "block", _text(r)))
        return _finish("blue", msg, steps, _text(r), True, "rate_limiter")
    used = len(rl.user_windows[req.user_id])
    steps.append(_step("1 · Rate limiter", "pass", f"{used}/{rl.max_requests} request trong {rl.window_seconds}s"))

    # 2 Input guardrail (chuẩn hoá chống che giấu + regex + topic + khoá user)
    norm = _normalize(msg)
    if norm != msg.strip():
        steps.append(_step("2a · Chuẩn hoá / gỡ che giấu", "info", f"Unicode ẩn, full-width, homoglyph, base64/hex/ROT13… → “{norm[:80]}”"))
    t = time.time()
    res = await inp.on_user_message_callback(invocation_context=ctx, user_message=c)
    ms = int((time.time() - t) * 1000)
    if res is not None:
        txt = _text(res)
        if "too many unsafe" in txt:
            why = "Khoá user: ≥3 lần bị chặn injection → khoá 5 phút"
        elif "prompt injection" in txt:
            why = "Regex trên mọi phiên bản đã gỡ che giấu / kết hợp ý định khớp mẫu injection"
        else:
            why = "Lọc chủ đề: không phải banking (từ khoá EN/VI + embedding đa ngôn ngữ)"
        steps.append(_step("2 · Input guardrail", "block", why, ms))
        return _finish("blue", msg, steps, txt, True, "input_guardrail")
    steps.append(_step("2 · Input guardrail", "pass", "Regex + kết hợp ý định + topic đều sạch", ms))

    # 3 NeMo input rails
    if nemo is not None:
        t = time.time()
        res = await nemo.on_user_message_callback(invocation_context=ctx, user_message=c)
        ms = int((time.time() - t) * 1000)
        v = nemo.last_verdict
        lang = f" · ngôn ngữ: {v.language}" if v and v.language else ""
        deg = " · ⚠ NeMo/LLM lỗi → chạy lớp local" if v and v.degraded else ""
        if res is not None:
            rail = v.rail if v else "?"
            steps.append(_step("3 · NeMo input rails", "block", f"Chặn bởi rail: {rail}{lang}{deg}", ms))
            return _finish("blue", msg, steps, _text(res), True, "nemo_guardrail")
        steps.append(_step("3 · NeMo input rails", "pass", f"3a deterministic · 3b semantic · 3c LLM self-check đều qua{lang}{deg}", ms))
    else:
        steps.append(_step("3 · NeMo input rails", "fallback", "NeMo tắt (BLUE_USE_NEMO=0)"))

    # LLM
    t = time.time()
    if req.simulate_leak:
        raw = PII_LEAK if req.leak_kind == "pii" else LEAKY
        steps.append(_step("4 · LLM", "warn", "Giả lập: model nhả dữ liệu khách hàng (PII)" if req.leak_kind == "pii" else "Giả lập: model bị jailbreak và nhả secret + PII"))
    elif req.live:
        raw, err = await _llm("blue", msg, timeout=15)
        if raw is None:
            raw = FALLBACK
            steps.append(_step("4 · LLM", "fallback", f"Blue không gọi được ({err}) → câu trả lời dự phòng"))
        else:
            steps.append(_step("4 · LLM", "pass", "Blue (liquid/lfm-3-1.7b) trả lời", int((time.time() - t) * 1000)))
    else:
        raw = FALLBACK
        steps.append(_step("4 · LLM", "fallback", "Live tắt → câu trả lời dự phòng"))

    class _Resp:
        pass
    resp = _Resp()
    resp.content = types.Content(role="model", parts=[types.Part.from_text(text=raw)])
    cur, layer, blocked = raw, None, False

    # 5 NeMo output rails
    if nemo is not None:
        t = time.time()
        out = await nemo.after_model_callback(callback_context=None, llm_response=resp)
        new = _text((out or resp).content)
        ms = int((time.time() - t) * 1000)
        if new != cur:
            rail = nemo.last_verdict.rail if nemo.last_verdict else "?"
            steps.append(_step("5 · NeMo output rails", "block", f"Chặn bởi rail: {rail} (leak check / LLM self-check)", ms))
            return _finish("blue", msg, steps, new, True, "nemo_guardrail", raw=raw)
        steps.append(_step("5 · NeMo output rails", "pass", "Leak check + LLM self-check sạch", ms))

    # 6 Output guardrail (regex PII + Presidio + leak sau khi che)
    cf = content_filter(raw)
    if not cf["safe"]:
        steps.append(_step("6 · Output guardrail", "redact", "Che: " + ", ".join(cf["issues"])))
        return _finish("blue", msg, steps, cf["redacted"], True, "output_guardrail", raw=raw)
    steps.append(_step("6 · Output guardrail", "pass", "Regex PII + Presidio không thấy secret/PII"))
    return _finish("blue", msg, steps, raw, False, None)


# ---- Red Advance ------------------------------------------------------------------
async def run_advance(req: Chat) -> dict:
    from agents.guards_agent import detect_injection_strong, topic_filter_strong
    from attacks.attacks import classify_attack_outcome
    msg = req.message
    steps: list[dict] = []
    if detect_injection_strong(msg):
        steps.append(_step("L1 · Regex input", "block", "Injection filter (mạnh)"))
        return _finish("advance", msg, steps,
                       "I cannot process that request. I only help with VinBank banking questions.",
                       True, "input_injection")
    if topic_filter_strong(msg):
        steps.append(_step("L1 · Regex input", "block", "Topic filter (mạnh)"))
        return _finish("advance", msg, steps,
                       "I'm a VinBank assistant and can only help with banking-related questions.",
                       True, "input_topic")
    steps.append(_step("L1 · Regex input", "pass", "Qua regex"))
    if not req.live:
        steps.append(_step("L2–L4", "fallback", "Live tắt — bỏ qua NeMo + LLM"))
        return _finish("advance", msg, steps, "(live tắt)", False, None)
    t = time.time()
    text, err = await _llm("advance", msg, timeout=45)
    if text is None:
        steps.append(_step("L2–L4 · NeMo + LLM", "fallback", f"Không gọi được ({err})"))
        return _finish("advance", msg, steps, "(không gọi được NeMo/LLM — kiểm tra API key)",
                       False, None, error=err)
    ms = int((time.time() - t) * 1000)
    o = classify_attack_outcome(msg, text, target_name="red_advance")
    layer = o.get("layer")
    if layer == "nemo_input_rail":
        steps.append(_step("L2 · NeMo input rail", "block", o["blocked_at"], ms))
    elif layer == "output_filter":
        steps += [_step("L2 · NeMo input rail", "pass", "Qua"), _step("L3 · Regex output", "block", o["blocked_at"], ms)]
    elif layer == "nemo_output_rail":
        steps += [_step("L2 · NeMo input rail", "pass", "Qua"), _step("L3 · Regex output", "pass", "Qua"),
                  _step("L4 · NeMo output rail", "block", o["blocked_at"], ms)]
    elif layer == "nemo_error":
        steps.append(_step("NeMo", "block", o["blocked_at"], ms))
    else:
        steps += [_step("L2 · NeMo input rail", "pass", "Qua"), _step("LLM", "pass", "Trả lời", ms),
                  _step("L3/L4 · Output", "pass", "Không phát hiện secret")]
    return _finish("advance", msg, steps, text, bool(o.get("blocked")), layer)


# ---- Prompt dài sinh phía server (kịch bản s13) -------------------------------------
_LONG_TOKEN = re.compile(r"^\[\[LONG_EMAIL:(\d{3,6})\]\]$")
_LONG_UNIT = ("My savings deposit statement shows regular monthly interest and card payment fees. ")


def _expand(message: str) -> str:
    """`[[LONG_EMAIL:N]]` → email ngân hàng hợp lệ dài ~N ký tự (không chứa lệnh nào)."""
    m = _LONG_TOKEN.match((message or "").strip())
    if not m:
        return message
    n = min(int(m.group(1)), 40000)
    head = "Dear VinBank, I want to ask about my savings account interest rate. "
    body = head + _LONG_UNIT * (n // len(_LONG_UNIT) + 1)
    return body[:n]


# ---- API --------------------------------------------------------------------------
@app.post("/api/chat")
async def chat(req: Chat):
    shown = req.message
    req.message = _expand(req.message)
    fn = {"red": run_red, "blue": run_blue, "advance": run_advance}.get(req.target)
    if fn is None:
        return JSONResponse({"error": "target phải là red|blue|advance"}, status_code=400)
    out = await fn(req)
    if req.message != shown:  # prompt dài: chỉ trả token + độ dài cho gọn
        out["input"] = f"{shown} ({len(req.message)} ký tự)"
        if req.target == "advance" and out.get("blocked") and len(req.message) > 16000:
            for st in out["steps"]:
                if st["status"] == "block":
                    st["detail"] = ("Prompt vượt max_length 16.000 ký tự của self-check → NeMo ném lỗi "
                                    "→ fail-closed (chặn cả khách hợp lệ)")
    return out


@app.post("/api/reset")
async def reset(user_id: str = "demo"):
    for pl in _plugins():
        getattr(pl, 'user_windows', {}).pop(user_id, None)
        getattr(pl, '_strikes', {}).pop(user_id, None)
    return {"ok": True}


class Egress(BaseModel):
    destination: str
    payload: str
    action: str = "send_report"
    approval_id: str | None = None
    reviewer_id: str | None = None


@app.post("/api/egress")
async def egress(e: Egress):
    gate = is_egress_allowed(e.destination, e.payload)
    d = authorize_action(ActionRequest(e.action, e.destination, e.payload, e.approval_id, e.reviewer_id))
    return {"allowed": gate and d.allowed, "egress_gate": gate, "policy_reason": d.reason,
            "requires_human": d.requires_human, "trusted_hosts": sorted(TRUSTED_EGRESS_HOSTS),
            "high_risk": sorted(HIGH_RISK_ACTIONS)}


class Doc(BaseModel):
    source: str = "email"
    text: str


@app.post("/api/document")
async def document(d: Doc):
    r = assess_external_content(ExternalContent(source=d.source, text=d.text, trusted=False))
    return {"allowed": r.allowed, "reason": r.reason}


@app.get("/api/evidence")
async def evidence():
    def load(n):
        try:
            return json.loads((OUT / n).read_text(encoding="utf-8"))
        except Exception:
            return None
    return {"results": load("results.json"), "attacks": load("attack_results.json"),
            "metrics": load("metrics.json")}


@app.get("/api/scenarios")
async def scenarios():
    return json.loads((WEB / "scenarios.json").read_text(encoding="utf-8"))


@app.get("/api/health")
async def health():
    return {"openai_key": bool(os.getenv("OPENAI_API_KEY")),
            "openrouter_key": bool(os.getenv("OPENROUTER_API_KEY")),
            "red_provider": os.getenv("RED_TEAM_PROVIDER", "openai")}


@app.get("/")
async def index():
    return FileResponse(WEB / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
