# -*- coding: utf-8 -*-
"""
power.py — کلید روشن/خاموش سرویس ابری از روی خود ربات تلگرام

/off → متغیر BOT_POWER رو روی off می‌ذاره و سرویس رو دوباره دیپلوی می‌کنه
        → ربات در حالت خواب (فقط /on جواب می‌ده)
/on  → برعکس.

پشتیبانی از دو پلتفرم (اولویت با Railway):
  - Railway: با RAILWAY_API_TOKEN + شناسه‌هایی که خود Railway توی کانتینر
             تزریق می‌کنه (RAILWAY_PROJECT_ID / ENVIRONMENT_ID / SERVICE_ID)
  - Render:  با RENDER_API_KEY + RENDER_SERVICE_ID (سازگاری با نسخهٔ قبل)

چرا کامل suspend نمی‌کنیم؟ چون اگه سرویس کلاً خاموش بشه، دیگه /on رو هم
نمی‌شنوه! برای خاموشی کامل سرویس، از داشبورد «Suspend» بزن.

اگه هیچ کلید API تنظیم نشده باشه، از حالت درون‌حافظه‌ای استفاده می‌شه
(روی پلن رایگان با خواب خودکار عملاً همون کار رو می‌کنه).
"""
import logging
import os

import aiohttp

log = logging.getLogger("power")

RAILWAY_GQL = "https://backboard.railway.app/graphql/v2"
RENDER_API = "https://api.render.com/v1"

_mem_power = True


def is_on() -> bool:
    env = os.getenv("BOT_POWER", "on").strip().lower()
    if env in ("off", "0", "false"):
        return False
    return _mem_power


def platform() -> str | None:
    """تشخیص پلتفرم ابری بر اساس کلیدهای API تنظیم‌شده."""
    if os.getenv("RAILWAY_API_TOKEN", "").strip():
        return "railway"
    if os.getenv("RENDER_API_KEY", "").strip():
        return "render"
    return None


async def _gql(token: str, query: str, variables: dict):
    """اجرای یک کوئری/میوتیشن GraphQL روی Railway → (data یا None, خطا یا None)."""
    headers = {"Authorization": f"Bearer {token}",
               "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=30)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.post(RAILWAY_GQL,
                              json={"query": query, "variables": variables},
                              headers=headers) as r:
                body = await r.json(content_type=None)
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    if not isinstance(body, dict) or "errors" in body:
        errs = (body or {}).get("errors") if isinstance(body, dict) else None
        msg = "; ".join(str(e.get("message", e)) for e in errs) if errs else \
            f"HTTP {r.status}"
        return None, msg
    return body.get("data"), None


async def _railway_set_power(on: bool):
    """کلید روشن/خاموش از طریق Railway GraphQL API.

    شناسه‌های پروژه/سرویس/محیط رو خود Railway توی کانتینر تزریق می‌کنه
    (RAILWAY_*_ID)؛ فقط RAILWAY_API_TOKEN رو باید کاربر به عنوان متغیر
    محیطی اضافه کنه (از https://railway.com/account/tokens).
    """
    token = os.getenv("RAILWAY_API_TOKEN", "").strip()
    pid = os.getenv("RAILWAY_PROJECT_ID", "").strip()
    eid = os.getenv("RAILWAY_ENVIRONMENT_ID", "").strip()
    sid = os.getenv("RAILWAY_SERVICE_ID", "").strip()
    if not (token and pid and eid and sid):
        return ("شناسه‌های Railway ناقصن — RAILWAY_API_TOKEN رو روی سرویس "
                "تنظیم کن (شناسه‌های *_ID خودکار تزریق می‌شن).")

    value = "on" if on else "off"

    # ۱) به‌روزرسانی متغیر BOT_POWER (بدون دیپلوی خودکار — بعداً صریح دیپلوی می‌کنیم)
    data, err = await _gql(
        token,
        "mutation($input: VariableUpsertInput!) { variableUpsert(input: $input) }",
        {"input": {"projectId": pid, "environmentId": eid, "serviceId": sid,
                   "name": "BOT_POWER", "value": value, "skipDeploys": True}},
    )
    if err:
        return f"خطای Railway (variableUpsert): {err}"

    # ۲) دیپلوی مجدد تا مقدار جدید اعمال بشه
    data, err = await _gql(
        token,
        "mutation($environmentId: String!, $serviceId: String!) { "
        "serviceInstanceRedeploy(environmentId: $environmentId, "
        "serviceId: $serviceId) }",
        {"environmentId": eid, "serviceId": sid},
    )
    if err:
        return f"متغیر تغییر کرد ولی redeploy خطا داد: {err}"
    return "ok"


async def _render_set_power(on: bool):
    """کلید روشن/خاموش از طریق API رندر (روش قبلی)."""
    key = os.getenv("RENDER_API_KEY", "").strip()
    sid = os.getenv("RENDER_SERVICE_ID", "").strip()
    if not key or not sid:
        return "شناسه‌های Render ناقصن (RENDER_API_KEY / RENDER_SERVICE_ID)."
    url = f"{RENDER_API}/services/{sid}/env-vars/BOT_POWER"
    headers = {"Authorization": f"Bearer {key}", "Accept": "application/json"}
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            async with s.put(url, json={"value": "on" if on else "off"},
                             headers=headers) as r:
                if r.status == 200:
                    return "ok"
                body = (await r.text())[:200]
                return f"HTTP {r.status}: {body}"
    except Exception as e:
        log.warning("render api error: %s", e)
        return f"خطا: {e}"


async def set_power(on: bool):
    """برمی‌گردونه:
    'ok'  = با API ابری (Railway/Render) انجام شد
    None  = کلید API نبود (فقط درون‌حافظه‌ای)
    متن   = خطا
    """
    global _mem_power
    _mem_power = on
    plat = platform()
    if plat == "railway":
        return await _railway_set_power(on)
    if plat == "render":
        return await _render_set_power(on)
    return None
