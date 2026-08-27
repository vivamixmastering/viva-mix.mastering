# -*- coding: utf-8 -*-
"""
power.py — کلید روشن/خاموش سرویس Render از روی خود ربات تلگرام

/off → متغیر BOT_POWER رو از طریق API رندر روی off می‌ذاره
        → رندر دیپلوی مجدد می‌کنه → ربات در حالت خواب (فقط /on جواب می‌ده)
/on  → برعکس.

چرا کامل suspend نمی‌کنیم؟ چون اگه سرویس کلاً خاموش بشه، دیگه /on رو هم نمی‌شنوه!
برای خاموشی کامل سرویس، از داشبورد رندر «Suspend» بزن.

اگه RENDER_API_KEY تنظیم نشده باشه، از حالت درون‌حافظه‌ای استفاده می‌شه
(که روی پلن رایگان رندر با خواب خودکار بعد از ۱۵ دقیقه عملاً همون کار رو می‌کنه).
"""
import logging
import os

import aiohttp

log = logging.getLogger("power")

API = "https://api.render.com/v1"
_mem_power = True


def is_on() -> bool:
    env = os.getenv("BOT_POWER", "on").strip().lower()
    if env in ("off", "0", "false"):
        return False
    return _mem_power


def _creds():
    key = os.getenv("RENDER_API_KEY", "").strip()
    sid = os.getenv("RENDER_SERVICE_ID", "").strip()
    return key, sid


async def set_power(on: bool):
    """برمی‌گردونه:
    'ok'  = با API رندر انجام شد
    None  = کلید API نبود (فقط درون‌حافظه‌ای)
    متن   = خطا
    """
    global _mem_power
    _mem_power = on
    key, sid = _creds()
    if not key or not sid:
        return None
    url = f"{API}/services/{sid}/env-vars/BOT_POWER"
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
