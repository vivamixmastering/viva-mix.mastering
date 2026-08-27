# -*- coding: utf-8 -*-
"""
config.py — تنظیمات اصلی ربات
توکن ربات و آیدی ادمین رو از فایل .env می‌خونه.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# توکن ربات تلگرام (از BotFather می‌گیری و توی فایل .env می‌ذاری)
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

# آیدی عددی ادمین (اختیاری) — اگه تنظیم بشه فقط ادمین می‌تونه /on و /off بزنه
ADMIN_ID = int(os.getenv("ADMIN_ID", "0") or "0")

# مسیرهای پروژه
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"          # تنظیمات و فایل‌های موقت
TMP_DIR = DATA_DIR / "tmp"            # فایل‌های صوتی موقت
PRESETS_FILE = BASE_DIR / "presets" / "mastering_presets.yaml"

# محدودیت حجم فایل تلگرام (ربات‌ها فقط می‌تونن فایل تا ۲۰ مگابایت دانلود کنن)
MAX_FILE_SIZE = 20 * 1024 * 1024

for d in (DATA_DIR, TMP_DIR):
    d.mkdir(parents=True, exist_ok=True)
