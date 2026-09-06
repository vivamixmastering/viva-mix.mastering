# -*- coding: utf-8 -*-
"""
bot.py — نقطه ورود ربات
اجرا:
    python bot.py

قبل از اجرا:
    1) pip install -r requirements.txt
    2) توکن ربات رو توی فایل .env بذار (نمونه: .env.example)
"""
import asyncio
import logging
import shutil
from pathlib import Path

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.types import BotCommand, MenuButtonCommands

from config import BOT_TOKEN, TMP_DIR
from src.handlers import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("bot")


def cleanup_tmp_on_start() -> None:
    """پاکسازی کامل فایل‌های موقت موقع استارت — بازیابی دیسک پر (سرویس ۱۲۸MB).

    موقع استارت هیچ job فعالی نیست، پس همهٔ پوشه‌های job_* / test_* و فایل‌های
    tmp آپلودشده stale هستن و حذفشون امنه. این جلوی پرشدن دیسک Railway رو می‌گیره.
    """
    try:
        base = Path(TMP_DIR)
        removed = 0
        for p in base.iterdir():
            try:
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink(missing_ok=True)
                removed += 1
            except Exception:
                continue
        base.mkdir(parents=True, exist_ok=True)
        log.info("🧹 پاک‌سازی موقت شروع: %d آیتم حذف شد", removed)
    except Exception as e:  # noqa: BLE001
        log.warning("پاک‌سازی موقت خطا داد: %s", e)


async def main() -> None:
    if not BOT_TOKEN:
        log.error("❌ توکن ربات پیدا نشد! فایل .env رو بساز و BOT_TOKEN رو توش بذار.")
        return

    # سشن با timeout طولانی‌تر — ارسال فایل‌های صوتی بزرگ timeout نشه
    session = AiohttpSession(timeout=600)
    bot = Bot(token=BOT_TOKEN, session=session)
    dp = Dispatcher()
    dp.include_router(router)

    # ── منوی تلگرام (دکمه پایین چپ کنار کادر نوشتن) ──
    commands = [
        BotCommand(command="start", description="🚀 شروع و منوی اصلی"),
        BotCommand(command="presets", description="🎚️ مشاهده ۱۲ پریست میکس و مستر"),
        BotCommand(command="on", description="⚡️ روشن کردن ربات"),
        BotCommand(command="off", description="😴 خاموش کردن ربات"),
        BotCommand(command="test", description="🎧 تست سریع پردازش"),
    ]
    await bot.set_my_commands(commands)
    await bot.set_chat_menu_button(menu_button=MenuButtonCommands())
    log.info("📲 منوی ربات تنظیم شد (دکمه پایین چپ تلگرام)")

    # پاکسازی فایل‌های موقت قدیمی (بازیابی دیسک پر از اجراهای قبلی)
    cleanup_tmp_on_start()

    # پیام‌های قدیمی که موقع خاموش بودن ربات اومدن رو نادیده بگیر
    await bot.delete_webhook(drop_pending_updates=True)

    log.info("🤖 ربات روشن شد! منتظر پیام...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("ربات خاموش شد.")
