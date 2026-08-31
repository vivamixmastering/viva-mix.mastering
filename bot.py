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

from aiogram import Bot, Dispatcher
from aiogram.types import BotCommand, MenuButtonCommands

from config import BOT_TOKEN
from src.handlers import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("bot")


async def main() -> None:
    if not BOT_TOKEN:
        log.error("❌ توکن ربات پیدا نشد! فایل .env رو بساز و BOT_TOKEN رو توش بذار.")
        return

    bot = Bot(token=BOT_TOKEN)
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

    # پیام‌های قدیمی که موقع خاموش بودن ربات اومدن رو نادیده بگیر
    await bot.delete_webhook(drop_pending_updates=True)

    log.info("🤖 ربات روشن شد! منتظر پیام...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        log.info("ربات خاموش شد.")
