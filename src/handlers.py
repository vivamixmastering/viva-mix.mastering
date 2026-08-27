# -*- coding: utf-8 -*-
"""
handlers.py — منطق کامل ربات تلگرام

کامندها:
  /start /help   — راهنما
  /on /off       — کلید روشن/خاموش سرویس (روی خود ربات!)
  /presets       — لیست ۱۰ پریست
  /test          — تست زنجیره با یک صدای کوتاه

روند کار:
  ۱) فایل صوتی می‌فرستی (وکال خالی / آهنگ کامل / دو فایل: وکال + بیت)
  ۲) حالت انتخاب می‌کنی
  ۳) از بین ۱۰ پریست انتخاب می‌کنی
  ۴) ربات پردازش می‌کنه و فایل نهایی رو برمی‌گردونه
"""
import logging
import os
import uuid
from pathlib import Path

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery, FSInputFile, InputMediaAudio, Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from config import ADMIN_ID, MAX_FILE_SIZE, TMP_DIR
from src import power
from src.pipeline import get_preset, load_presets, process_mode, separate_stems, smart_available

log = logging.getLogger("handlers")
router = Router()

AUDIO_EXT = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus", ".wma", ".amr", ".mp4"}
VOCAL_HINTS = ["vocal", "voice", "وکال", "صدا", "اکاپلا", "a capella", "acapella", "vox"]
INST_HINTS = ["inst", "بیت", "موزیک", "ساز", "بدون", "no vocal", "instrumental", "beat", "off vocal"]
FULL_HINTS = ["full", "mix", "آهنگ", "کامل", "کل", "song", "final", "raw"]


class Step(StatesGroup):
    wait_file = State()       # در انتظار فایل اول
    wait_file2 = State()      # در انتظار فایل دوم (وکال+بیت)
    wait_mode = State()
    wait_preset = State()


# ══════════════════ ابزارهای کمکی ══════════════════

def _power_gate() -> str | None:
    """اگه ربات خاموش باشه، پیام خطا برمی‌گردونه (به‌جز /on)."""
    if not power.is_on():
        return ("😴 ربات فعلاً در حالت خوابه (خاموش).\n"
                "برای روشن کردن: /on")
    return None


def _check_admin(user_id: int) -> bool:
    return ADMIN_ID == 0 or user_id == ADMIN_ID


def _fmt_name(ext):
    """پیشوند فارسی برای فایل‌های موقت (فایل‌سیستم ابری UTF-8 رو پشتیبانی می‌کنه)."""
    try:
        return f"ربات_میکس_{uuid.uuid4().hex[:8]}"
    except Exception:
        return f"mix_{uuid.uuid4().hex[:8]}"


def _kb():
    b = InlineKeyboardBuilder()
    return b


def _mode_kb():
    b = _kb()
    b.button(text="🎤 وکال خالی", callback_data="m:vocal")
    b.button(text="🎵 آهنگ کامل (فقط مستر)", callback_data="m:full")
    b.button(text="🎛️ وکال + بیت (دو فایل)", callback_data="m:two")
    b.button(text="✨ هوشمند (جداسازی خودکار)", callback_data="m:smart")
    b.adjust(2)
    return b.as_markup()


def _preset_kb():
    b = _kb()
    for p in load_presets():
        b.button(text=p["name"], callback_data=f"p:{p['id']}")
    b.adjust(1)
    return b.as_markup()


def _preset_desc(p):
    return (f"<b>{p['name']}</b>\n"
            f"<i>{p['desc']}</i>\n\n"
            f"<b>وکال:</b> اتوتیون {int(p['vocal']['tune']['strength'] * 100)}٪ • "
            f"کمپرس {p['vocal']['comp1']['ratio']}:۱ + {p['vocal']['comp2']['ratio']}:۱ • "
            f"هوا @{p['vocal']['air']['freq']}Hz\n"
            f"<b>مستر:</b> لیمیتر {p['master']['lufs']} LUFS • "
            f"پهنا {int(p['master']['width'] * 100)}٪ • سقف {p['master']['ceiling']}dB")


async def _ask_mode(msg, state, edit=False):
    await state.set_state(Step.wait_mode)
    text = ("🎛️ <b>حالت پردازش رو انتخاب کن:</b>\n\n"
            "🎤 <b>وکال خالی</b> — فقط صدای خودت رو می‌دی، زنجیره کامل وکال اجرا می‌شه\n"
            "🎵 <b>آهنگ کامل</b> — آهنگ بدون جداسازی، فقط مسترینگ ۱۰ پریست\n"
            "🎛️ <b>وکال + بیت</b> — دو فایل می‌دی، میکس و مستر نهایی انجام می‌شه\n"
            "✨ <b>هوشمند</b> — آهنگ کامل می‌دی، ربات خودش وکال رو جدا می‌کنه، "
            "وکال و موزیک جدا پردازش می‌شن و دوباره میکس و مستر می‌شن")
    if edit:
        await msg.edit_text(text, reply_markup=_mode_kb(), parse_mode="HTML")
    else:
        await msg.answer(text, reply_markup=_mode_kb(), parse_mode="HTML")


# ══════════════════ کامندهای اصلی ══════════════════

@router.message(Command("start", "help"))
async def cmd_start(msg: Message):
    kb = InlineKeyboardBuilder()
    kb.button(text="🎤 شروع میکس و مستر", callback_data="menu:howto")
    kb.button(text="🎚️ دیدن ۱۰ پریست", callback_data="menu:presets")
    kb.button(text="✨ حالت هوشمند چیه؟", callback_data="menu:smart")
    kb.button(text="⚡️ روشن / 😴 خاموش", callback_data="menu:power")
    kb.adjust(1)
    await msg.answer(
        "🎚️ <b>ویوا میکس مستر</b> — استودیوی خودکار تلگرامی!\n\n"
        "<b>روش کار:</b>\n"
        "۱️⃣ فایل صوتی‌ات رو بفرست (وکال خالی، آهنگ کامل، یا دو فایل وکال+بیت)\n"
        "۲️⃣ حالت پردازش رو انتخاب کن\n"
        "۳️⃣ یکی از <b>۱۰ پریست</b> حرفه‌ای رو انتخاب کن\n"
        "۴️⃣ فایل میکس و مستر شده رو تحویل بگیر ✅\n\n"
        "🎹 پردازش‌ها: اصلاح نت نرم، دی‌اسر، کمپرسور ۱۱۷۶ و LA-2A، کمپرسور موازی، "
        "EQ، گرماساز، هوا و درخشش، اکو، ریورب، پهنای استریو و مسترینگ LUFS\n\n"
        "⬅️ از دکمه <b>منو</b> پایین چپ تلگرام هم می‌تونی همه کامندها رو ببینی.",
        reply_markup=kb.as_markup(), parse_mode="HTML",
    )


@router.callback_query(F.data == "menu:howto")
async def cb_howto(cb: CallbackQuery):
    await cb.message.answer(
        "📥 <b>فقط فایل صوتی رو بفرست</b> (mp3 / wav / ogg / ویس تلگرام، تا ۲۰ مگ)\n\n"
        "اگه می‌خوای <b>وکال + بیت</b> جدا بفرستی، دو فایل رو پشت سر هم بفرست.\n"
        "بعدش حالت و پریست رو انتخاب می‌کنی و فایل نهایی برمی‌گرده! 🎧",
        parse_mode="HTML",
    )
    await cb.answer()


@router.callback_query(F.data == "menu:presets")
async def cb_presets(cb: CallbackQuery):
    lines = ["🎚️ <b>۱۰ پریست میکس و مستر:</b>\n"]
    for i, p in enumerate(load_presets(), 1):
        lines.append(f"{i}. {p['name']} — {p['desc']}")
    lines.append("\nفایل بفرست تا شروع کنیم ⬆️")
    await cb.message.answer("\n".join(lines), parse_mode="HTML")
    await cb.answer()


@router.callback_query(F.data == "menu:smart")
async def cb_smart(cb: CallbackQuery):
    ok, why = smart_available()
    state = ("✅ فعاله" if ok else f"❌ غیرفعاله — {why}")
    await cb.message.answer(
        "✨ <b>حالت هوشمند</b>\n\n"
        "آهنگ کامل می‌فرستی، ربات خودش با هوش مصنوعی (Demucs) وکال رو از موزیک "
        "جدا می‌کنه، وکال و موزیک رو <b>جداگونه</b> پردازش می‌کنه و در نهایت "
        "میکس و مستر نهایی انجام می‌ده.\n\n"
        f"وضعیت روی سرور فعلی: {state}",
        parse_mode="HTML",
    )
    await cb.answer()


@router.callback_query(F.data == "menu:power")
async def cb_power(cb: CallbackQuery):
    await cb.message.answer(
        "⚡️ <b>کنترل ربات:</b>\n\n"
        "/on — روشن کردن سرویس\n"
        "/off — خاموش کردن سرویس (حالت خواب)\n\n"
        "کامندهای دیگه: /presets و /test",
        parse_mode="HTML",
    )
    await cb.answer()


@router.message(Command("presets"))
async def cmd_presets(msg: Message):
    gate = _power_gate()
    if gate:
        await msg.answer(gate)
        return
    lines = ["🎚️ <b>۱۰ پریست میکس و مستر:</b>\n"]
    for i, p in enumerate(load_presets(), 1):
        lines.append(f"{i}. {p['name']} — {p['desc']}")
    lines.append("\nفایل بفرست تا شروع کنیم ⬆️")
    await msg.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("on"))
async def cmd_on(msg: Message):
    if not _check_admin(msg.from_user.id):
        await msg.answer("⛔️ فقط ادمین می‌تونه ربات رو روشن/خاموش کنه.")
        return
    await msg.answer("⚡️ در حال روشن کردن سرویس...")
    res = await power.set_power(True)
    if res is None:
        await msg.answer("✅ ربات روشنه (بدون API رندر).")
    elif res == "ok":
        await msg.answer("✅ روشن شد! رندر داره سرویس رو دوباره راه می‌ندازه — چند ثانیه صبر کن.")
    else:
        await msg.answer(f"⚠️ کلید روشن شد ولی API رندر خطا داد:\n{res}")


@router.message(Command("off"))
async def cmd_off(msg: Message):
    if not _check_admin(msg.from_user.id):
        await msg.answer("⛔️ فقط ادمین می‌تونه ربات رو روشن/خاموش کنه.")
        return
    await msg.answer("🌙 در حال خاموش کردن سرویس...")
    res = await power.set_power(False)
    if res is None:
        await msg.answer("😴 ربات در حالت خوابه (بدون API رندر). با /on برمی‌گرده.")
    elif res == "ok":
        await msg.answer("😴 خاموش شد! رندر داره سرویس رو دوباره راه می‌ندازه — "
                         "بعد از چند ثانیه فقط /on جواب می‌ده.")
    else:
        await msg.answer(f"⚠️ ربات خاموش شد ولی API رندر خطا داد:\n{res}")


@router.message(Command("test"))
async def cmd_test(msg: Message):
    gate = _power_gate()
    if gate:
        await msg.answer(gate)
        return
    st = await msg.answer("🎧 در حال تست زنجیره...")
    try:
        import numpy as np
        from src.audio_engine import SR, save_wav
        from src.pipeline import process_mode
        t = np.arange(SR * 4) / SR
        x = (0.35 * np.sin(2 * np.pi * 220 * t) * (1 + 0.3 * np.sin(2 * np.pi * 2 * t)))
        x = np.stack([x, x], 1).astype("float32")
        f = TMP_DIR / f"test_{uuid.uuid4().hex}.wav"
        save_wav(f, x)
        out, rep, dt = process_mode({"vocal": f}, "vocal", load_presets()[0])
        msg2 = f"✅ تست موفق — پردازش در {dt:.1f} ثانیه\n\n" + "\n".join(rep[:12])
        await msg.answer(msg2, parse_mode="HTML")
        await msg.answer_audio(FSInputFile(out, filename="test_mixed.mp3"))
    except Exception as e:
        log.exception("test failed")
        await msg.answer(f"❌ خطای تست:\n{e}")
    finally:
        try:
            await st.delete()
        except Exception:
            pass


# ══════════════════ دریافت فایل ══════════════════

def _guess_kind(fname):
    n = fname.lower()
    if any(h in n for h in VOCAL_HINTS):
        return "vocal"
    if any(h in n for h in INST_HINTS):
        return "inst"
    if any(h in n for h in FULL_HINTS):
        return "full"
    return "unknown"


def _need_download(typ) -> bool:
    if typ in ("audio", "voice", "video", "document"):
        return True
    if typ in ("video_note",):
        return False
    return False


@router.message(F.audio | F.voice | F.document | F.video)
async def on_audio(msg: Message, state: FSMContext):
    gate = _power_gate()
    if gate:
        await msg.answer(gate)
        return

    typ = (msg.audio and "audio") or (msg.voice and "voice") or \
          (msg.video and "video") or "document"
    if not _need_download(typ):
        await msg.answer("❌ این نوع فایل پشتیبانی نمی‌شه (فقط صدا).")
        return

    fname = ""
    size = 0
    try:
        if typ == "audio":
            fname = msg.audio.file_name or "audio.mp3"
            size = msg.audio.file_size or 0
        elif typ == "voice":
            fname = "voice.ogg"
            size = msg.voice.file_size or 0
        elif typ == "video":
            fname = msg.video.file_name or "video.mp4"
            size = msg.video.file_size or 0
        elif typ == "document":
            fname = msg.document.file_name or "document.bin"
            size = msg.document.file_size or 0
    except Exception:
        pass

    ext = Path(fname).suffix.lower()
    if typ == "document" and ext not in AUDIO_EXT:
        await msg.answer("❌ سند فرستادی که فرمت صوتی نداره! "
                         "فایل صوتی (mp3/wav/ogg/...) بفرست.")
        return

    if size and size > MAX_FILE_SIZE:
        await msg.answer("❌ فایل بزرگ‌تر از ۲۰ مگابایته! ربات‌های تلگرام "
                         "بیشتر از این نمی‌تونن دانلود کنن. فایل سبک‌تر بفرست.")
        return

    ds = await state.get_data()
    first = ds.get("first")
    cur = await state.get_state()
    base = Path(TMP_DIR)
    tmp = base / f"{_fmt_name(ext)}{ext or '.bin'}"

    if cur == Step.wait_file2.state and first:
        # فایل دوم — وکال + بیت
        wait = await msg.answer("📥 فایل دوم در حال دانلود...")
        try:
            await msg.bot.download(msg.document or msg.audio or msg.video or msg.voice, tmp)
        except Exception as e:
            await msg.answer(f"❌ دانلود نشد: {e}")
            return
        try:
            await wait.delete()
        except Exception:
            pass
        await state.update_data(second={"path": str(tmp), "kind": _guess_kind(fname),
                                        "name": fname})
        await _ask_mode(msg, state)
        return

    # فایل اول
    wait = await msg.answer("📥 دانلود شد، آماده‌سازی...")
    try:
        await msg.bot.download(msg.document or msg.audio or msg.video or msg.voice, tmp)
    except Exception as e:
        await msg.answer(f"❌ دانلود نشد: {e}")
        return
    try:
        await wait.delete()
    except Exception:
        pass
    await state.update_data(first={"path": str(tmp), "kind": _guess_kind(fname),
                                   "name": fname})
    await state.set_state(Step.wait_file2)
    await msg.answer(
        "✅ فایل اول گرفتم.\n\n"
        "اگه می‌خوای <b>وکال + بیت (دو فایل)</b> بفرستی، فایل دوم رو الان بفرست.\n"
        "اگه نه، یکی از دکمه‌های زیر رو بزن 👇",
        reply_markup=_mode_kb(), parse_mode="HTML",
    )


# ══════════════════ انتخاب حالت ══════════════════

@router.callback_query(F.data.startswith("m:"))
async def on_mode(cb: CallbackQuery, state: FSMContext):
    if not power.is_on():
        await cb.answer("ربات خاموشه — /on بزن")
        return
    mode = cb.data.split(":", 1)[1]
    ds = await state.get_data()
    first = ds.get("first")
    second = ds.get("second")
    if not first:
        await cb.answer("اول یک فایل صوتی بفرست!")
        return

    # منطق حالت‌ها
    if mode == "two" and not second:
        await cb.answer("برای این حالت دو فایل لازمه: وکال + بیت")
        await state.set_state(Step.wait_file2)
        await cb.message.answer("فایل دوم (بیت/موزیک) رو بفرست 👇")
        return
    if mode == "two" and second:
        # اگر فایل اول آهنگ کامل بود و فایل دوم بیت، جای‌گذاری مناسب
        pass
    if mode == "smart":
        ok, why = smart_available()
        if not ok:
            await cb.answer(why, show_alert=True)
            return

    await state.update_data(mode=mode)
    await state.set_state(Step.wait_preset)
    await cb.message.edit_text(
        "🎚️ <b>عالی! حالا یکی از ۱۰ پریست رو انتخاب کن:</b>",
        reply_markup=_preset_kb(), parse_mode="HTML",
    )
    await cb.answer()


# ══════════════════ انتخاب پریست و پردازش ══════════════════

@router.callback_query(F.data.startswith("p:"))
async def on_preset(cb: CallbackQuery, state: FSMContext):
    if not power.is_on():
        await cb.answer("ربات خاموشه — /on بزن")
        return
    pid = cb.data.split(":", 1)[1]
    ds = await state.get_data()
    first = ds.get("first")
    if not first:
        await cb.answer("اول فایل بفرست!")
        return

    preset = get_preset(pid)
    mode = ds.get("mode", "full")
    workdir = Path(TMP_DIR) / f"job_{uuid.uuid4().hex}"
    workdir.mkdir(parents=True, exist_ok=True)

    progress = await cb.message.edit_text(
        f"⏳ پردازش با پریست «{preset['name']}» شروع شد...\n"
        f"🧠 حالت: {mode}\n\n"
        "این کار می‌تونه چند دقیقه طول بکشه، صبور باش 🙏",
        parse_mode="HTML",
    )

    try:
        paths = {"vocal": first["path"]}
        if mode == "full":
            paths["full"] = first["path"]
        elif mode in ("two", "smart"):
            if mode == "smart":
                await progress.edit_text("✨ در حال جداسازی وکال با هوش مصنوعی "
                                         "(Demucs)...\nاین مرحله سنگین‌ترین مرحله‌ست، "
                                         "ممکنه چند دقیقه طول بکشه ⏳")
                vp, ip = await _run_async(separate_stems, first["path"], workdir)
                await progress.edit_text("✨ جداسازی تموم شد! در حال پردازش...")
                paths = {"vocal": vp, "inst": ip}
            else:
                ds2 = await state.get_data()
                second = ds2.get("second") or ds.get("second")
                paths = {"vocal": first["path"], "inst": second["path"]}

        out, rep, dt = await _run_async(process_mode, paths, mode, preset, workdir)
        size_mb = Path(out).stat().st_size / 1e6
        lines = [f"✅ <b>تموم شد!</b> پردازش {dt:.0f} ثانیه طول کشید.",
                 f"📦 حجم فایل: {size_mb:.1f} مگابایت", "",
                 "🎛️ <b>مراحل انجام شده:</b>"]
        lines += rep
        await progress.delete()
        await cb.message.answer("\n".join(lines), parse_mode="HTML")
        await cb.message.answer_audio(FSInputFile(out, filename=f"{preset['id']}_final.mp3"),
                                      title=f"{preset['name']}",
                                      performer="MixMaster Bot")
    except Exception as e:
        log.exception("processing failed")
        try:
            await progress.edit_text(f"❌ پردازش خطا داد:\n{e}")
        except Exception:
            await cb.message.answer(f"❌ پردازش خطا داد:\n{e}")
    finally:
        await state.clear()


# ══════════════════ اجرای هم‌زمان (غیرهمگام) ══════════════════

async def _run_async(func, *args, **kwargs):
    import asyncio
    return await asyncio.get_event_loop().run_in_executor(None, lambda: func(*args, **kwargs))
