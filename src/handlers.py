# -*- coding: utf-8 -*-
"""
handlers.py — منطق کامل ربات تلگرام

کامندها:
  /start /help   — راهنما
  /match         — تطبیق تُنال با آهنگ مرجع (Match EQ)
  /on /off       — کلید روشن/خاموش سرویس (روی خود ربات!)
  /presets       — لیست ۱۲ پریست
  /test          — تست زنجیره با یک صدای کوتاه

روند کار:
  ۱) فایل صوتی می‌فرستی (وکال خالی / آهنگ کامل / دو فایل: وکال + بیت)
  ۲) (اختیاری) آهنگ مرجع آپلود می‌کنی تا تُنال خروجی بهش نزدیک بشه
  ۳) حالت انتخاب می‌کنی
  ۴) از بین ۱۲ پریست انتخاب می‌کنی
  ۵) ربات پردازش می‌کنه و فایل نهایی رو برمی‌گردونه
"""
import logging
import uuid
from pathlib import Path

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery, FSInputFile, Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from config import ADMIN_ID, MAX_FILE_SIZE, TMP_DIR
from src import power
from src.match_eq import analyze as analyze_reference
from src.pipeline import (
    HARMONY_INTERVALS, get_mix_model, get_preset, load_mix_models,
    load_presets, process_harmony, process_mix, process_mode,
    process_reference,
)

log = logging.getLogger("handlers")
router = Router()

AUDIO_EXT = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus", ".wma", ".amr", ".mp4"}
VOCAL_HINTS = ["vocal", "voice", "وکال", "صدا", "اکاپلا", "a capella", "acapella", "vox"]
INST_HINTS = ["inst", "بیت", "موزیک", "ساز", "بدون", "no vocal", "instrumental", "beat", "off vocal"]
FULL_HINTS = ["full", "mix", "آهنگ", "کامل", "کل", "song", "final", "raw"]

# ── جداکنندهٔ بصری پیام‌ها ──
SEP = "─" * 24


class Step(StatesGroup):
    wait_ref = State()         # در انتظار فایل مرجع (Match EQ)
    wait_file = State()        # در انتظار فایل اول
    wait_file2 = State()       # در انتظار فایل دوم (وکال+بیت)
    wait_mode = State()
    wait_preset = State()
    wait_mixmodel = State()    # در انتظار انتخاب مدل میکس خالص
    wait_interval = State()    # در انتظار انتخاب فاصلهٔ هارمونی (سوم/پنجم)


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
    try:
        return f"ربات_میکس_{uuid.uuid4().hex[:8]}"
    except Exception:
        return f"mix_{uuid.uuid4().hex[:8]}"


def _kb():
    return InlineKeyboardBuilder()


def _audio_meta(msg: Message):
    """استخراج (نوع, نام فایل, حجم) از یک پیام صوتی."""
    typ = (msg.audio and "audio") or (msg.voice and "voice") or \
          (msg.video and "video") or "document"
    fname, size = "", 0
    try:
        if typ == "audio":
            fname = msg.audio.file_name or "audio.mp3"
            size = msg.audio.file_size or 0
        elif typ == "voice":
            fname, size = "voice.ogg", msg.voice.file_size or 0
        elif typ == "video":
            fname = msg.video.file_name or "video.mp4"
            size = msg.video.file_size or 0
        elif typ == "document":
            fname = msg.document.file_name or "document.bin"
            size = msg.document.file_size or 0
    except Exception:
        pass
    return typ, fname, size


def _file_obj(msg: Message):
    return msg.document or msg.audio or msg.video or msg.voice


def _extract_key(rep):
    """استخراج نام دقیق گام از گزارش مراحل (اگه وجود داشته باشه)."""
    for r in rep:
        if "گام:" in r:
            return r.split("گام:", 1)[1].strip()
    return None


def _mode_kb():
    b = _kb()
    b.button(text="🎤 وکال خالی", callback_data="m:vocal")
    b.button(text="🎵 آهنگ کامل (فقط مستر)", callback_data="m:full")
    b.button(text="🔀 میکس خالص (استم‌های مسترشده)", callback_data="m:mix")
    b.button(text="🎶 هارمونی (فاصله سوم/پنجم)", callback_data="m:harmony")
    b.button(text="🎯 مطابق مرجع (بدون پریست)", callback_data="m:match")
    b.adjust(2)
    return b.as_markup()


def _interval_kb():
    b = _kb()
    b.button(text="3️⃣ سوم (ماژور +4)", callback_data="h:third_maj")
    b.button(text="3️⃣ سوم (مینور +3)", callback_data="h:third_min")
    b.button(text="5️⃣ پنجم (+7)", callback_data="h:fifth")
    b.button(text="🎵 سوم + پنجم (هر دو)", callback_data="h:both")
    b.adjust(1)
    return b.as_markup()


def _preset_kb():
    b = _kb()
    for p in load_presets():
        b.button(text=p["name"], callback_data=f"p:{p['id']}")
    b.adjust(1)
    return b.as_markup()


def _mixmodel_kb():
    b = _kb()
    for m in load_mix_models():
        b.button(text=m["name"], callback_data=f"x:{m['id']}")
    b.adjust(1)
    return b.as_markup()


def _presets_text():
    """لیست پریست‌ها با جداکننده — برای نمایش تمیز."""
    lines = ["🎚️ <b>پریست‌های میکس و مستر</b>", SEP]
    for i, p in enumerate(load_presets(), 1):
        lines.append(f"<b>{i}. {p['name']}</b>")
        lines.append(f"   {p['desc']}")
        lines.append(SEP)
    lines[-1] = ""  # حذف جداکنندهٔ آخر
    lines.append("فایل بفرست تا شروع کنیم ⬆️")
    return "\n".join(lines)


async def _ask_mode(msg, state, edit=False):
    await state.set_state(Step.wait_mode)
    text = ("🎛️ <b>حالت پردازش</b>\n" + SEP + "\n"
            "🎤 <b>وکال خالی</b> — فقط صدای خودت؛ زنجیرهٔ کامل وکال\n"
            "🎵 <b>آهنگ کامل</b> — فقط مسترینگ (زنجیرهٔ بیت)\n"
            "🔀 <b>میکس خالص</b> — دو استمِ مسترشده (وکال + بیت)؛ فقط بالانس\n"
            "🎶 <b>هارمونی</b> — وکال رو به فاصلهٔ سوم/پنجم می‌بره (برای لایه‌گذاری)\n"
            "🎯 <b>مطابق مرجع</b> — بدون پریست، دقیقاً با منحنی آهنگ مرجع")
    if edit:
        await msg.edit_text(text, reply_markup=_mode_kb(), parse_mode="HTML")
    else:
        await msg.answer(text, reply_markup=_mode_kb(), parse_mode="HTML")


# ══════════════════ کامندهای اصلی ══════════════════

@router.message(Command("start", "help"))
async def cmd_start(msg: Message):
    kb = _kb()
    kb.button(text="🎤 شروع میکس و مستر", callback_data="menu:howto")
    kb.button(text="🎚️ پریست‌ها", callback_data="menu:presets")
    kb.button(text="🎯 تطبیق با مرجع (Match EQ)", callback_data="menu:match")
    kb.adjust(1)
    await msg.answer(
        "🎚️ <b>ویوا میکس مستر</b>\n" + SEP + "\n"
        "استودیوی خودکار تلگرامی — میکس و مستر حرفه‌ای وکال و بیت.\n\n"
        "<b>روش کار:</b>\n"
        "۱️⃣ فایل صوتی بفرست (وکال / آهنگ کامل / دو استم مسترشده)\n"
        "۲️⃣ حالت پردازش رو انتخاب کن\n"
        "۳️⃣ یکی از ۱۲ پریست حرفه‌ای رو بزن\n"
        "۴️⃣ خروجی میکس‌شده رو تحویل بگیر ✅\n\n"
        "🎯 <b>نکتهٔ حرفه‌ای:</b> با دکمهٔ «تطبیق با مرجع» یه آهنگ که صداش رو "
        "دوست داری آپلود کن تا تُنال خروجیت دقیقاً به اون نزدیک بشه.",
        reply_markup=kb.as_markup(), parse_mode="HTML",
    )


@router.callback_query(F.data == "menu:howto")
async def cb_howto(cb: CallbackQuery):
    await cb.message.answer(
        "📥 <b>راهنمای سریع</b>\n" + SEP + "\n"
        "• فایل صوتی بفرست (mp3 / wav / ogg / ویس، تا ۲۰ مگ)\n"
        "• برای «میکس خالص» دو استمِ مسترشده (وکال + بیت) پشت سر هم بفرست\n"
        "• حالت و پریست رو انتخاب کن\n"
        "• فایل نهایی برمی‌گرده 🎧\n" + SEP + "\n"
        "🎯 برای تطبیق با آهنگ دلخواه، اول دکمهٔ «تطبیق با مرجع» رو بزن.",
        parse_mode="HTML",
    )
    await cb.answer()


@router.callback_query(F.data == "menu:presets")
async def cb_presets(cb: CallbackQuery):
    await cb.message.answer(_presets_text(), parse_mode="HTML")
    await cb.answer()


@router.callback_query(F.data == "menu:power")
async def cb_power(cb: CallbackQuery):
    await cb.message.answer(
        "⚡️ <b>کنترل ربات</b>\n" + SEP + "\n"
        "/on — روشن کردن سرویس\n"
        "/off — خاموش کردن (حالت خواب)\n" + SEP + "\n"
        "سایر: /presets • /test • /match",
        parse_mode="HTML",
    )
    await cb.answer()


@router.message(Command("presets"))
async def cmd_presets(msg: Message):
    gate = _power_gate()
    if gate:
        await msg.answer(gate)
        return
    await msg.answer(_presets_text(), parse_mode="HTML")


@router.message(Command("on"))
async def cmd_on(msg: Message):
    if not _check_admin(msg.from_user.id):
        await msg.answer("⛔️ فقط ادمین می‌تونه ربات رو روشن/خاموش کنه.")
        return
    await msg.answer("⚡️ در حال روشن کردن سرویس...")
    res = await power.set_power(True)
    if res is None:
        await msg.answer("✅ ربات روشنه (بدون API ابری).")
    elif res == "ok":
        await msg.answer("✅ روشن شد! سرویس ابری داره دوباره راه می‌افته — چند ثانیه صبر کن.")
    else:
        await msg.answer(f"⚠️ کلید روشن شد ولی API ابری خطا داد:\n{res}")


@router.message(Command("off"))
async def cmd_off(msg: Message):
    if not _check_admin(msg.from_user.id):
        await msg.answer("⛔️ فقط ادمین می‌تونه ربات رو روشن/خاموش کنه.")
        return
    await msg.answer("🌙 در حال خاموش کردن سرویس...")
    res = await power.set_power(False)
    if res is None:
        await msg.answer("😴 ربات در حالت خوابه (بدون API ابری). با /on برمی‌گرده.")
    elif res == "ok":
        await msg.answer("😴 خاموش شد! سرویس ابری داره دوباره راه می‌افته — "
                         "بعد از چند ثانیه فقط /on جواب می‌ده.")
    else:
        await msg.answer(f"⚠️ ربات خاموش شد ولی API ابری خطا داد:\n{res}")


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


# ══════════════════ تطبیق با مرجع (Match EQ) ══════════════════

@router.message(Command("match"))
async def cmd_match(msg: Message, state: FSMContext):
    gate = _power_gate()
    if gate:
        await msg.answer(gate)
        return
    await state.set_state(Step.wait_ref)
    await msg.answer(
        "🎯 <b>تطبیق با آهنگ مرجع</b>\n" + SEP + "\n"
        "یه آهنگ که میکس/مسترش رو دوست داری آپلود کن "
        "(مثلاً کار مجید رضوی یا هر خوانندهٔ معروف).\n\n"
        "ربات پروفایل تُنالش رو تحلیل می‌کنه، بعد خروجی تو رو به همون "
        "منحنی می‌رسونه.\n" + SEP + "\n"
        "⬇️ <b>فایل مرجع رو بفرست.</b>",
        parse_mode="HTML",
    )


@router.callback_query(F.data == "menu:match")
async def cb_match(cb: CallbackQuery, state: FSMContext):
    await state.set_state(Step.wait_ref)
    await cb.message.answer(
        "🎯 <b>تطبیق با آهنگ مرجع</b>\n" + SEP + "\n"
        "یه آهنگ که صداش رو دوست داری آپلود کن تا پروفایل تُنالش رو بگیرم.\n"
        + SEP + "\n"
        "⬇️ <b>فایل مرجع رو بفرست.</b>",
        parse_mode="HTML",
    )
    await cb.answer()


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


@router.message(F.audio | F.voice | F.document | F.video)
async def on_audio(msg: Message, state: FSMContext):
    gate = _power_gate()
    if gate:
        await msg.answer(gate)
        return

    typ, fname, size = _audio_meta(msg)
    if typ == "document" and Path(fname).suffix.lower() not in AUDIO_EXT:
        await msg.answer("❌ سند فرستادی که فرمت صوتی نداره! "
                         "فایل صوتی (mp3/wav/ogg/...) بفرست.")
        return
    if size and size > MAX_FILE_SIZE:
        await msg.answer("❌ فایل بزرگ‌تر از ۲۰ مگابایته! ربات‌های تلگرام "
                         "بیشتر از این نمی‌تونن دانلود کنن. فایل سبک‌تر بفرست.")
        return

    ext = Path(fname).suffix.lower()
    cur = await state.get_state()
    base = Path(TMP_DIR)
    tmp = base / f"{_fmt_name(ext)}{ext or '.bin'}"

    # ── حالت مرجع: تحلیل پروفایل تُنال ──
    if cur == Step.wait_ref.state:
        wait = await msg.answer("🔬 در حال تحلیل پروفایل تُنال مرجع...")
        try:
            await msg.bot.download(_file_obj(msg), tmp)
        except Exception as e:
            await msg.answer(f"❌ دانلود نشد: {e}")
            return
        try:
            profile = await _run_async(analyze_reference, str(tmp))
        except Exception as e:
            log.exception("match analyze failed")
            await msg.answer(f"❌ تحلیل مرجع خطا داد:\n{e}")
            return
        finally:
            try:
                await wait.delete()
            except Exception:
                pass

        await state.update_data(ref_profile=profile, ref_name=fname, match=True)
        await state.set_state(Step.wait_file)

        w = profile.get("warmth_db", 0)
        m = profile.get("mid_db", 0)
        b = profile.get("brightness_db", 0)
        t = profile.get("tilt_db", 0)
        lufs = profile.get("lufs")
        crest = profile.get("crest_db")
        width = profile.get("width")

        def _sign(v):
            return f"{'+' if v >= 0 else ''}{v}"

        lines = ["🎯 <b>پروفایل مرجع گرفته شد</b>", SEP,
                 f"📁 <b>فایل:</b> {fname}", SEP,
                 "<b>📊 آنالیز نتیجهٔ شنیداری مرجع:</b>",
                 f"🔥 گرما (بم):     <b>{_sign(w)} dB</b>",
                 f"🎚️ میدرنج:         <b>{_sign(m)} dB</b>",
                 f"✨ درخشش (بالا):  <b>{_sign(b)} dB</b>",
                 f"📈 شیب تُنال:     <b>{_sign(t)} dB</b>"]
        if lufs is not None:
            lines.append(f"🔊 بلندی:         <b>{lufs} LUFS</b>")
        if crest is not None:
            lines.append(f"🎚️ داینامیک (کرست): <b>{crest} dB</b>")
        if width is not None:
            lines.append(f"🎧 پهنا (side/mid): <b>{width}</b>")
        lines += [SEP,
                  "✅ حالا <b>وکال و بیت خودت</b> رو بفرست (یا آهنگ کامل).",
                  "بعد حالت «🎯 مطابق مرجع» رو بزن تا بدون پریست، دقیقاً با "
                  "همین منحنی میکس و مستر بشه."]
        await msg.answer("\n".join(lines), parse_mode="HTML")
        return

    ds = await state.get_data()
    first = ds.get("first")

    if cur == Step.wait_file2.state and first:
        # فایل دوم — وکال + بیت
        wait = await msg.answer("📥 فایل دوم در حال دانلود...")
        try:
            await msg.bot.download(_file_obj(msg), tmp)
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
        await msg.bot.download(_file_obj(msg), tmp)
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
        "✅ فایل اول گرفتم.\n" + SEP + "\n"
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

    if mode == "mix" and not second:
        await cb.answer("برای میکس خالص دو فایل لازمه: وکال + بیت (هر دو مسترشده)")
        await state.set_state(Step.wait_file2)
        await cb.message.answer("فایل دوم (بیت/موزیک مسترشده) رو بفرست 👇")
        return

    # ── حالت مطابق مرجع: بدون پریست، مستقیم پردازش ──
    if mode == "match":
        if not ds.get("ref_profile"):
            await cb.answer("اول با /match یا دکمهٔ «تطبیق با مرجع» یه آهنگ مرجع بده.", show_alert=True)
            return
        await _run_reference(cb, state, first, second)
        return

    # ── میکس خالص: انتخاب مدل میکس (نه پریست مستر) ──
    if mode == "mix":
        await state.update_data(mode=mode)
        await state.set_state(Step.wait_mixmodel)
        await cb.message.edit_text(
            "🔀 <b>میکس خالص — یک مدل میکس انتخاب کن:</b>\n"
            "وکال و بیتِ مسترشده رو فقط بالانس می‌کنم (بدون مستر دوباره).",
            reply_markup=_mixmodel_kb(), parse_mode="HTML")
        await cb.answer()
        return

    # ── هارمونی: انتخاب فاصله (سوم/پنجم) ──
    if mode == "harmony":
        await state.update_data(mode=mode)
        await state.set_state(Step.wait_interval)
        await cb.message.edit_text(
            "🎶 <b>هارمونی — کدوم فاصله؟</b>\n" + SEP + "\n"
            "وکال رو به این فاصله جابه‌جا می‌کنم و با همون پریستِ وکال "
            "پردازش می‌کنم تا خودت روش لایه بذاری.",
            reply_markup=_interval_kb(), parse_mode="HTML")
        await cb.answer()
        return

    await state.update_data(mode=mode)
    await state.set_state(Step.wait_preset)
    await cb.message.edit_text(
        "🎚️ <b>عالی! حالا یکی از پریست‌ها رو انتخاب کن:</b>",
        reply_markup=_preset_kb(), parse_mode="HTML",
    )
    await cb.answer()


# ══════════════════ انتخاب فاصلهٔ هارمونی ══════════════════

@router.callback_query(F.data.startswith("h:"))
async def on_interval(cb: CallbackQuery, state: FSMContext):
    if not power.is_on():
        await cb.answer("ربات خاموشه — /on بزن")
        return
    iv = cb.data.split(":", 1)[1]
    ds = await state.get_data()
    first = ds.get("first")
    if not first:
        await cb.answer("اول فایل وکال بفرست!")
        return

    if iv == "both":
        intervals = [
            HARMONY_INTERVALS["third_maj"],
            HARMONY_INTERVALS["fifth"],
        ]
    else:
        intervals = [HARMONY_INTERVALS[iv]]

    await state.update_data(mode="harmony", intervals=intervals)
    await state.set_state(Step.wait_preset)
    labels = " + ".join(lb for _, lb in intervals)
    await cb.message.edit_text(
        f"🎶 هارمونی انتخاب شد: <b>{labels}</b>\n" + SEP + "\n"
        "🎚️ <b>حالا پریستِ وکال رو انتخاب کن:</b>",
        reply_markup=_preset_kb(), parse_mode="HTML")
    await cb.answer()


async def _run_reference(cb: CallbackQuery, state: FSMContext, first, second):
    """پردازش مطابق مرجع: وکال/بیت جدا + میکس نهایی، بدون پریست."""
    ds = await state.get_data()
    profile = ds.get("ref_profile")
    ref_name = ds.get("ref_name", "")

    workdir = Path(TMP_DIR) / f"job_{uuid.uuid4().hex}"
    workdir.mkdir(parents=True, exist_ok=True)

    paths = {}
    if second:
        paths = {"vocal": first["path"], "inst": second["path"]}
    else:
        paths = {"full": first["path"]}

    progress = await cb.message.edit_text(
        "🎯 <b>مستر مطابق مرجع</b>\n" + SEP + "\n"
        f"📁 مرجع: {ref_name}\n"
        "در حال تطبیق تُنال + بلندی + پهنا... (بدون پریست)\n" + SEP + "\n"
        "این کار ممکنه چند دقیقه طول بکشه 🙏",
        parse_mode="HTML",
    )

    try:
        full, vocal_out, inst_out, rep, dt = await _run_async(
            process_reference, paths, profile, workdir)

        lufs = profile.get("lufs")
        crest = profile.get("crest_db")
        width = profile.get("width")

        lines = ["✅ <b>مستر مطابق مرجع تموم شد!</b>", SEP,
                 f"⏱ زمان: {dt:.0f} ثانیه",
                 f"📁 مرجع: {ref_name}",
                 SEP, "<b>📊 مشخصات مرجع (اعمال‌شده):</b>"]
        if lufs is not None:
            lines.append(f"🔊 بلندی هدف: {lufs} LUFS")
        if crest is not None:
            lines.append(f"🎚️ داینامیک (کرست): {crest} dB")
        if width is not None:
            lines.append(f"🎧 پهنا (side/mid): {width}")
        lines += [SEP, "<b>🧾 مراحل انجام‌شده:</b>"] + rep

        await progress.delete()
        await cb.message.answer("\n".join(lines), parse_mode="HTML")

        # کار کامل
        await cb.message.answer_audio(
            FSInputFile(full, filename="final_full.mp3"),
            title="نسخهٔ کامل", performer="Viva MixMaster")
        # وکال جدا
        if vocal_out:
            await cb.message.answer_audio(
                FSInputFile(vocal_out, filename="final_vocal.mp3"),
                title="وکال جدا (مطابق مرجع)", performer="Viva MixMaster")
        # بیت جدا
        if inst_out:
            await cb.message.answer_audio(
                FSInputFile(inst_out, filename="final_beat.mp3"),
                title="بیت جدا (مطابق مرجع)", performer="Viva MixMaster")
    except Exception as e:
        log.exception("reference processing failed")
        try:
            await progress.edit_text(f"❌ پردازش خطا داد:\n{e}")
        except Exception:
            await cb.message.answer(f"❌ پردازش خطا داد:\n{e}")
    finally:
        await state.clear()


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
    match = ds.get("ref_profile") if ds.get("match") else None
    workdir = Path(TMP_DIR) / f"job_{uuid.uuid4().hex}"
    workdir.mkdir(parents=True, exist_ok=True)

    extra = " + 🎯 تطبیق با مرجع" if match else ""
    progress = await cb.message.edit_text(
        f"⏳ پردازش با پریست «{preset['name']}»{extra} شروع شد...\n"
        f"🧠 حالت: {mode}\n" + SEP + "\n"
        "این کار می‌تونه چند دقیقه طول بکشه، صبور باش 🙏",
        parse_mode="HTML",
    )

    try:
        paths = {"vocal": first["path"]}
        if mode == "full":
            paths["full"] = first["path"]

        # ── هارمونی: چند فایل خروجی (هر فاصله جدا) ──
        if mode == "harmony":
            intervals = ds.get("intervals") or [HARMONY_INTERVALS["third_maj"]]
            outs, rep, dt = await _run_async(process_harmony, paths, preset,
                                             intervals, workdir)
            await progress.delete()
            await cb.message.answer(
                f"✅ <b>هارمونی تموم شد!</b>\n" + SEP + "\n"
                f"⏱ زمان: {dt:.0f} ثانیه\n"
                f"🎛️ پریست: {preset['name']}\n" + SEP + "\n"
                + "\n".join(rep),
                parse_mode="HTML")
            for path, label in outs:
                fname = f"harmony_{preset['id']}_{label.replace(' ', '_')}.mp3"
                await cb.message.answer_audio(
                    FSInputFile(path, filename=fname),
                    title=f"{preset['name']} — {label}",
                    performer="Viva MixMaster")
            return

        out, rep, dt = await _run_async(process_mode, paths, mode, preset,
                                        workdir, match=match)
        size_mb = Path(out).stat().st_size / 1e6
        key = _extract_key(rep)

        lines = ["✅ <b>پردازش تموم شد!</b>", SEP,
                 f"⏱ زمان: {dt:.0f} ثانیه",
                 f"📦 حجم: {size_mb:.1f} مگابایت",
                 f"🎛️ پریست: {preset['name']}"]
        if key:
            lines.append(f"🎵 گام تشخیص‌شده: <b>{key}</b>")
        if match:
            lines.append(f"🎯 مرجع: {ds.get('ref_name', '')}")
        lines += [SEP, "<b>🧾 مراحل انجام‌شده:</b>"] + rep
        await progress.delete()
        await cb.message.answer("\n".join(lines), parse_mode="HTML")
        await cb.message.answer_audio(FSInputFile(out, filename=f"{preset['id']}_final.mp3"),
                                      title=f"{preset['name']}",
                                      performer="Viva MixMaster")
    except Exception as e:
        log.exception("processing failed")
        try:
            await progress.edit_text(f"❌ پردازش خطا داد:\n{e}")
        except Exception:
            await cb.message.answer(f"❌ پردازش خطا داد:\n{e}")
    finally:
        await state.clear()


# ══════════════════ انتخاب مدل میکس خالص ══════════════════

@router.callback_query(F.data.startswith("x:"))
async def on_mixmodel(cb: CallbackQuery, state: FSMContext):
    if not power.is_on():
        await cb.answer("ربات خاموشه — /on بزن")
        return
    mid = cb.data.split(":", 1)[1]
    ds = await state.get_data()
    first = ds.get("first")
    second = ds.get("second")
    if not first or not second:
        await cb.answer("اول دو فایل (وکال + بیت مسترشده) بفرست!")
        return

    model = get_mix_model(mid)
    workdir = Path(TMP_DIR) / f"job_{uuid.uuid4().hex}"
    workdir.mkdir(parents=True, exist_ok=True)

    progress = await cb.message.edit_text(
        f"⏳ میکس خالص با مدل «{model.get('name', '')}»...\n"
        f"🧠 فقط بالانس (بدون مستر دوباره)\n" + SEP + "\n"
        "چند لحظه طول می‌کشه 🙏", parse_mode="HTML")

    try:
        paths = {"vocal": first["path"], "inst": second["path"]}
        out, rep, dt = await _run_async(process_mix, paths, model, workdir)
        size_mb = Path(out).stat().st_size / 1e6

        lines = ["✅ <b>میکس خالص تموم شد!</b>", SEP,
                 f"⏱ زمان: {dt:.0f} ثانیه",
                 f"📦 حجم: {size_mb:.1f} مگابایت",
                 f"🔀 مدل: {model.get('name', '')}"]
        lines += [SEP, "<b>🧾 مراحل:</b>"] + rep
        await progress.delete()
        await cb.message.answer("\n".join(lines), parse_mode="HTML")
        await cb.message.answer_audio(
            FSInputFile(out, filename="final_mix.mp3"),
            title=f"میکس {model.get('name', '')}", performer="Viva MixMaster")
    except Exception as e:
        log.exception("mix failed")
        try:
            await progress.edit_text(f"❌ میکس خطا داد:\n{e}")
        except Exception:
            await cb.message.answer(f"❌ میکس خطا داد:\n{e}")
    finally:
        await state.clear()


# ══════════════════ اجرای هم‌زمان (غیرهمگام) ══════════════════

async def _run_async(func, *args, **kwargs):
    import asyncio
    return await asyncio.get_event_loop().run_in_executor(None, lambda: func(*args, **kwargs))
