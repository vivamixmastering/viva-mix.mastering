# -*- coding: utf-8 -*-
"""
pipeline.py — اجرای کامل پردازش بر اساس حالت و پریست

حالت‌ها:
  vocal : فقط زنجیره وکال
  full  : فقط زنجیره مسترینگ روی آهنگ کامل
  two   : وکال + بیت (دو فایل) → میکس و مستر نهایی
  smart : جداسازی خودکار وکال با Demucs → مثل حالت two
"""
import logging
import os
import time
from pathlib import Path

import numpy as np
import yaml

from config import PRESETS_FILE, TMP_DIR
from src.audio_engine import (
    SR, db2lin, duck_under_vocal, encode_mp3, load_audio, master_chain,
    mix_only, save_wav, to_stereo, vocal_chain,
)
from src.match_eq import (
    apply_match_eq, apply_reference_target, build_target_profile,
    match_to_target,
)
from src.reference_chain import derive_master_cfg, derive_vocal_cfg

log = logging.getLogger("pipeline")


def load_presets():
    with open(PRESETS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)["presets"]


PRESETS = load_presets()


def get_preset(pid):
    for p in PRESETS:
        if p["id"] == pid:
            return p
    return PRESETS[0]


def load_mix_models():
    """مدل‌های میکسِ خالص (برای دو استمِ از قبل مسترشده)."""
    with open(PRESETS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f).get("mix_models", [])


MIX_MODELS = load_mix_models()


def get_mix_model(mid):
    for m in MIX_MODELS:
        if m["id"] == mid:
            return m
    return MIX_MODELS[0] if MIX_MODELS else {}


# ══════════════════ پردازش اصلی ══════════════════

def process_mode(paths, mode, preset, workdir=None, match=None):
    """paths: {'vocal':..., 'inst':..., 'full':...} بر اساس حالت
    match: پروفایل مرجع (dict از match_eq.analyze) — اگه داده بشه، ۷ مقدار
           مرجع (تُنال + بلندی + پهنا + داینامیک) روی خروجی پریست «سوار» می‌شن
           بدون دست‌زدن به تنظیمات خود پریست.
    خروجی: (مسیر فایل نهایی, لیست گزارش, ثانیه زمان پردازش)
    """
    workdir = Path(workdir or TMP_DIR)
    t0 = time.time()
    header = f"🎛️ پریست: {preset['name']}"

    if mode == "vocal":
        x, sr = load_audio(paths["vocal"])
        y, vrep = vocal_chain(x, sr, preset.get("vocal", {}))
        rep = [header, "🎤 زنجیره وکال:"] + vrep
    elif mode == "full":
        x, sr = load_audio(paths["full"])
        y, mrep = master_chain(x, sr, preset.get("master", {}))
        rep = [header, "🎵 زنجیره مسترینگ:"] + mrep
    else:
        raise ValueError(f"حالت ناشناخته: {mode}")

    # ── سوار کردن ۷ مقدار مرجع روی خروجی پریست (بدون تغییر پریست) ──
    if match:
        y, measured = match_to_target(y, sr, match)
        rep.append("🎯 ترکیب ۷ تنظیمات مرجع (تُنال/بلندی/پهنا/داینامیک)")
        rep.append(f"   📊 اندازه‌گیریشده: بم {measured['warmth_db']:+}dB • "
                   f"مید {measured['mid_db']:+}dB • درخشش {measured['brightness_db']:+}dB • "
                   f"{measured['lufs']} LUFS")

    # ── اعمال ۷ مقدار هدفِ صریح روی پریست (ref_target) — واقعی، نه عدد الکی ──
    rt = preset.get("ref_target")
    if rt:
        tgt = build_target_profile(
            rt["warmth_db"], rt["mid_db"], rt["brightness_db"], rt["tilt_db"],
            rt["lufs"], rt["crest_db"], rt["width"])
        y, measured = match_to_target(y, sr, tgt)
        rep.append("🎯 اعمال ۷ مقدار هدف (واقعی، با اندازه‌گیری خروجی)")
        rep.append(f"   📊 نتیجهٔ واقعی: بم {measured['warmth_db']:+}dB • "
                   f"مید {measured['mid_db']:+}dB • درخشش {measured['brightness_db']:+}dB • "
                   f"شیب {measured['tilt_db']:+}dB • {measured['lufs']} LUFS • "
                   f"کرست {measured['crest_db']}dB • پهنا {measured['width']}")
        import gc as _gc
        _gc.collect()

    wav = workdir / f"out_{int(time.time())}.wav"
    save_wav(wav, y, sr)
    try:
        mp3 = wav.with_suffix(".mp3")
        encode_mp3(wav, mp3)
        out = mp3
    except Exception as e:
        log.warning("MP3 encode failed: %s", e)
        out = wav
    return out, rep, time.time() - t0


# ══════════════════ میکس خالص (دو استم مسترشده) ══════════════════

def process_mix(paths, model, workdir=None):
    """میکس دو فایلِ از قبل مسترشده (وکال + بیت) — فقط بالانس، بدون مستر دوباره.

    خروجی: (مسیر فایل نهایی, لیست گزارش, ثانیه زمان)
    """
    workdir = Path(workdir or TMP_DIR)
    t0 = time.time()
    header = f"🎛️ مدل میکس: {model.get('name', '')}"

    vx, sr = load_audio(paths["vocal"])
    ix, _ = load_audio(paths["inst"])

    # وکال و بیت جدا لود می‌شن و هم‌طول می‌شن؛ فقط بالانس سبک (بدون اتوتیون/مستر)
    y, mixrep = mix_only(vx, ix, sr, model)
    del vx, ix
    import gc as _gc
    _gc.collect()

    rep = [header, "🎧 میکس خالص (استم‌های مسترشده):"] + mixrep

    wav = workdir / f"mix_{int(time.time())}.wav"
    save_wav(wav, y, sr)
    try:
        mp3 = wav.with_suffix(".mp3")
        encode_mp3(wav, mp3)
        out = mp3
    except Exception as e:
        log.warning("MP3 encode failed: %s", e)
        out = wav
    return out, rep, time.time() - t0


# ══════════════════ مستر مطابق مرجع (بدون پریست) ══════════════════

def _combine(vocal, inst, sr, vocal_db=2.5, inst_db=-4.0, duck_db=3.5):
    """میکس سادهٔ وکال + بیت — بالانس و داکینگ."""
    vocal = to_stereo(vocal.astype(np.float32, copy=False))
    inst = to_stereo(inst.astype(np.float32, copy=False))
    n = max(len(vocal), len(inst))
    if len(vocal) < n:
        vocal = np.pad(vocal, ((0, n - len(vocal)), (0, 0)))
    if len(inst) < n:
        inst = np.pad(inst, ((0, n - len(inst)), (0, 0)))
    if duck_db:
        inst = duck_under_vocal(inst, vocal, sr, duck_db)
    np.multiply(vocal, np.float32(db2lin(vocal_db)), out=vocal)
    np.multiply(inst, np.float32(db2lin(inst_db)), out=inst)
    return (vocal + inst).astype(np.float32)


def process_reference(paths, profile, workdir=None):
    """مستر مطابق مرجع — بدون عبور از پریست‌ها، با زنجیرهٔ کامل مشتق‌شده.

    همهٔ پلاگین‌ها (اتوتیون/ملوداین، کمپرسورها، EQ، دی‌اسر، گرماساز، هوا،
    ریورب، اکو، لیمیتر) بر اساس ۷ مقدار شنیداری مرجع از نو ساخته می‌شن
    (derive_vocal_cfg / derive_master_cfg)، بعد تُنال و بلندی/پهنا/داینامیک
    دقیقاً به مرجع می‌رسن.

    paths: {'vocal':..., 'inst':...} یا {'full':...}
    خروجی: (full_path, vocal_path, inst_path, گزارش, ثانیه)
    """
    import gc as _gc

    workdir = Path(workdir or TMP_DIR)
    t0 = time.time()
    rep = ["🎯 مستر مطابق مرجع (بدون پریست)"]
    sr = SR

    vocal_cfg = derive_vocal_cfg(profile)
    master_cfg = derive_master_cfg(profile)

    has_vocal = bool(paths.get("vocal"))
    has_inst = bool(paths.get("inst"))

    # ── پردازش وکال (اگه هست) — زنجیرهٔ کامل مشتق از مرجع ──
    if has_vocal:
        vx, sr = load_audio(paths["vocal"])
        v, vrep = vocal_chain(vx, sr, vocal_cfg)
        del vx
        _gc.collect()
        rep += ["🎤 زنجیره وکال (بازسازی‌شده از مرجع):"] + vrep
        v = apply_match_eq(v, sr, profile)
        rep.append("🎯 تطبیق تُنال وکال با مرجع")
    else:
        v = None

    # ── پردازش بیت (اگه هست) ──
    if has_inst:
        ix, sr = load_audio(paths["inst"])
        i = apply_match_eq(ix, sr, profile)
        del ix
        _gc.collect()
        rep.append("🎯 تطبیق تُنال بیت با مرجع")
    else:
        i = None

    # ── حالت فقط آهنگ کامل: زنجیرهٔ کامل مستر مشتق از مرجع ──
    if not has_vocal and not has_inst and paths.get("full"):
        fx, sr = load_audio(paths["full"])
        y, mrep = master_chain(fx, sr, master_cfg)
        del fx
        _gc.collect()
        rep += ["🎵 زنجیره مستر (بازسازی‌شده از مرجع):"] + mrep
        y = apply_match_eq(y, sr, profile)
        rep.append("🎯 تطبیق تُنال نهایی با مرجع")
        y, _lufs = apply_reference_target(y, sr, profile)
        rep.append("🎚️ رسوندن بلندی/پهنا/داینامیک به مرجع")
        full_path = _save_out(y, sr, workdir, "ref_full")
        return full_path, None, None, rep, time.time() - t0

    # ── میکس نهایی (وکال + بیت) → زنجیرهٔ مستر مشتق از مرجع ──
    y = _combine(v, i, sr)
    rep.append("🎧 میکس: وکال + بیت (بالانس + داکینگ)")
    del v, i
    _gc.collect()
    y, mrep = master_chain(y, sr, master_cfg)
    rep += ["🎵 زنجیره مستر (بازسازی‌شده از مرجع):"] + mrep
    y = apply_match_eq(y, sr, profile)
    rep.append("🎯 تطبیق تُنال نهایی با مرجع")
    y, _lufs = apply_reference_target(y, sr, profile)
    rep.append("🎚️ رسوندن بلندی/پهنا/داینامیک به مرجع")

    # ── خروجی‌ها ──
    full_path = _save_out(y, sr, workdir, "ref_full")
    del y
    _gc.collect()

    vocal_out = inst_out = None
    if has_vocal:
        vv, _ = load_audio(paths["vocal"])
        vv = vocal_chain(vv, sr, vocal_cfg)[0]
        vv = apply_match_eq(vv, sr, profile)
        vv, _lufs = apply_reference_target(vv, sr, profile)
        vocal_out = _save_out(vv, sr, workdir, "ref_vocal")
        del vv
        _gc.collect()
    if has_inst:
        ii, _ = load_audio(paths["inst"])
        ii = apply_match_eq(ii, sr, profile)
        ii, _lufs = apply_reference_target(ii, sr, profile)
        inst_out = _save_out(ii, sr, workdir, "ref_beat")
        del ii
        _gc.collect()
    return full_path, vocal_out, inst_out, rep, time.time() - t0


def _save_out(y, sr, workdir, stem):
    wav = Path(workdir) / f"{stem}_{int(time.time())}.wav"
    save_wav(wav, y.astype(np.float32), sr)
    try:
        mp3 = wav.with_suffix(".mp3")
        encode_mp3(wav, mp3)
        return mp3
    except Exception as e:
        log.warning("MP3 encode failed: %s", e)
        return wav
