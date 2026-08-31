# -*- coding: utf-8 -*-
"""
reference_chain.py — ساخت زنجیرهٔ کامل پردازش از ۷ مقدار مرجع

پروفایل شنیداری مرجع (گرما/بم، میدرنج، درخشش، شیب، بلندی LUFS، داینامیک
«کرست»، پهنا) رو می‌گیره و بر اساس اون، تنظیمات کامل زنجیرهٔ وکال و مستر
رو «از نو» می‌نویسه — اتوتیون/ملوداین، کمپرسورها، EQ، دی‌اسر، گرماساز،
هوا، ریورب، اکو و لیمیتر. این یک بازسازی «شنیداری» است (نه استخراج دقیق
پلاگین‌های سازندهٔ مرجع — که از فایل صوتی ممکن نیست)، ولی نتیجهٔ شنیداری
رو به سمت همون کاراکتر می‌بره.

نقشهٔ مشتق‌سازی (هیوریستیک مهندسی صدا):
  warmth_db    → شلف بم + درایو گرماساز
  brightness_db→ شلف بالا + میزان/درایوِ Air Exciter
  crest_db     → شدت کمپرسورها (کرست بالا = داینامیک‌تر = کمپرس ملایم‌تر)
  lufs         → هدف بلندی خروجی
  width        → پهنای استریو
"""
from __future__ import annotations

import numpy as np


def _clip(v, lo, hi):
    return float(np.clip(v, lo, hi))


def derive_vocal_cfg(profile):
    """ساخت زنجیرهٔ کامل وکال از پروفایل مرجع."""
    warm = float(profile.get("warmth_db", 0.0))
    bright = float(profile.get("brightness_db", 0.0))
    crest = float(profile.get("crest_db", 10.0))

    # شلف بم از گرما (محدود تا افراطی نشه)
    low_gain = _clip(warm * 0.4, -3.0, 4.0)
    # شلف بالا + هوا از درخشش
    high_gain = _clip(bright * 0.5, -5.0, 3.0)
    gloss_gain = _clip(high_gain * 0.8, -3.0, 2.5)
    # Air exciter: مرجع تیره → هوای کمتر
    air_mix = _clip(0.14 + bright * 0.015, 0.05, 0.3)
    air_drive = _clip(1.6 + bright * 0.08, 0.5, 2.5)
    # گرماساز از گرما
    warm_drive = _clip(1.5 + warm * 0.15, 1.0, 4.0)
    # کمپرسور از کرست: کرست بالا (داینامیک) → نسبت کمتر
    ratio = _clip(np.interp(crest, [8.0, 12.0, 16.0], [3.5, 2.2, 1.4]), 1.3, 5.0)

    return {
        "hpf_hz": 90,
        "working_lufs": -20,
        "tune": {"strength": 0.55, "snap": 55, "scale": "auto", "vibrato_keep": 0.9},
        "deplosive": {"threshold_db": -14.0, "ratio": 4.0},
        "comp1": {"threshold_db": -14, "ratio": ratio, "attack_ms": 8, "release_ms": 120},
        "comp2": {"threshold_db": -16, "ratio": _clip(ratio * 0.8, 1.3, 3.5),
                  "attack_ms": 20, "release_ms": 300},
        "parallel": {"threshold_db": -40, "ratio": 6, "mix": 0.2},
        "eq_low_shelf": {"freq": 110, "gain_db": low_gain},
        "eq_presence": {"freq": 3000, "gain_db": 2.0},
        "eq_gloss": {"freq": 9500, "gain_db": gloss_gain},
        "eq_air_shelf": {"freq": 12000, "gain_db": high_gain},
        "deess": {"threshold_db": -24.0, "ratio": 6.0, "freq": 7200},
        "harshness": {"freq": 3500.0, "threshold_db": -22.0, "ratio": 3.5},
        "warmth": {"drive_db": warm_drive, "mix": 0.3},
        "air": {"freq": 12000, "drive_db": air_drive, "mix": air_mix},
        "delay": {"time_s": 0.19, "feedback": 0.25, "mix": 0.08},
        "reverb": {"room": 0.35, "damping": 0.55, "wet": 0.13},
        "out_lufs": -18,
    }


def derive_master_cfg(profile):
    """ساخت زنجیرهٔ کامل مستر از پروفایل مرجع."""
    warm = float(profile.get("warmth_db", 0.0))
    bright = float(profile.get("brightness_db", 0.0))
    crest = float(profile.get("crest_db", 10.0))
    lufs = _clip(profile.get("lufs", -10.0), -14.0, -6.0)
    width = float(profile.get("width", 0.5))

    low_gain = _clip(warm * 0.4, -2.0, 3.0)
    high_gain = _clip(bright * 0.5, -4.0, 2.0)
    bus_ratio = _clip(np.interp(crest, [8.0, 12.0, 16.0], [2.5, 1.8, 1.3]), 1.2, 3.0)
    width_amt = _clip(width / 0.5, 0.8, 1.3)
    sat_drive = _clip(1.0 + warm * 0.1, 0.5, 3.0)

    return {
        "working_lufs": -18,
        "hpf_hz": 25,
        "eq_dip": {"freq": 300, "gain_db": -1.0, "q": 1.2},
        "bus_comp": {"threshold_db": -14, "ratio": bus_ratio, "attack_ms": 20,
                     "release_ms": 250},
        "bus_parallel": {"threshold_db": -30, "ratio": 2.0, "mix": 0.2},
        "eq_low_shelf": {"freq": 80, "gain_db": low_gain},
        "eq_high_shelf": {"freq": 10000, "gain_db": high_gain},
        "width": width_amt,
        "sat": {"drive_db": sat_drive, "mix": 0.18},
        "clip": {"ceiling_db": -2.2},
        "lufs": lufs,
        "ceiling": -1.0,
    }
