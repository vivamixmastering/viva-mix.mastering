# -*- coding: utf-8 -*-
"""
reference_chain.py — ساخت زنجیرهٔ کامل پردازش از ۷ مقدار مرجع

پروفایل شنیداری مرجع (گرما/بم، میدرنج، درخشش، شیب، بلندی LUFS، داینامیک
«کرست»، پهنا) رو می‌گیره و بر اساس اون، تنظیمات کامل زنجیرهٔ وکال و مستر
رو «از نو» می‌نویسه.

اصل راهنما: مرجع فقط «جهت» می‌ده، نه «مقدار مطلق». یعنی اگه مرجع گرم باشه
بم رو کمی بیشتر می‌کنیم، ولی هرگز های وکال رو له نمی‌کنیم — چون وکالِ
شفاف و درخشان همیشه اولویته (فارغ از اینکه مرجع چقدر تیره باشه).
تغییرات تُنال با سقف‌های کوچیک محدود می‌شن تا کیفیت خام خراب نشه.
"""
from __future__ import annotations

import numpy as np


def _clip(v, lo, hi):
    return float(np.clip(v, lo, hi))


def derive_vocal_cfg(profile):
    """ساخت زنجیرهٔ کامل وکال از پروفایل مرجع — شفاف و درخشان، گرمِ ملایم."""
    warm = float(profile.get("warmth_db", 0.0))
    bright = float(profile.get("brightness_db", 0.0))
    crest = float(profile.get("crest_db", 10.0))

    # گرما → شلف بم، محدود (هیچوقت باد نمیشه)
    low_gain = _clip(warm * 0.15, -1.5, 2.5)
    # درخشش → شلف بالا، ولی حداقل -2dB (هیچوقت خفه نمی‌شه)
    high_gain = _clip(bright * 0.15, -2.0, 1.5)
    # جلا/براقیت همیشه مثبت-محور برای شفافیت
    gloss_gain = _clip(1.6 + bright * 0.04, 0.5, 2.5)
    # هوا (Air Exciter) همیشه حضور داره تا وکال روشن بمونه
    air_mix = _clip(0.24 + bright * 0.004, 0.16, 0.32)
    air_drive = _clip(2.0 + bright * 0.03, 1.4, 2.4)
    # گرماساز ملایم از گرما
    warm_drive = _clip(1.8 + warm * 0.1, 1.0, 3.0)
    # کمپرسور از کرست (داینامیک): کرست بالا → نسبت کمتر
    ratio = _clip(np.interp(crest, [8.0, 12.0, 16.0], [3.2, 2.2, 1.4]), 1.3, 4.5)

    return {
        "hpf_hz": 85,
        "working_lufs": -20,
        "tune": {"strength": 0.5, "snap": 50, "scale": "auto", "vibrato_keep": 0.9},
        "deplosive": {"threshold_db": -14.0, "ratio": 4.0},
        "comp1": {"threshold_db": -14, "ratio": ratio, "attack_ms": 8, "release_ms": 120},
        "comp2": {"threshold_db": -16, "ratio": _clip(ratio * 0.8, 1.3, 3.5),
                  "attack_ms": 20, "release_ms": 300},
        "parallel": {"threshold_db": -40, "ratio": 6, "mix": 0.2},
        "eq_low_shelf": {"freq": 110, "gain_db": low_gain},
        "eq_presence": {"freq": 2500, "gain_db": 1.5},
        "eq_gloss": {"freq": 9500, "gain_db": gloss_gain},
        "eq_air_shelf": {"freq": 12000, "gain_db": high_gain},
        "deess": {"threshold_db": -24.0, "ratio": 6.0, "freq": 7200},
        "harshness": {"freq": 3500.0, "threshold_db": -22.0, "ratio": 3.5},
        "warmth": {"drive_db": warm_drive, "mix": 0.28},
        "air": {"freq": 12000, "drive_db": air_drive, "mix": air_mix},
        "delay": {"time_s": 0.19, "feedback": 0.25, "mix": 0.08},
        "reverb": {"room": 0.35, "damping": 0.5, "wet": 0.14},
        "out_lufs": -18,
    }


def derive_master_cfg(profile):
    """ساخت زنجیرهٔ کامل مستر از پروفایل مرجع — گرم و پانچی، بدون له شدن."""
    warm = float(profile.get("warmth_db", 0.0))
    bright = float(profile.get("brightness_db", 0.0))
    crest = float(profile.get("crest_db", 10.0))
    # بلندی مرجع، ولی سقف 7- تا له‌نشدگی/اعوجاج نده
    lufs = _clip(profile.get("lufs", -10.0), -12.0, -7.0)
    width = float(profile.get("width", 0.5))

    low_gain = _clip(warm * 0.12, -1.5, 2.0)
    high_gain = _clip(bright * 0.12, -1.5, 1.0)
    bus_ratio = _clip(np.interp(crest, [8.0, 12.0, 16.0], [2.4, 1.8, 1.3]), 1.2, 3.0)
    width_amt = _clip(width / 0.45, 0.85, 1.25)
    sat_drive = _clip(1.2 + warm * 0.08, 0.5, 2.5)

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
