# -*- coding: utf-8 -*-
"""
match_eq.py — تطبیق تُنال با آهنگ مرجع (Match EQ)

پروفایل طیفی یک آهنگ مرجع رو استخراج می‌کنه، و بعد منحنی EQ لازم برای
رسوندن تُنالِ یک فایل (وکال / بیت / آهنگ کامل) به همون منحنی رو محاسبه
و اعمال می‌کنه — تا صدای خروجی «حس و بافت» همون مرجع رو بگیره.

روش (کاملاً خودکار):
  ۱) پروفایل مرجع: میانگین طیف (Welch) → dB روی گرید لاگ‌فرکانسی → لنگر
     میدرنج (۲۵۰–۴kHz = ۰dB) → هموارسازی → ذخیرهٔ منحنی
  ۲) اعمال: برای فایل هدف همون منحنی حساب می‌شه، اختلاف
     gain = ref − target در میاد، محدود می‌شه، و با یک فیلتر FIR
     (firls + کانولوشن FFT) اعمال می‌شه.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as spsig

from src.audio_engine import SR, load_audio, to_mono

# گرید فرکانسی لاگ‌اسپیس (۴۰Hz تا ۱۸kHz) — ۶۴ نقطه
FREQ_GRID = np.geomspace(40.0, 18000.0, 64).astype(np.float64)

# محدودهٔ گین مجاز برای هر باند (dB)
MAX_BOOST_DB = 8.0
MAX_CUT_DB = -8.0

# لنگر میدرنج برای نرمال‌سازی «شکل» (مستقل از بلندی کلی)
_MID_LO, _MID_HI = 250.0, 4000.0


def _spectrum_db(x, sr):
    """میانگین طیف توان (Welch) → (فرکانس‌ها, dB) روی محدودهٔ مفید."""
    mono = to_mono(x).astype(np.float64)
    if len(mono) < 8192:
        mono = np.pad(mono, (0, 8192 - len(mono)))
    f, p = spsig.welch(mono, fs=sr, nperseg=8192, scaling="density")
    mask = (f >= 30.0) & (f <= 18000.0)
    f, p = f[mask], p[mask]
    return f, 10.0 * np.log10(p + 1e-12)


def _smooth(db, k=5):
    k = max(3, (k // 2) * 2 + 1)
    kern = np.hanning(k)
    kern /= kern.sum()
    return np.convolve(db, kern, mode="same")


def _on_grid(f, db):
    return np.interp(FREQ_GRID, f, db)


def _normalize_shape(db):
    """لنگر میدرنج به ۰dB → «شکل» تُنال مستقل از بلندی کلی."""
    mid = float(np.mean(db[(FREQ_GRID >= _MID_LO) & (FREQ_GRID < _MID_HI)]))
    return db - mid


def analyze(path, sr=SR):
    """استخراج پروفایل تُنال از فایل مرجع.

    → dict با کلیدهای JSON-سریال‌شدنی:
      {'grid': [...], 'db': [...], 'warmth_db': .., 'mid_db': ..,
       'brightness_db': .., 'tilt_db': ..}
    """
    x, sr = load_audio(path, sr)
    f, db = _spectrum_db(x, sr)
    db = _on_grid(f, db)
    db = _normalize_shape(db)
    db = _smooth(db)

    profile = {
        "grid": FREQ_GRID.tolist(),
        "db": [round(float(v), 2) for v in db],
    }
    profile.update(_describe(db))
    return profile


def _describe(db):
    """چند توصیف‌گر قابل‌نمایش برای گزارش تلگرام (نسبت به میدرنج)."""
    g = FREQ_GRID
    low = float(db[g < _MID_LO].mean())
    mid = float(db[(g >= _MID_LO) & (g < _MID_HI)].mean())
    high = float(db[g >= _MID_HI].mean())
    return {
        "warmth_db": round(low, 1),       # گرما/بم (زیر 250Hz)
        "mid_db": round(mid, 1),          # میدرنج (250–4k)
        "brightness_db": round(high, 1),  # درخشش (بالای 4k)
        "tilt_db": round(high - low, 1),  # شیب کلی تُنال
    }


def _gain_curve(x, sr, profile, amount=1.0):
    """منحنی گین (dB) برای رسوندن x به پروفایل مرجع."""
    f, db = _spectrum_db(x, sr)
    db = _on_grid(f, db)
    db = _normalize_shape(db)
    db = _smooth(db)
    ref = np.asarray(profile["db"], dtype=np.float64)
    gain = np.clip((ref - db) * float(amount), MAX_CUT_DB, MAX_BOOST_DB)
    return _smooth(gain, 7)


def apply_match_eq(x, sr, profile, amount=1.0):
    """اعمال Match EQ روی سیگنال (float32، مونو یا استریو) → هم‌شکل هم‌طول.

    با شکل‌دهی مستقیم دامنه در حوزهٔ فرکانس (STFT/ISTFT): منحنی گین دقیقاً
    روی طیف اعمال می‌شه (فاز دست‌نخورده، بدون نوسان/رینگِ طراحی فیلتر).
    """
    gain_db = _gain_curve(x, sr, profile, amount)

    mono = x.ndim == 1
    if mono:
        x = x[:, None]

    nperseg = 4096
    hop = nperseg // 4
    noverlap = nperseg - hop

    channels = []
    for c in range(x.shape[1]):
        ch = x[:, c].astype(np.float64)
        f, _, Z = spsig.stft(ch, fs=sr, nperseg=nperseg, noverlap=noverlap,
                             boundary=None, padded=True)
        # گین dB → خطی روی فرکانس‌های STFT
        g = np.interp(f, FREQ_GRID, gain_db)
        Z2 = (Z.astype(np.complex64)
              * np.power(10.0, g / 20.0).astype(np.float32)[:, None])
        _, y = spsig.istft(Z2, fs=sr, nperseg=nperseg, noverlap=noverlap)
        channels.append(y[:len(ch)].astype(np.float32))

    out = np.stack(channels, axis=1)
    return out[:, 0] if mono else out
