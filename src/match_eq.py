# -*- coding: utf-8 -*-
"""
match_eq.py — تطبیق کامل با آهنگ مرجع (Reference Matching)

آهنگ مرجع رو آنالیز می‌کنه و «نتیجهٔ شنیداری» اون رو استخراج می‌کنه:
  📊 منحنی تُنال (طیف)  🔊 بلندی LUFS  🎚️ داینامیک (کِرست)  🎧 پهنای استریو
بعد خروجی کاربر رو به همون مشخصات می‌رسونه — بدون عبور از پریست‌ها.

⚠️ نکتهٔ فنی صادقانه: تنظیمات دقیق پلاگین‌ها (اتوتیون/کمپرسور/...) از روی
فایل صوتی قابل استخراج نیست (خروجی = جمعِ همهٔ تصمیم‌ها). ولی «نتیجهٔ شنیداری»
(تُن، بلندی، داینامیک، پهنا) کاملاً قابل اندازه‌گیری و تطبیقه — و این دقیقاً
همون چیزیه که شنونده می‌شنوه.

همهٔ پردازش‌های سنگین، بلوکی (chunked) انجام می‌شن تا از خطای خارج‌شدن
از رم (OOM) روی سرورهای کم‌رم جلوگیری بشه.
"""
from __future__ import annotations

import numpy as np
from scipy import signal as spsig

from src.audio_engine import (
    SR, load_audio, to_mono, integrated_lufs, normalize_lufs,
    width_ms, lin2db,
)

# گرید فرکانسی لاگ‌اسپیس (۴۰Hz تا ۱۸kHz) — ۶۴ نقطه
FREQ_GRID = np.geomspace(40.0, 18000.0, 64).astype(np.float64)

# محدودهٔ گین مجاز برای هر باند (dB)
MAX_BOOST_DB = 8.0
MAX_CUT_DB = -8.0

# لنگر میدرنج برای نرمال‌سازی «شکل» (مستقل از بلندی کلی)
_MID_LO, _MID_HI = 250.0, 4000.0


def _spectrum_db(x, sr):
    """میانگین طیف توان (Welch) → (فرکانس‌ها, dB) روی محدودهٔ مفید."""
    mono = to_mono(x).astype(np.float32)
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


def _true_peak_db(x):
    return float(lin2db(np.max(np.abs(x))))


def _rms_db(x):
    return float(lin2db(np.sqrt(np.mean(np.square(to_mono(x))) + 1e-12)))


def _width_ratio(x):
    """نسبت انرژی پهلو به وسط (side/mid) — شاخص پهنای استریو."""
    if x.ndim != 2 or x.shape[1] < 2:
        return 1.0
    mid = (x[:, 0] + x[:, 1]) / 2.0
    side = (x[:, 0] - x[:, 1]) / 2.0
    sm = float(np.sqrt(np.mean(np.square(side))) + 1e-12)
    mm = float(np.sqrt(np.mean(np.square(mid))) + 1e-12)
    return sm / mm


def analyze(path, sr=SR):
    """آنالیز کامل آهنگ مرجع.

    → dict با کلیدهای JSON-سریال‌شدنی:
      {'grid', 'db', 'warmth_db', 'mid_db', 'brightness_db', 'tilt_db',
       'lufs', 'true_peak_db', 'rms_db', 'crest_db', 'width'}
    """
    x, sr = load_audio(path, sr)

    # ── منحنی تُنال ──
    f, db = _spectrum_db(x, sr)
    db = _on_grid(f, db)
    db = _normalize_shape(db)
    db = _smooth(db)

    profile = {
        "grid": FREQ_GRID.tolist(),
        "db": [round(float(v), 2) for v in db],
    }
    profile.update(_describe(db))

    # ── بلندی / داینامیک / پهنا ──
    profile["lufs"] = round(integrated_lufs(x, sr), 1)
    profile["true_peak_db"] = round(_true_peak_db(x), 1)
    profile["rms_db"] = round(_rms_db(x), 1)
    profile["crest_db"] = round(profile["true_peak_db"] - profile["rms_db"], 1)
    profile["width"] = round(_width_ratio(x), 3)
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


def apply_match_eq(x, sr, profile, amount=1.0, block_s=20.0, overlap_s=2.0):
    """اعمال Match EQ (تُنال) — بلوکی با هم‌پوشانی تا رم مستقل از طول بمونه.

    منحنی گین یک‌بار از کل سیگنال (welch) حساب می‌شه، بعد اعمالِ آن به‌صورت
    STFT بلوکی با crossfade هانینگ انجام می‌شه → بدون OOM روی فایل‌های بلند.
    """
    gain_db = _gain_curve(x, sr, profile, amount)

    mono = x.ndim == 1
    if mono:
        x = x[:, None]

    n = len(x)
    blen = int(block_s * sr)
    ov = int(overlap_s * sr)

    # فرکانس‌های STFT و گین خطی متناظر (یک‌بار حساب می‌شه)
    nperseg = 2048
    hop = nperseg // 4
    noverlap = nperseg - hop
    fstft = np.fft.rfftfreq(nperseg, 1.0 / sr)
    g_lin = np.power(10.0, np.interp(fstft, FREQ_GRID, gain_db) / 20.0)
    g_lin = g_lin.astype(np.float32)[:, None]

    if n <= blen:
        blocks = [(0, n)]
    else:
        blocks = []
        pos = 0
        while pos < n:
            blocks.append((pos, min(pos + blen, n)))
            if pos + blen >= n:
                break
            pos += blen - ov

    out = np.zeros_like(x, dtype=np.float32)
    w = np.zeros(n, dtype=np.float32)

    for (s, e) in blocks:
        seg = x[s:e]
        chans = []
        for c in range(seg.shape[1]):
            ch = seg[:, c].astype(np.float32)
            # boundary='zeros' → stft+istft هم‌طولِ ورودی برمی‌گردونه
            _, _, Z = spsig.stft(ch, fs=sr, nperseg=nperseg, noverlap=noverlap,
                                 boundary="zeros")
            Z2 = (Z * g_lin).astype(np.complex64)
            _, yc = spsig.istft(Z2, fs=sr, nperseg=nperseg, noverlap=noverlap)
            chans.append(yc[: e - s].astype(np.float32))
        yseg = np.stack(chans, axis=1)
        fade = np.hanning(e - s).astype(np.float32)[:, None]
        out[s:e] += yseg * fade
        w[s:e] += fade[:, 0]

    w = np.maximum(w, 1e-8)[:, None]
    out = out / w
    return out[:, 0] if mono else out


def apply_reference_target(x, sr, profile):
    """رسوندن بلندی + پهنا + سقف به مرجع (عملگرهای سبک روی کل سیگنال).

    - پهنای استریو → نسبت side/mid مرجع
    - بلندی → LUFS مرجع (محدود به بازهٔ امن 14- تا 7-)
    - سقف → از طریق normalize_lufs (سقف 1dB-)
    """
    # ── پهنای استریو ──
    ref_width = profile.get("width")
    if ref_width and x.ndim == 2 and x.shape[1] >= 2:
        cur = _width_ratio(x)
        amount = float(np.clip(ref_width / max(cur, 1e-6), 0.7, 1.4))
        x = width_ms(x, amount).astype(np.float32)

    # ── بلندی ──
    ref_lufs = profile.get("lufs")
    if ref_lufs is not None and ref_lufs > -70.0:
        target = float(np.clip(ref_lufs, -14.0, -7.0))
        x = normalize_lufs(x, sr, target=target, ceiling_db=-1.0)

    return x.astype(np.float32)
