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
    width_ms, lin2db, db2lin,
)

# گرید فرکانسی لاگ‌اسپیس (۴۰Hz تا ۱۸kHz) — ۶۴ نقطه
FREQ_GRID = np.geomspace(40.0, 18000.0, 64).astype(np.float64)

# محدودهٔ گین مجاز برای هر باند (dB) — ملایم تا کیفیت له نشه
# (قبلاً 8± بود و برای مرجع‌های تیره، های‌ها رو 14dB می‌برید → خفه و کدر)
MAX_BOOST_DB = 3.0
MAX_CUT_DB = -3.5

# قدرت کلی تطبیق — 0.6 یعنی 60٪ راه تا مرجع (نه موبهمو)
DEFAULT_AMOUNT = 0.6

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
    """نسبت انرژی پهلو به وسط (side/mid) — شاخص پهنای استریو.

    بدون ساخت آرایهٔ میانی mid/side (با dot) تا رم مصرف نشه.
    """
    if x.ndim != 2 or x.shape[1] < 2:
        return 1.0
    L = x[:, 0]
    R = x[:, 1]
    LL = float(np.dot(L, L))
    RR = float(np.dot(R, R))
    LR = float(np.dot(L, R))
    # sum(side²) = (LL+RR−2LR)/4 ، sum(mid²) = (LL+RR+2LR)/4
    sm = np.sqrt(max(0.0, (LL + RR - 2.0 * LR)) * 0.25) + 1e-12
    mm = np.sqrt(max(0.0, (LL + RR + 2.0 * LR)) * 0.25) + 1e-12
    return float(sm / mm)


def _widen_to_target(x, sr, target):
    """رسوندن واقعی پهنای استریو به هدف (side/mid = target).

    اگه سیگنال باریک‌تر از هدف باشه (مثل وکال مونو)، از یک سیگنال
    decorrelation‌شده (اختلاف تأخیر هاس ~12ms) برای «ساخت» پهنا استفاده می‌کنه —
    این پهنا واقعیه و با side/mid قابل اندازه‌گیریه، نه صرفاً برچسب.
    اگه عریض‌تر از هدف باشه، فقط پهلو رو کم می‌کنه (بدون دست‌کاری تُنال).

    پیاده‌سازی با حداقل کپی (in-place + dot) تا روی فایل‌های بلند OOM نشه.
    """
    if x.ndim != 2 or x.shape[1] < 2:
        return x
    cur = _width_ratio(x)
    if cur >= target and cur > 0.0:
        # فقط باریک‌کردن — مقیاس پهلو
        amt = float(np.clip(target / cur, 0.05, 1.0))
        return width_ms(x, amt)

    mid = (x[:, 0] + x[:, 1]) * 0.5
    side = (x[:, 0] - x[:, 1]) * 0.5

    # سیگنال decorrelation‌شده از mid (تأخیر هاس ~12ms — درجا)
    d = max(1, int(round(0.012 * sr)))
    decor = np.empty_like(mid)
    decor[:d] = 0.0
    np.subtract(mid[d:], mid[:-d], out=decor[d:])

    m_rms = float(np.sqrt(np.dot(mid, mid) / len(mid)) + 1e-12)
    d_rms = float(np.sqrt(np.dot(decor, decor) / len(decor)) + 1e-12)
    s_rms = float(np.sqrt(np.dot(side, side) / len(side)) + 1e-12)
    if d_rms > 1e-9:
        decor *= (m_rms / d_rms)  # هم‌انرژی با mid (درجا)

    # c*decor به‌طوری‌که rms(side+c*decor)/rms(mid) = target
    # (decor ⊥ side و ⊥ mid تقریباً، پس جمع انرژی‌ها برقراره)
    c = float(np.sqrt(max(0.0, target * target - (s_rms / m_rms) ** 2)))
    side += decor * c  # درجا
    del decor

    s_new = float(np.sqrt(np.dot(side, side) / len(side)) + 1e-12)
    if s_new > 1e-9:
        side *= (target * m_rms / s_new)  # درجا

    out = np.empty_like(x)
    out[:, 0] = mid + side
    out[:, 1] = mid - side
    return out


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


def _gain_curve(x, sr, profile, amount=DEFAULT_AMOUNT,
                max_boost=MAX_BOOST_DB, max_cut=MAX_CUT_DB):
    """منحنی گین (dB) برای رسوندن x به پروفایل مرجع — ملایم و هموار."""
    f, db = _spectrum_db(x, sr)
    db = _on_grid(f, db)
    db = _normalize_shape(db)
    db = _smooth(db)
    ref = np.asarray(profile["db"], dtype=np.float64)
    gain = np.clip((ref - db) * float(amount), float(max_cut), float(max_boost))
    # هموارسازی سنگین‌تر → بدون تغییرات ناگهانی/رینگ
    return _smooth(gain, 11)


def apply_match_eq(x, sr, profile, amount=DEFAULT_AMOUNT, block_s=20.0,
                   overlap_s=2.0, max_boost=MAX_BOOST_DB, max_cut=MAX_CUT_DB):
    """اعمال Match EQ (تُنال) — بلوکی با هم‌پوشانی تا رم مستقل از طول بمونه.

    منحنی گین یک‌بار از کل سیگنال (welch) حساب می‌شه، بعد اعمالِ آن به‌صورت
    STFT بلوکی با crossfade هانینگ انجام می‌شه → بدون OOM روی فایل‌های بلند.
    """
    gain_db = _gain_curve(x, sr, profile, amount, max_boost, max_cut)

    mono = x.ndim == 1
    if mono:
        x = x[:, None]

    n = len(x)
    blen = int(block_s * sr)
    ov = int(overlap_s * sr)

    # فرکانس‌های STFT و گین خطی متناظر (یک‌بار حساب می‌شه)
    nperseg = 4096
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
    """رسوندن بلندی + پهنا + داینامیک + سقف به مرجع (زنجیرهٔ لیمیتر اصولی).

    - پهنای استریو → نسبت side/mid مرجع
    - بلندی → LUFS مرجع با لیمیتر واقعی (brickwall با lookahead، سقف 1dB-)
      — کرست نهایی نتیجهٔ طبیعیِ بلندی + سقفه (مثل مسترینگ واقعی، نه عدد جدا)
    → (سیگنال, بلندیِ نهایی LUFS)
    """
    from pedalboard import Limiter

    # ── پهنای استریو (واقعی — اگه لازم باشه پهنا ساخته می‌شه) ──
    ref_width = profile.get("width")
    if ref_width and ref_width > 0.0:
        x = _widen_to_target(x, sr, float(ref_width)).astype(np.float32)

    # ── بلندی با لیمیتر واقعی (gain → limit → تکرار تا LUFS به هدف برسه) ──
    ref_lufs = profile.get("lufs")
    final_lufs = None
    if ref_lufs is not None and ref_lufs > -70.0:
        target = float(np.clip(ref_lufs, -14.0, -6.0))
        x = _limiter_to_lufs(x, sr, target, ceiling_db=-1.0)
        final_lufs = integrated_lufs(x, sr)

    return x.astype(np.float32), final_lufs


def _limiter_to_lufs(x, sr, target_lufs, ceiling_db=-1.0):
    """رسوندن بلندی به هدف با زنجیرهٔ گلو + گین + سقف‌گذاری صادقانه (بدون کلیپ).

    - یک کمپرس ملایم «گلو» (چگالی/قدرت بیشتر، کاهش کرست جزئی)
    - گین تا هدف + سقف بدون کلیپ (normalize_lufs) — اگه کرست اجازه بده دقیقاً
      به هدف می‌رسه؛ اگه نه، بلندترین حالتِ بدون دیستورشن رو می‌ده.
    """
    from pedalboard import Compressor

    y = x.astype(np.float32)
    # گلو: کمپرس ملایم برای چگالی/قدرت بیشتر (بدون له شدن داینامیک)
    y = np.asarray(
        Compressor(threshold_db=-10.0, ratio=2.5, attack_ms=10.0,
                   release_ms=120.0)(y, sr), dtype=np.float32)
    # گین تا هدف (تکرار تا همگرایی، سقف بدون کلیپ)
    for _ in range(8):
        l = integrated_lufs(y, sr)
        if l <= -70.0 or abs(target_lufs - l) < 0.1:
            break
        y = normalize_lufs(y, sr, target=target_lufs,
                           ceiling_db=ceiling_db, max_boost_db=30.0)
    # گارد نهایی سقف — هرگز بالای سقف نره (بدون کلیپ/دیستورشن)
    ceil = float(db2lin(ceiling_db))
    pk = float(np.max(np.abs(y)))
    if pk > ceil:
        y = (y * np.float32(ceil / pk)).astype(np.float32)
    return y.astype(np.float32)


# ══════════════════ پروفایل هدف از ۷ مقدار ══════════════════

def build_target_profile(warmth_db, mid_db, brightness_db, tilt_db,
                         lufs, crest_db, width):
    """ساخت پروفایل کامل هدف از ۷ توصیف‌گر شنیداری (نه عدد الکی — منحنی تُنال
    واقعی از همین مقادیر بازسازی می‌شه).

    - زیر ۲۵۰Hz  → warmth_db (گرما/بم)
    - ۲۵۰–۴kHz   → mid_db (میدرنج = لنگر)
    - بالای ۴kHz → brightness_db (درخشش)
    - tilt_db     فقط برای ثبت/گزارش (high − low)
    """
    db = np.zeros_like(FREQ_GRID)
    low = FREQ_GRID < _MID_LO
    mid = (FREQ_GRID >= _MID_LO) & (FREQ_GRID < _MID_HI)
    high = FREQ_GRID >= _MID_HI
    db[low] = float(warmth_db)
    db[mid] = float(mid_db)
    db[high] = float(brightness_db)
    db = _smooth(db, 9)

    return {
        "grid": FREQ_GRID.tolist(),
        "db": [round(float(v), 2) for v in db],
        "warmth_db": round(float(warmth_db), 1),
        "mid_db": round(float(mid_db), 1),
        "brightness_db": round(float(brightness_db), 1),
        "tilt_db": round(float(tilt_db), 1),
        "lufs": float(lufs),
        "crest_db": float(crest_db),
        "width": float(width),
    }


def _measure(x, sr):
    """اندازه‌گیری توصیف‌گرهای شنیداری یک سیگنال (برای اثبات تطبیق واقعی)."""
    f, db = _spectrum_db(x, sr)
    db = _on_grid(f, db)
    db = _normalize_shape(db)
    db = _smooth(db)
    desc = _describe(db)
    return {
        "warmth_db": desc["warmth_db"],
        "mid_db": desc["mid_db"],
        "brightness_db": desc["brightness_db"],
        "tilt_db": desc["tilt_db"],
        "lufs": round(integrated_lufs(x, sr), 1),
        "crest_db": round(_true_peak_db(x) - _rms_db(x), 1),
        "width": round(_width_ratio(x), 3),
    }


def match_to_target(x, sr, target, amount=1.0, max_boost=15.0, max_cut=-15.0):
    """رسوندن واقعی سیگنال به ۷ مقدار هدف + برگرداندن مقادیر اندازه‌گیریشده.

    → (سیگنال, dict مقادیر واقعیِ بعد از پردازش)
    """
    # ۱) تُنال: Match EQ دقیق (سقف‌های بالا چون هدف صریحه)
    y = apply_match_eq(x, sr, target, amount=amount,
                       max_boost=max_boost, max_cut=max_cut)
    # ۲) بلندی/پهنا/داینامیک
    y, _ = apply_reference_target(y, sr, target)
    return y, _measure(y, sr)
