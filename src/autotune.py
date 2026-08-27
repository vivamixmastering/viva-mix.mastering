# -*- coding: utf-8 -*-
"""
autotune.py — اصلاح نرم نت (جایگزین متن‌باز اتوتیون/ملوداین)

با pyworld (تجزیه F0 + پوش طیفی). صدای رباتی نمی‌شه چون:
  1) قدرت اصلاح (strength) پایینه (۳۰-۶۰٪)
  2) نت‌ها فقط تا سقف snap (سنت) جابه‌جا می‌شن
  3) پوش طیفی (فرمانت‌ها) دست‌نخورده می‌مونه
  4) فرکانس‌های بالای ۷.۵kHz از صدای اصلی حفظ می‌شن (جزئیات و هوا از بین نمی‌ره)
  5) اگه ورودی تک‌صدایی نباشه (کُر، چند نفر با هم، ساز) — خودش تشخیص می‌ده و رد می‌شه
  6) پردازش قطعه‌قطعه (پیش‌فرض ۶۰ ثانیه) — مصرف رم مستقل از طول آهنگه

⚠️ باگ مهمی که قبلاً رفع شد:
pw.synthesize پیش‌فرض frame_period=5ms رو فرض می‌کنه در حالی که تحلیل با
frame_period=10ms انجام می‌شه → خروجی نصف طول و صدا لهیده/نامفهوم می‌شد.
حالا frame_period صریح پاس داده می‌شه و طول خروجی هم چک می‌شه.

ملوداین و اتوتیونِ آنتارس API ندارن و نمی‌شه به ربات وصلشون کرد؛
این ماژول نزدیک‌ترین جایگزین قابل اجرای خودکاره.
"""
import logging
import os

import numpy as np
from scipy import signal as spsig

from src.audio_engine import SR, highpass, lowpass, resample

log = logging.getLogger("autotune")

WORK_SR = 24000      # برای سرعت، اصلاح نت روی ۲۴ کیلوهرتز انجام می‌شه
FRAME_PERIOD = 10.0  # میلی‌ثانیه — باید توی synthesize هم همین باشه!
CROSSOVER = 7500.0   # بالای این فرکانس صدای اصلی حفظ می‌شه (هوا و جزئیات تیس)
MIN_RECON_SIM = 0.45  # اگه شباهت طیفی بازسازی کمتر از این باشه → چندصدایی/ناوکال → رد


def _reconstruction_similarity(x, f0, t, fs, frame_period):
    """شباهت طیفی اورجینال و بازسازی WORLD (روی فریم‌های واکدار).
    WORLD فاز رو بازسازی نمی‌کنه، پس باید طیف (نه شکل موج) مقایسه بشه:
    تک‌صدایی تمیز ≈ ۰.۶ تا ۱.۰ — کُر/چندصدایی/ساز خیلی پایین‌تر.
    برای کم‌مصرف بودن: STFT در float32 و فقط روی هر سومین فریم."""
    import pyworld as pw
    sp = pw.cheaptrick(x, f0, t, fs)
    ap = pw.d4c(x, f0, t, fs)
    rec = pw.synthesize(f0, sp, ap, fs, frame_period=frame_period)
    n = min(len(x), len(rec))
    xa = x[:n].astype(np.float32)
    ra = rec[:n].astype(np.float32)

    nfft = 1024
    hop = max(int(frame_period / 1000.0 * fs), 64)
    noverlap = max(nfft - hop, 0)
    if noverlap >= nfft:
        nfft = 2048
        noverlap = nfft - hop
    _, _, Zx = spsig.stft(xa, fs=fs, nperseg=nfft, noverlap=noverlap,
                          boundary=None, padded=False)
    _, _, Zr = spsig.stft(ra, fs=fs, nperseg=nfft, noverlap=noverlap,
                          boundary=None, padded=False)

    nfr = min(Zx.shape[1], Zr.shape[1], len(f0))
    sims = []
    for i in range(0, nfr, 3):  # هر سومین فریم کافیه
        if f0[i] <= 0:
            continue
        fx = np.abs(Zx[:, i]) + 1e-12
        fr = np.abs(Zr[:, i]) + 1e-12
        fx = fx / (np.linalg.norm(fx) + 1e-12)
        fr = fr / (np.linalg.norm(fr) + 1e-12)
        sims.append(float(np.dot(fx, fr)))
    if len(sims) < 10:
        return 0.0
    return float(np.median(sims))


def _tune_segment(x_mono, sr, strength, snap_cents, f0_floor, f0_ceil,
                  work_sr, frame_period, blend):
    """یک تکه تک‌صدایی رو تیون می‌کنه.
    ورودی: float64 مونو با نرخ sr — خروجی: (float32 هم‌طول با نرخ sr, 'ok')
    یا (None, دلیل رد شدن)"""
    import pyworld as pw

    x = np.ascontiguousarray(resample(x_mono, sr, work_sr).astype(np.float64))
    f0, t = pw.dio(x, work_sr, f0_floor=f0_floor, f0_ceil=f0_ceil,
                   frame_period=frame_period)
    f0 = pw.stonemask(x, f0, t, work_sr)

    # ── تشخیص چندصدایی ۱: شباهت طیفی بازسازی ──
    sim = _reconstruction_similarity(x, f0, t, work_sr, frame_period)
    if sim < MIN_RECON_SIM:
        return None, (f"چندصدایی/کُر تشخیص داده شد (شباهت طیفی {sim:.2f}) — "
                      f"اصلاح نت برای جلوگیری از خراب شدن صدا رد شد")

    # ── تشخیص چندصدایی ۲: پایداری F0 ──
    # توی صدای تک‌نفره نت بین فریم‌ها نرم جابه‌جا می‌شه؛ توی کُر/چندصدایی F0 می‌پره.
    # فیلتر میانه خطاهای لحظه‌ای F0 (نویز، ترق‌ولق صفحه قدیمی و...) رو حذف می‌کنه
    # تا صدای سالم به اشتباه رد نشه.
    voiced_f0 = f0[f0 > 0]
    if len(voiced_f0) > 20:
        sm = spsig.medfilt(voiced_f0, kernel_size=5)
        jumps = np.abs(np.diff(np.log(np.maximum(sm, 1.0))))
        jumpy = float((jumps > 0.06).mean())  # بیش از ~۱ نیم‌پرده در ۱۰ms
        if jumpy > 0.05:
            return None, ("چندصدایی/کُر تشخیص داده شد (پرش نت بین فریم‌ها) — "
                          "اصلاح نت برای جلوگیری از خراب شدن صدا رد شد")

    sp = pw.cheaptrick(x, f0, t, work_sr)
    ap = pw.d4c(x, f0, t, work_sr)

    # نزدیک‌ترین نیم‌پرده (مقیاس کروماتیک) و جابه‌جایی نرم
    voiced = f0 > 0
    safe = np.where(voiced, f0, 1.0)
    semis = np.round(12.0 * np.log2(safe / 440.0)) + 69.0
    target = 440.0 * np.power(2.0, (semis - 69.0) / 12.0)
    cents = 1200.0 * np.log2(safe / target)
    pull = np.clip(cents, -snap_cents, snap_cents) * strength
    new_f0 = np.where(voiced, f0 * np.power(2.0, pull / 1200.0), 0.0)
    new_f0 = spsig.medfilt(new_f0, kernel_size=3)

    # ⚠️ مهم: frame_period حتماً باید با تحلیل یکی باشه، وگرنه طول و سرعت پخش خراب می‌شه
    y = pw.synthesize(new_f0, sp, ap, work_sr, frame_period=frame_period).astype(np.float64)
    y = resample(y, work_sr, sr).astype(np.float32)

    n = min(len(x_mono), len(y))
    mid = x_mono[:n].astype(np.float32)
    y = y[:n]

    blended = mid * (1.0 - blend) + y * blend
    # حفظ هوای اصلی بالای کات‌آف تا جزئیات ریز از بین نره
    air = highpass(mid, sr, CROSSOVER, order=2)
    body = lowpass(blended, sr, CROSSOVER, order=2)
    return (body + air).astype(np.float32), "ok"


def autotune(data, sr=SR, strength=0.5, snap_cents=50, f0_floor=75,
             f0_ceil=800, work_sr=WORK_SR, frame_period=FRAME_PERIOD,
             blend=0.65, chunk_s=None):
    """اصلاح زیر و بمی نرم — قطعه‌قطعه (پیش‌فرض ۶۰ ثانیه، قابل تنظیم با
    متغیر محیطی AUTOTUNE_CHUNK) تا مصرف رم مستقل از طول آهنگ باشه.

    برمی‌گردونه: (خروجی یا None, دلیل یا 'ok')
    """
    try:
        import pyworld  # noqa: F401
    except ImportError:
        return None, "pyworld نصب نیست"

    if chunk_s is None:
        chunk_s = float(os.getenv("AUTOTUNE_CHUNK", "45"))

    stereo = data.ndim == 2
    if stereo:
        # ورودی صوتی اصلاً float32 هست، پس نیازی به float64 نگه‌داشتن کل فایل نیست
        mid = ((data[:, 0] + data[:, 1]) / 2.0).astype(np.float32)
        side = ((data[:, 0] - data[:, 1]) / 2.0).astype(np.float32)
    else:
        mid = data.astype(np.float32)

    if len(mid) < work_sr:  # خیلی کوتاهه
        return None, "ورودی خیلی کوتاه"

    blen = int(chunk_s * sr)
    if len(mid) <= blen:
        tuned, reason = _tune_segment(mid.astype(np.float64), sr, strength,
                                      snap_cents, f0_floor, f0_ceil, work_sr,
                                      frame_period, blend)
        if tuned is None:
            return None, reason
    else:
        # پردازش بلوکی با هم‌پوشانی ۵۰٪ و اتصال نرم هانینگ
        hop = blen // 2
        out = np.zeros(len(mid), dtype=np.float32)
        w = np.zeros(len(mid), dtype=np.float32)
        total = 0
        rejected = 0
        reason = None
        pos = 0
        while pos < len(mid):
            total += 1
            end = min(pos + blen, len(mid))
            y, reason = _tune_segment(mid[pos:end].astype(np.float64), sr,
                                      strength, snap_cents, f0_floor, f0_ceil,
                                      work_sr, frame_period, blend)
            if y is None:
                rejected += 1
                y = mid[pos:end]  # این قطعه خشک بمونه
            fade = np.hanning(end - pos).astype(np.float32)
            out[pos:end] += y * fade
            w[pos:end] += fade
            if end >= len(mid):
                break
            pos += hop
        if rejected == total:
            return None, (reason or "چندصدایی تشخیص داده شد — اصلاح نت رد شد")
        tuned = out / np.maximum(w, 1e-8)

    if stereo:
        return np.stack([tuned + side, tuned - side], axis=1).astype(np.float32), "ok"
    return tuned.astype(np.float32), "ok"
