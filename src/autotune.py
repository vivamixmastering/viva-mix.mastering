# -*- coding: utf-8 -*-
"""
autotune.py — اصلاح نت سبک ملوداین (جایگزین متن‌باز اتوتیون/ملوداین)

با pyworld (تجزیه F0 + پوش طیفی). چیزهایی که این نسخه بلده:

  ۱) تشخیص خودکار گام (Auto Key/Scale):
     ماژور، مینور، مینور هارمونیک، بایاتی ایرانی (کوارترتون) — و در صورت
     نبود تطابق، کروماتیک. دیتیون وکال به «نزدیک‌ترین نتِ گام» می‌شینه،
     نه نزدیک‌ترین نیم‌پردهٔ دلخواه → دیگه نت‌های عربی/ایرانی خراب نمی‌شن.

  ۲) حفظ ویبراتو و تحریر (مثل ملوداین):
     مسیر F0 به «محور نت» (پیچ ملایم) + «انحراف لحظه‌ای» (ویبراتو/تحریر/گلیس)
     تجزیه می‌شه. فقط محور تصحیح می‌شه؛ انحراف با ضریب vibrato_keep
     زنده می‌مونه → صدای انسانی و تازه، نه ربات صاف‌وشده.

  ۳) بدون blend دومین‌زمانی:
     نسخهٔ قدیمی ۶۵٪ سیگنال تیون‌شده + ۳۵٪ اورجینال رو جمع می‌کرد → دو موج با
     کوک متفاوت = بیitting/فیزر (همون «صدا پیچیده می‌شد»). حالا سنتز کامل
     جایگزین می‌شه و قدرت اصلاح با سقف سنت + strength کنترل می‌شه.

  ۴) چسبیدن نرم به نت (Hysteresis) + پایداری فرمانت:
     محور نت با تأخیر ۳۰ میلی‌ثانیه بین نت‌های همسایه سوییچ می‌کنه (نوسان‌ناپذیر)
     و پوش طیفی WORLD دست‌نخورده می‌مونه (فرمانت‌ها جابه‌جا نمی‌شن).

  ۵) فرکانس‌های بالای ۷.۵kHz از صدای اصلی حفظ می‌شن (هوا و تیس).

پلی‌فونی (کُر/چند نفر) خودکار تشخیص داده می‌شه و اصلاح رد می‌شه تا
صدای گروهی خراب نشه. پردازش قطعه‌قطعه — مصرف رم مستقل از طول آهنگ.

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

# ── الگوهای گام (سنت نسبت به تونیک) ──
SCALE_TEMPLATES = {
    "major":          (0, 200, 400, 500, 700, 900, 1100),
    "minor":          (0, 200, 300, 500, 700, 800, 1000),
    "harmonic_minor": (0, 200, 300, 500, 700, 800, 1100),
    # بایاتی (دستگاه ایرانی): نیم‌بمل روی درجهٔ ۲ و ۶ — کوارترتون
    "bayati":         (0, 150, 300, 500, 700, 850, 1000),
    "chromatic":      tuple(range(0, 1200, 100)),
}
SCALE_NAMES_FA = {
    "major": "ماژور", "minor": "مینور", "harmonic_minor": "مینور هارمونیک",
    "bayati": "بایاتی (ایرانی)", "chromatic": "کروماتیک",
}

# نام نت‌ها (برای گزارش گام دقیق: tonic 0..11 → نت ریشه)
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def format_key(scale_name, tonic):
    """ساخت نام دقیق گام برای گزارش تلگرام — مثلاً «E مینور هارمونیک».

    scale_name: نام گام (major/minor/harmonic_minor/bayati/chromatic)
    tonic:      نت ریشه به صورت نیم‌پردهٔ ۰..۱۱ (۰=C)
    """
    if not scale_name or scale_name == "chromatic":
        return SCALE_NAMES_FA.get("chromatic", "کروماتیک")
    note = NOTE_NAMES[int(round(tonic)) % 12]
    return f"{note} {SCALE_NAMES_FA.get(scale_name, scale_name)}"


def detect_scale(pitch_cents, voiced, energy):
    """بهترین گام و تونیک از روی هیستوگرام وزنی پیچ.

    pitch_cents: midi*100 — فقط فریم‌های واکدار.
    برمی‌گردونه: (نام گام, تونیک_midi_cents, میانگین خطای سنت)
    """
    p = pitch_cents[voiced]
    w = energy[voiced]
    if len(p) < 40:
        return "chromatic", 0.0, 50.0
    pc = np.mod(p, 1200.0)  # کلاس پیچ ۰..۱۲۰۰ سنت
    # هیستوگرام ۱۰ سانتی با وزن انرژی
    bins = np.minimum((pc / 10.0).astype(int), 119)
    hist = np.zeros(120, dtype=np.float64)
    np.add.at(hist, bins, w.astype(np.float64))
    if hist.sum() <= 0:
        return "chromatic", 0.0, 50.0
    hist /= hist.sum()
    bin_centers = np.arange(120) * 10.0 + 5.0

    def _best_of(names):
        b = ("chromatic", 0.0, 1e9)
        for name in names:
            tmpl = np.array(SCALE_TEMPLATES[name], dtype=np.float64)
            for tonic in range(12):
                allowed = np.mod(tmpl + tonic * 100.0, 1200.0)
                # فاصلهٔ دایره‌ای هر بین تا نزدیک‌ترین نت مجاز
                d = np.abs(bin_centers[:, None] - allowed[None, :])
                d = np.minimum(d, 1200.0 - d).min(axis=1)
                score = float(np.dot(hist, d))
                if score < b[2]:
                    b = (name, float(tonic), score)
        return b

    # گام‌های استاندارد (غربی) و بایاتی (ایرانی) جدا ارزیابی می‌شن؛ بایاتی فقط
    # وقتی انتخاب می‌شه که به‌طور محسوس بهتر از بهترین گام استاندارد باشه —
    # وگرنه برای موسیقی معمولی (غیر فارسی) کوارترتون اشتباه اعمال می‌شه.
    best_std = _best_of(("major", "minor", "harmonic_minor"))
    best_bay = _best_of(("bayati",))
    if best_bay[2] < best_std[2] - 4.0:  # حداقل ۴ سنت بهتر → بایاتی واقعی
        name, tonic, score = best_bay
    else:
        name, tonic, score = best_std
    # اگه بهترین گام هم به‌طور متوسط بیش از ~۳۶ سنت خطا داشت، آهنگ با هیچ گام
    # استانداردی جور نیست (مثلاً کوارترتون آزاد یا بی‌کوکی شدید) → کروماتیک امن‌تره
    if score > 36.0:
        return "chromatic", 0.0, score
    return name, tonic, score


def _snap_with_hysteresis(center_cents, allowed_abs, switch_cents=25.0,
                          hold_frames=3):
    """محور پیچ رو به نزدیک‌ترین نتِ مجاز می‌چسبونه با مکث (hysteresis):
    تا وقتی نت فعلی بهتر از بقیه‌ست عوضش نمی‌کنه؛ نت جدید باید حداقل
    switch_cents بهتر باشه و hold_frames فریم بمونه → بدون نوسان.

    خروجی: نت مطلق (در اکتاوِ خودِ محور) — نه کلاس‌پیچ ۰..۱۲۰۰."""
    n = len(center_cents)
    out = np.empty(n, dtype=np.float64)
    cur_pc = None
    better_count = 0
    cand = None
    for i in range(n):
        p = center_cents[i]
        if not np.isfinite(p):
            out[i] = p
            continue
        p_pc = np.mod(p, 1200.0)
        d = np.abs(allowed_abs - p_pc)
        d = np.minimum(d, 1200.0 - d)
        j = int(np.argmin(d))
        nearest_pc = allowed_abs[j]
        if cur_pc is None:
            cur_pc, better_count, cand = nearest_pc, 0, None
        elif nearest_pc != cur_pc:
            d_cur = abs(((p_pc - cur_pc + 600.0) % 1200.0) - 600.0)
            if (d_cur - d[j]) >= switch_cents:
                if cand == nearest_pc:
                    better_count += 1
                else:
                    cand, better_count = nearest_pc, 1
                if better_count >= hold_frames:
                    cur_pc, cand, better_count = nearest_pc, None, 0
            else:
                cand, better_count = None, 0
        # برگردوندن کلاس‌پیچ به نزدیک‌ترین اکتواب پیچ فعلی
        delta = ((cur_pc - p_pc + 600.0) % 1200.0) - 600.0
        out[i] = p + delta
    return out


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
                  work_sr, frame_period, scale="auto", vibrato_keep=0.85,
                  center_ms=160.0):
    """یک تکه تک‌صدایی رو تیون می‌کنه.
    ورودی: float64 مونو با نرخ sr — خروجی: (float32 هم‌طول با نرخ sr, 'ok')
    یا (None, دلیل رد شدن) یا ('scale', نام گام) از طریق گزارش کلی."""
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
    voiced_f0 = f0[f0 > 0]
    if len(voiced_f0) > 20:
        sm = spsig.medfilt(voiced_f0, kernel_size=5)
        jumps = np.abs(np.diff(np.log(np.maximum(sm, 1.0))))
        jumpy = float((jumps > 0.06).mean())
        if jumpy > 0.05:
            return None, ("چندصدایی/کُر تشخیص داده شد (پرش نت بین فریم‌ها) — "
                          "اصلاح نت برای جلوگیری از خراب شدن صدا رد شد")

    sp = pw.cheaptrick(x, f0, t, work_sr)
    ap = pw.d4c(x, f0, t, work_sr)

    voiced = f0 > 0
    # ── پیچ مطلق بر حسب سنت (midi×100) ──
    safe = np.where(voiced, f0, 1.0)
    pitch = 1200.0 * np.log2(safe / 440.0) + 6900.0  # A4=440 → 69×100

    # ── گام: خودکار یا اجباری از پریست ──
    seg_en = np.where(voiced, np.abs(f0), 0.0)
    if scale in (None, "", "auto"):
        scale_name, tonic, err = detect_scale(pitch, voiced, seg_en)
    else:
        scale_name, tonic, err = scale, 0.0, 0.0
    tmpl = np.array(SCALE_TEMPLATES.get(scale_name, SCALE_TEMPLATES["chromatic"]),
                    dtype=np.float64)
    allowed_abs = np.mod(tmpl + tonic * 100.0, 1200.0)

    # ── تجزیهٔ پیچ به محور (نوت) + انحراف (ویبراتو/تحریر/گلیس) ──
    # پیچ واکدار رو از روی سکوت‌ها درون‌یابی می‌کنیم تا محور تازه در سکوت‌ها
    # ننشنه و پیوستگی نت حفظ بشه
    idx = np.arange(len(pitch))
    vidx = idx[voiced]
    p_interp = np.interp(idx, vidx, pitch[voiced]) if len(vidx) else pitch
    k = max(3, int(round(center_ms / frame_period)))
    if k % 2 == 0:
        k += 1
    win = np.hanning(k + 2)
    win = win / win.sum()
    center = np.convolve(p_interp, win, mode="same")
    center[~voiced] = np.nan  # سکوت: تصحیح نمی‌شه
    deviation = pitch - center  # ویبراتو/تحریر (تا ±چندصد سنت)

    # ── چسباندن محور به گام (با hysteresis) ──
    vidx2 = idx[voiced]
    snapped = np.full(len(pitch), np.nan)
    if len(vidx2) >= 8:
        sn = _snap_with_hysteresis(center[vidx2], allowed_abs)
        snapped[vidx2] = sn

    # مقدار تصحیح: فقط تا سقف snap و با قدرت strength
    err_c = (snapped - center)                      # + یعنی بالا کشیدن
    err_c = np.clip(np.nan_to_num(err_c), -snap_cents, snap_cents) * strength

    # پیچ نهایی = محور تصحیح‌شده + انحراف زنده (ویبراتو/تحریر)
    new_pitch = np.where(voiced, center + err_c + deviation * vibrato_keep,
                         np.nan)
    new_f0 = 440.0 * np.power(2.0, (np.nan_to_num(new_pitch) - 6900.0) / 1200.0)
    new_f0[~voiced] = 0.0
    new_f0 = np.maximum(new_f0, f0_floor * 0.9)

    # ⚠️ مهم: frame_period حتماً باید با تحلیل یکی باشه، وگرنه طول و سرعت پخش خراب می‌شه
    y = pw.synthesize(new_f0.astype(np.float64), sp, ap, work_sr,
                      frame_period=frame_period).astype(np.float64)
    y = resample(y, work_sr, sr).astype(np.float32)

    n = min(len(x_mono), len(y))
    mid = x_mono[:n].astype(np.float32)
    y = y[:n]

    # بدون blend دومین‌زمانی — جابه‌جایی کوک کامل انجام شده و فیزر/بیiting نداریم
    tuned = y
    # حفظ هوای اصلی بالای کات‌آف تا جزئیات ریز از بین نره
    air = highpass(mid, sr, CROSSOVER, order=2)
    body = lowpass(tuned, sr, CROSSOVER, order=2)
    out = (body + air).astype(np.float32)
    # گارد پیک: سنتز WORLD + جمع باند هوا می‌تونه از سقف رد بشه → بدون این،
    # ذخیرهٔ PCM_16 کلیپ می‌کرد
    pk = float(np.max(np.abs(out)))
    if pk > 0.985:
        out *= np.float32(0.985 / pk)
    return out, format_key(scale_name, tonic)


def autotune(data, sr=SR, strength=0.5, snap_cents=50, f0_floor=75,
             f0_ceil=800, work_sr=WORK_SR, frame_period=FRAME_PERIOD,
             blend=None, chunk_s=None, scale="auto", vibrato_keep=0.85,
             center_ms=160.0):
    """اصلاح کوک سبک ملوداین — قطعه‌قطعه (پیش‌فرض ۴۵ ثانیه) تا مصرف رم
    مستقل از طول آهنگ باشه.

    پارامترها:
      strength     — قدرت کشیدن به نت (۰..۱)
      snap_cents   — حداکثر جابه‌جایی هر فریم (سنت)
      scale        — auto | chromatic | major | minor | harmonic_minor | bayati
      vibrato_keep — سهم ویبراتو/تحریر که زنده می‌مونه (۰.۸ پیشنهادی)
      blend        — فقط برای سازگاری با نسخهٔ قبل؛ دیگه استفاده نمی‌شه
                     (blend دومین‌زمانی باعث فیزر/بیiting می‌شد)

    برمی‌گردونه: (خروجی یا None, دلیل یا نام گام کشف‌شده)
    """
    try:
        import pyworld  # noqa: F401
    except ImportError:
        return None, "pyworld نصب نیست"

    if chunk_s is None:
        chunk_s = float(os.getenv("AUTOTUNE_CHUNK", "45"))

    data = np.asarray(data)
    if data.ndim == 2 and data.shape[1] == 1:  # (نمونه، ۱) → مونو
        data = data[:, 0]
    stereo = data.ndim == 2
    if stereo:
        mid = ((data[:, 0] + data[:, 1]) / 2.0).astype(np.float32)
        side = ((data[:, 0] - data[:, 1]) / 2.0).astype(np.float32)
    else:
        mid = data.astype(np.float32)

    if len(mid) < work_sr:  # خیلی کوتاهه
        return None, "ورودی خیلی کوتاه"

    def run(seg):
        return _tune_segment(seg.astype(np.float64), sr, strength, snap_cents,
                             f0_floor, f0_ceil, work_sr, frame_period,
                             scale=scale, vibrato_keep=vibrato_keep,
                             center_ms=center_ms)

    blen = int(chunk_s * sr)
    if len(mid) <= blen:
        tuned, info = run(mid)
        if tuned is None:
            return None, info
    else:
        # پردازش بلوکی با هم‌پوشانی ۵۰٪ و اتصال نرم هانینگ
        hop = blen // 2
        out = np.zeros(len(mid), dtype=np.float32)
        w = np.zeros(len(mid), dtype=np.float32)
        total = 0
        rejected = 0
        reason = None
        info = "ok"
        pos = 0
        while pos < len(mid):
            total += 1
            end = min(pos + blen, len(mid))
            y, info_i = run(mid[pos:end])
            if y is None:
                rejected += 1
                reason = info_i
                y = mid[pos:end]  # این قطعه خشک بمونه
                info_i = None
            if info_i and info_i != "ok":
                info = info_i  # نام گام کشف‌شده (از آخرین قطعهٔ موفق)
            fade = np.hanning(end - pos).astype(np.float32)
            out[pos:end] += y * fade
            w[pos:end] += fade
            if end >= len(mid):
                break
            pos += hop
        if rejected == total:
            return None, (reason or "چندصدایی تشخیص داده شد — اصلاح نت رد شد")
        tuned = out / np.maximum(w, 1e-8)

    # گارد پیک نهایی (مسیر تکه‌ای)
    pk = float(np.max(np.abs(tuned))) if tuned.size else 0.0
    if pk > 0.985:
        tuned *= np.float32(0.985 / pk)

    if stereo:
        return np.stack([tuned + side, tuned - side], axis=1).astype(np.float32), info
    return tuned.astype(np.float32), info
