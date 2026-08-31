# -*- coding: utf-8 -*-
"""
audio_engine.py — هسته پردازش صدا
همه توابع روی آرایه‌های numpy (float32) کار می‌کنن؛ فرمت: (نمونه، کانال)

زنجیره وکال (بر اساس زنجیره مهندس‌های معروف):
  HPF → اتوتیون نرم → ۱۱۷۶ → LA-2A → EQ → دی‌اسر → گرماساز → هوا → اکو → ریورب → نرمال‌سازی
زنجیره مستر:
  HPF → EQ اصلاحی → کمپرسور باس → EQ تُنال → پهنای استریو → اشباع → لیمیتر → LUFS
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import (
    Compressor, Delay, HighShelfFilter, HighpassFilter, Limiter,
    LowShelfFilter, LowpassFilter, PeakFilter, Reverb,
)
from scipy import signal as spsig

log = logging.getLogger("audio")

SR = 44100  # نرخ نمونه‌برداری داخلی

# ══════════════════ ابزارهای پایه ══════════════════

def db2lin(db):
    """تبدیل dB به ضریب — float32 خالص تا روی آرایه‌های بزرگ رم نخوره"""
    return np.power(np.float32(10.0), np.asarray(db, dtype=np.float32) / np.float32(20.0))

def lin2db(x, eps=1e-10):
    return np.float32(20.0) * np.log10(
        np.maximum(np.asarray(x, dtype=np.float32), np.float32(eps)))

def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

def decode_to_wav(path, out_wav):
    exe = ffmpeg_exe()
    if not exe:
        raise RuntimeError("ffmpeg پیدا نشد")
    subprocess.run(
        [exe, "-y", "-i", str(path), "-ac", "2", "-ar", str(SR),
         "-c:a", "pcm_f32le", str(out_wav)],
        check=True, capture_output=True,
    )

def encode_mp3(wav_path, mp3_path, bitrate="320k"):
    exe = ffmpeg_exe()
    if not exe:
        raise RuntimeError("ffmpeg پیدا نشد")
    subprocess.run(
        [exe, "-y", "-i", str(wav_path), "-codec:a", "libmp3lame",
         "-b:a", bitrate, str(mp3_path)],
        check=True, capture_output=True,
    )

def load_audio(path, sr=SR):
    """هر فرمتی (mp3/wav/ogg/m4a/...) → آرایه float32 استریو با نرخ SR"""
    p = Path(path)
    if p.suffix.lower() != ".wav":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        decode_to_wav(p, tmp)
        p = Path(tmp)
    data, s = sf.read(p, dtype="float32", always_2d=True)
    if s != sr:
        data = resample(data, s, sr)
    return data.astype(np.float32), sr

def save_wav(path, data, sr=SR, subtype="PCM_16"):
    sf.write(str(path), data.astype(np.float32), sr, subtype=subtype)

def resample(x, sr_from, sr_to):
    if sr_from == sr_to:
        return x.astype(np.float32)
    from fractions import Fraction
    fr = Fraction(sr_to, sr_from)
    return spsig.resample_poly(x, fr.numerator, fr.denominator, axis=0).astype(np.float32)

def to_mono(x):
    return x.mean(axis=1) if x.ndim == 2 else x

def to_stereo(x):
    if x.ndim == 1:
        return np.stack([x, x], axis=1)
    if x.ndim == 2 and x.shape[1] == 1:  # (نمونه، ۱) → استریوی واقعی
        return np.repeat(x, 2, axis=1)
    return x


def highpass(x, sr, freq, order=2):
    """فیلتر بالاگذر با pedalboard (موتور float32) — برخلاف scipy.sosfiltfilt
    هیچ کپی float64 و padding بزرگ نمی‌سازه، پس مصرف رم خیلی کمه."""
    return HighpassFilter(cutoff_frequency_hz=freq)(
        np.ascontiguousarray(x, dtype=np.float32), sr)


def lowpass(x, sr, freq, order=2):
    return LowpassFilter(cutoff_frequency_hz=freq)(
        np.ascontiguousarray(x, dtype=np.float32), sr)


# ══════════════════ دنبال‌کننده دامنه ══════════════════

def env_follow(x, sr, attack_ms=10.0, release_ms=120.0):
    """پوش دامنه با max-pool سبک (نرخ ~۳۴۴Hz) — مصرف رم ناچیز و مستقل از طول"""
    a = np.abs(x.mean(axis=1) if x.ndim == 2 else x).astype(np.float32)
    ds = 128
    n = len(a)
    pad = (-n) % ds
    if pad:
        a = np.pad(a, (0, pad))
    b = a.reshape(-1, ds).max(axis=1)
    sr_low = sr / ds
    rt_att = float(np.exp(-1000.0 / (attack_ms * sr_low)))
    rt_rel = float(np.exp(-1000.0 / (release_ms * sr_low)))
    e1 = spsig.lfilter([1.0 - rt_att], [1.0, -rt_att], b.astype(np.float64))
    e2 = spsig.lfilter([1.0 - rt_rel], [1.0, -rt_rel], b.astype(np.float64))
    e = np.maximum(e1, e2).astype(np.float32)
    return np.repeat(e, ds)[:n]

# ══════════════════ دی‌اسر (کنترل سوت «س») ══════════════════

def _band_deess(x, sr, lo_hz, hi_hz, threshold_db, ratio,
                attack_ms, release_ms):
    """هستهٔ دی‌اسر باندی: باند lo..hi رو جدا می‌کنه و فقط همون باند رو
    بر اساس انرژی خودش فشرده می‌کنه (بدون دست‌زدن به بقیهٔ طیف)."""
    hp = highpass(x, sr, lo_hz, order=4)
    band = lowpass(hp, sr, hi_hz, order=4)
    rest = x - band
    env = env_follow(band, sr, attack_ms, release_ms)
    over = np.maximum(lin2db(env) - np.float32(threshold_db), 0.0)
    gr = over * (1.0 - 1.0 / ratio)
    gain = db2lin(-gr).astype(np.float32)
    if gain.ndim == 1 and x.ndim == 2:
        gain = gain[:, None]
    return (rest + band * gain).astype(np.float32)


def deesser(x, sr, freq=6200.0, threshold_db=-24.0, ratio=5.0,
            attack_ms=1.5, release_ms=35.0, makeup_db=0.0):
    """دی‌اسر (کنترل سوت «س/ش») — حملهٔ سریع، روی باند بالای freq."""
    hp = highpass(x, sr, freq, order=4)
    lo = x - hp
    env = env_follow(hp, sr, attack_ms, release_ms)
    env_db = lin2db(env)
    over = np.maximum(env_db - threshold_db, 0.0)
    gr = over * (1.0 - 1.0 / ratio)
    gain = db2lin(-gr).astype(np.float32)
    if gain.ndim == 1 and x.ndim == 2:
        gain = gain[:, None]
    out = lo + hp * gain
    if makeup_db:
        out = out * np.float32(db2lin(makeup_db))
    return out.astype(np.float32)


def de_harshness(x, sr, freq=3500.0, threshold_db=-21.0, ratio=3.5,
                 attack_ms=3.0, release_ms=60.0):
    """مهار سخت‌خوانی «ش/خ/ج» و لبه‌های تیز ۲.۵–۵kHz که صدا رو «نیش‌دار» می‌کنن.
    دی‌اسرِ باند میانی — حروف صدادار تیز رو نرم می‌کنه بدون خفه کردن درخشش."""
    return _band_deess(x, sr, freq * 0.6, freq * 1.6, threshold_db, ratio,
                       attack_ms, release_ms)


def transient_tame(x, sr, threshold_db=-14.0, ratio=4.0,
                   attack_ms=0.3, release_ms=25.0):
    """مهار پاپِ صامت‌های انفجاری (پ/ب/ت/ک/د) با یک کمپرسور فوق‌سریع.
    attack خیلی کوتاه فقط لبهٔ تیز ترنزینت رو می‌گیره؛ بدنهٔ واکه
    دست‌نخورده می‌مونه → «پ/ک/ت/د» دیگه بیرون نمی‌زنه بدون اینکه صدا خفه بشه."""
    return np.asarray(
        Compressor(threshold_db=threshold_db, ratio=ratio,
                   attack_ms=attack_ms, release_ms=release_ms)(x, sr),
        dtype=np.float32)

# ══════════════════ گرماساز / اشباع ══════════════════

def _upsample_zerostuff(x, factor):
    """آپ‌سمپلینگ با صفرگذاری — هیچ کپی float64 و فیلتر سنگینی نمی‌سازه
    (ضد الیاس لازم رو فیلتر بعدی انجام می‌ده)"""
    shape = (len(x) * factor,) + x.shape[1:]
    out = np.zeros(shape, dtype=np.float32)
    out[::factor] = x
    return out


def saturation(x, sr, drive_db=3.0, mix=0.3, asym=1.15):
    """اشباع نرم لامپی — هارمونیک‌های زوج + فرد
    اورسمپلینگ ۲× با صفرگذاری + ضد الیاس ۱۹kHz (بدون الیاسینگ/هیس و کم‌مصرف)"""
    single = x.ndim == 1
    if single:
        x = x[:, None]
    up = _upsample_zerostuff(x, 2)
    g = float(db2lin(drive_db))
    pos = np.tanh(g * up)
    neg = np.tanh(g * asym * up)
    wet = np.where(up >= 0, pos, neg) / np.tanh(g)
    wet = lowpass(wet, sr * 2, 19000.0)
    wet = wet[::2]
    n = min(len(x), len(wet))
    out = (1.0 - mix) * x[:n] + mix * wet[:n]
    return (out[:, 0] if single else out).astype(np.float32)

# ══════════════════ هوا / جزئیات ریز تیس (Air Exciter) ══════════════════

def air_exciter(x, sr, freq=10000.0, drive_db=2.0, mix=0.3):
    """جزئیات ریز و درخشش بالای وکال رو بیرون می‌کشه (شبیه باند Air در Maag EQ4)

    ⚠️ نسخه اول بدون اورسمپلینگ الیاسینگ تولید می‌کرد (صدای «برفکی»).
    حالا: اورسمپلینگ ۴× با صفرگذاری + ضد الیاس ۱۹kHz → درخشش تمیز،
    بی‌هیس و کم‌مصرف (همه float32)."""
    single = x.ndim == 1
    if single:
        x = x[:, None]
    lp = lowpass(x, sr, freq)
    hp = x - lp
    up = _upsample_zerostuff(hp, 4)
    g = float(db2lin(drive_db))
    sat = np.tanh(g * up) / np.tanh(g)
    wet = (1.0 - mix) * up + mix * sat
    wet = lowpass(wet, sr * 4, 19000.0)
    wet = wet[::4]
    n = min(len(x), len(wet))
    out = lp[:n] + wet[:n]
    return (out[:, 0] if single else out).astype(np.float32)


# ══════════════════ ابریشمی‌کردن وکال (Silk) ══════════════════
def silk(x, sr, freq=6000.0, drive_db=1.5, mix=0.4):
    """ابریشمی‌کردن وکال — نرم و براق بدون تیزی (شبیه کنترل Silk در Neve).

    باند بالای freq جدا می‌شه؛ روی اون اشباع زوج‌هارمونیکِ ملایم (شینِ نرم)
    + فشرده‌سازی نرم می‌آد تا لبه‌های تیزِ تیس صاف و «ابریشمی» بشن — بدون
    این‌که درخشش کم بشه. خروجی با سیگنال اصلی بلند می‌شه (میزان با mix).
    """
    from pedalboard import Compressor
    single = x.ndim == 1
    if single:
        x = x[:, None]
    lp = lowpass(x, sr, freq)
    hp = x - lp

    # اشباع زوج‌هارمونیک ملایم روی باند بالا (اورسمپلینگ ۴× — بدون الیاس)
    up = _upsample_zerostuff(hp, 4)
    g = float(db2lin(drive_db))
    sat = np.tanh(g * up) / np.tanh(g)
    sat = lowpass(sat, sr * 4, 19000.0)[::4]
    n = len(hp)
    sat = sat[:n]

    # فشرده‌سازی نرم روی باند بالا — صاف‌کردن لبه‌های تیز
    hp_sm = Compressor(threshold_db=-28.0, ratio=2.0, attack_ms=3.0,
                       release_ms=150.0)(sat, sr)

    hp_out = (1.0 - mix) * hp + mix * hp_sm
    out = lp + hp_out
    return (out[:, 0] if single else out).astype(np.float32)


# ══════════════════ پهنای استریو (Mid/Side) ══════════════════

def width_ms(x, amount=1.1):
    if x.ndim == 1:
        return x
    mid = (x[:, 0] + x[:, 1]) / 2.0
    side = (x[:, 0] - x[:, 1]) / 2.0 * amount
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)

# ══════════════════ Ducking (جا باز کردن برای وکال) ══════════════════

def noise_gate(x, sr, threshold_db=-50.0, ratio=3.0,
               attack_ms=3.0, release_ms=90.0):
    """گیت نرم — نویز و هیس سکوت‌ها (بین کلمات) رو می‌بنده بدون آسیب به آواز"""
    e = env_follow(x, sr, attack_ms, release_ms)
    eb = lin2db(e)
    below = threshold_db - eb
    gr = np.maximum(below, np.float32(0.0)) * (ratio - 1.0) / ratio
    gain = db2lin(-gr)
    if x.ndim == 2:
        gain = gain[:, None]
    return (x * gain).astype(np.float32)


def parallel_compression(x, sr, threshold_db=-28.0, ratio=4.0,
                         attack_ms=0.5, release_ms=150.0, mix=0.25):
    """کمپرسور موازی واقعی سبک NY — حجم، چگالی و پایداری وکال بدون له شدن داینامیک

    ⚠️ نسخهٔ قدیدی: نسخهٔ لهیده بدون جبران دامنه، خیلی آرام‌تر از سیگنال خشک
    بود و عملاً چیزی اضافه نمی‌کرد. نسخهٔ NY درست: له‌شدگی شدید + برگردوندن
    دامنهٔ wet تا سطح dry (تطبیق RMS) → چگالی و گرمای واقعی با mix کم."""
    hard = Compressor(threshold_db=threshold_db, ratio=ratio,
                      attack_ms=attack_ms, release_ms=release_ms)(x, sr)
    hard = np.asarray(hard, dtype=np.float32)
    # تطبیق RMS: wet رو به بلندی dry برگردون (و ~۰.۵dB بیشتر — جوهر NY)
    r_dry = float(np.sqrt(np.mean(np.square(x))) + 1e-12)
    r_wet = float(np.sqrt(np.mean(np.square(hard))) + 1e-12)
    if r_wet > 1e-9:
        hard = hard * np.float32(min((r_dry / r_wet) * db2lin(0.5), 12.0))
    return ((1.0 - mix) * x + mix * hard).astype(np.float32)


def calibrate_lufs(x, sr, target=-20.0, max_gain_db=15.0):
    """آوردن ورودی به سطح کاری استاندارد قبل از کمپرسورها.

    بدون این، آستانهٔ کمپرسورها به سطح ضبط وابسته بود: وکال بلند = ۱۵+dB
    گین‌ریداکشن پشت سر هم = صدای «مچاله». حالا رفتار همهٔ پریست‌ها روی
    هر فایلی، هر لوڈی، یکسان و قابل پیش‌بینیه."""
    l = integrated_lufs(x, sr)
    if l <= -69.0:
        return x.astype(np.float32)
    gain = float(np.clip(target - l, -max_gain_db, max_gain_db))
    return (x * np.float32(db2lin(gain))).astype(np.float32)


def leveler(x, sr, target_rms_db=-18.0, max_gain_db=3.0,
            attack_ms=300.0, release_ms=1500.0):
    """لولر نرم (کمپرسور اپتیکال خیلی کُند) — یکدستی بلندی بین جمله‌ها
    بدون پامپینگ؛ حداکثر ±max_gain_db جابه‌جایی می‌ده."""
    e = env_follow(x, sr, attack_ms, release_ms)
    e_db = lin2db(np.maximum(e, np.float32(1e-6)))
    g_db = np.clip(np.float32(target_rms_db) - e_db,
                   np.float32(-max_gain_db), np.float32(max_gain_db))
    gain = db2lin(g_db).astype(np.float32)
    if x.ndim == 2:
        gain = gain[:, None]
    return (x * gain).astype(np.float32)


def soft_clip(x, ceiling_db=-2.2, knee_div=2.0):
    """کلیپر نرم با زانوی C1-پیوسته — پیک‌های تیز رو بی‌صدا گرد می‌کنه.

    زیر نیمی از سقف سیگنال دست‌نخورده رد می‌شه؛ بین زانو و سقف با tanh
    نرم می‌شینه (بدون پرش، بدون کلیک). تکنیک استاندارد قبل از لیمیتر
    برای بلندیِ بالا بدون لهیدگی ترنزینت‌ها."""
    single = x.ndim == 1
    if single:
        x = x[:, None]
    c = float(db2lin(ceiling_db))
    t = c / knee_div
    a = np.abs(x)
    over = a > t
    y = x.copy()
    if over.any():
        knee = t + (c - t) * np.tanh((a - t) / (c - t))
        y = np.where(over, np.sign(x) * knee, y)
    return (y[:, 0] if single else y).astype(np.float32)



def block_apply(func, x, sr, block_s=60.0, overlap_s=1.0):
    """مرحله سنگین رو روی بلوک‌های هم‌پوشان اجرا می‌کنه تا مصرف رم
    مستقل از طول آهنگ بمونه (اتصال نرم هانینگ — بدون کلیک).
    func باید سیگنالی هم‌طول ورودی برگردونه."""
    single = x.ndim == 1
    if single:
        x = x[:, None]
    n = len(x)
    blen = int(block_s * sr)
    ov = int(overlap_s * sr)
    if n <= blen:
        y = func(x)
        return y[:, 0] if single else y
    out = np.zeros_like(x, dtype=np.float32)
    w = np.zeros(n, dtype=np.float32)
    pos = 0
    while pos < n:
        end = min(pos + blen, n)
        y = func(x[pos:end]).astype(np.float32)
        fade = np.hanning(end - pos).astype(np.float32)[:, None]
        out[pos:end] += y * fade
        w[pos:end] += fade[:, 0]
        if end >= n:
            break
        pos += blen - ov
    w = np.maximum(w, 1e-8)[:, None]
    y = out / w
    return y[:, 0] if single else y


def duck_under_vocal(inst, vocal, sr, depth_db=2.0, attack_ms=15.0, release_ms=140.0):
    env = env_follow(to_mono(vocal), sr, attack_ms, release_ms)
    norm = env / (np.percentile(env, 95) + 1e-9)
    gr = depth_db * np.clip(norm, 0.0, 1.0)
    gain = db2lin(-gr).astype(np.float32)
    if inst.ndim == 2:
        gain = gain[:, None]
    return (inst * gain).astype(np.float32)

# ══════════════════ بلندی صدا (LUFS) ══════════════════

_meters = {}

def integrated_lufs(x, sr):
    if sr not in _meters:
        _meters[sr] = pyln.Meter(sr)
    try:
        l = _meters[sr].integrated_loudness(x)
    except Exception:
        l = float("-inf")
    if not np.isfinite(l):
        l = -70.0
    return float(l)

def normalize_lufs(x, sr, target=-14.0, ceiling_db=-1.0, max_boost_db=12.0):
    l = integrated_lufs(x, sr)
    if l <= -70.0:
        return x.astype(np.float32)
    gain = min(target - l, max_boost_db)
    y = x * np.float32(db2lin(gain))
    peak_db = lin2db(np.max(np.abs(y)))
    over = peak_db - ceiling_db
    if over > 0:
        y = y * np.float32(db2lin(-over))
    return y.astype(np.float32)

# ══════════════════ ترمیم استریو + دابل‌ترک وکال ══════════════════

def _stronger_channel(x):
    """کانال قوی‌تر (انرژی بیشتر) — برای ترمیم استریوی ناقص وکال جدا‌شده."""
    if x.ndim != 2 or x.shape[1] < 2:
        return to_mono(x)
    L = x[:, 0]
    R = x[:, 1]
    eL = float(np.dot(L, L))
    eR = float(np.dot(R, R))
    return L if eL >= eR else R


def stereo_repair(x, sr):
    """تعمیر استریوی ناقص (وکالی که از بیت جدا شده و یک کانالش ضعیفه).

    کانال قوی‌تر انتخاب می‌شه و به‌عنوان هستهٔ مونوی متعادل روی هر دو کانال
    می‌نشینه (بدون جابه‌جایی زمانی — دقیق و ژوست). عرضِ واقعی بعداً توسط
    لایهٔ بک‌ویس (دابل) ساخته می‌شه.
    """
    core = _stronger_channel(x)
    return np.stack([core, core], axis=1).astype(np.float32)


def _double_layer(core, sr, delay_ms=18.0, depth_ms=6.0, rate_hz=0.6):
    """بک‌ویس (دابل‌ترک): کپی با تأخیر مدوله‌شده (کورس‌مانند) → دی‌تیون طبیعی
    و حرکت، بدون کلیک/الیاس و بدون فریزِ پایان.

    پیاده‌سازی کم‌مصرف: float32 + تاخیر کسری با درون‌یابی خطی (دو شیفت
    برداری)، بدون آرایه‌های float64 بزرگ → روی فایل‌های بلند OOM نمی‌شه.
    """
    x = core.astype(np.float32)
    n = len(x)
    # مدولاسیون تاخیر: مقدار در گره‌ها محاسبه و بین گره‌ها درون‌یابی خطی می‌شه
    # (پیوسته → بدون کلیک/تیک). قبلاً با np.repeat پله‌ای بود و هر 0.25s
    # یک پرش ناگهانی تاخیر = یک «تیک» ریز (۴ تیک در ثانیه) درست می‌کرد.
    chunk = max(1, int(sr * 0.25))
    nseg = (n + chunk - 1) // chunk
    node_pos = (np.arange(nseg, dtype=np.float32) * chunk)
    node_t = node_pos / sr
    seg_d = (delay_ms / 1000.0 + (depth_ms / 1000.0)
             * np.sin(2.0 * np.pi * rate_hz * node_t)).astype(np.float32) * sr
    seg_d = np.clip(seg_d, 2.0, float(n - 2))
    # درون‌یابی خطی بین گره‌ها → تاخیر پیوسته به ازای هر نمونه (همان طول n)
    d = np.interp(np.arange(n, dtype=np.float32), node_pos, seg_d).astype(np.float32)
    d0 = np.floor(d).astype(np.int32)
    frac = (d - d0).astype(np.float32)
    d0 = np.clip(d0, 1, n - 2)
    i0 = np.arange(n, dtype=np.int32)
    a = x[np.clip(i0 - d0, 0, n - 1)]
    b = x[np.clip(i0 - d0 - 1, 0, n - 1)]
    return ((1.0 - frac) * a + frac * b).astype(np.float32)


def add_double_layer(y, sr, cfg):
    """افزودن لایهٔ بک‌ویس به وکال → استریو واقعی + حجم و گرما.

    وکال اصلی در مرکز می‌مونه؛ بک‌ویس (دابل) با ریورب/گین سبک (۳۰٪) روی
    پهلوها سوار می‌شه. خروجی: L = mid + back، R = mid − back (پهنای واقعی).
    """
    from pedalboard import Reverb

    mid = to_mono(y)
    back = _double_layer(mid, sr,
                         delay_ms=cfg.get("delay_ms", 18.0),
                         depth_ms=cfg.get("depth_ms", 6.0),
                         rate_hz=cfg.get("rate_hz", 0.6))
    back = np.stack([back, back], axis=1)
    if cfg.get("reverb_wet"):
        back = Reverb(room_size=cfg.get("room", 0.45),
                      damping=cfg.get("damping", 0.5),
                      wet_level=cfg["reverb_wet"], dry_level=1.0,
                      width=1.0)(back, sr)
    back = to_mono(back) * np.float32(cfg.get("back_gain", 0.3))
    mix = cfg.get("mix", 0.35)
    L = mid + back * mix
    R = mid - back * mix
    out = np.stack([L, R], axis=1)
    pk = float(np.max(np.abs(out)))
    if pk > 0.985:
        out = out * np.float32(0.985 / pk)
    return out.astype(np.float32)


# ══════════════════ زنجیره وکال ══════════════════

def vocal_chain(x, sr, v):
    """زنجیره کامل وکال بر اساس تنظیمات پریست → (سیگنال, گزارش مراحل)"""
    rep = []
    y = to_stereo(x).astype(np.float32)

    # ── ترمیم استریو: وکال جدا‌شده یک کانالش ضعیفه → کانال قوی مبنای هر دو ──
    if v.get("stereo_repair"):
        y = stereo_repair(y, sr)
        rep.append("ترمیم استریو — انتخاب کانال قوی‌تر و مرکزیت واقعی")

    # ── حالت وکال جداسازی‌شده (نشت‌دار) ──
    # وکالی که خود کاربر با ابزار جداسازی گرفته، معمولاً کمی از موزیک هم
    # باهاش آمده. زنجیرهٔ عادی این نشت رو بزرگ می‌کنه (ریورب/اکو/هوا بهش
    # عمق می‌دن). این حالت: گیت محکم‌تر، HPF بالاتر و افکت‌های فضاساز کمتر
    if v.get("bleed_safe"):
        v = {k: vv for k, vv in v.items() if k != "bleed_safe"}
        v["hpf_hz"] = max(int(v.get("hpf_hz") or 0), 110)
        v["gate"] = {"threshold_db": -30.0, "ratio": 6.0}
        if v.get("delay"):
            v["delay"] = {**v["delay"], "mix": v["delay"].get("mix", 0.1) * 0.5}
        if v.get("reverb"):
            v["reverb"] = {**v["reverb"], "wet": v["reverb"].get("wet", 0.12) * 0.6}
        if v.get("air"):
            v["air"] = {**v["air"], "mix": v["air"].get("mix", 0.3) * 0.7}
        if v.get("eq_gloss"):
            v["eq_gloss"] = {**v["eq_gloss"],
                             "gain_db": v["eq_gloss"].get("gain_db", 2.0) * 0.75}
        rep.append("🧹 پاکسازی نشت موزیک (حالت وکال جداسازی‌شده)")

    hpf = v.get("hpf_hz")
    if hpf:
        y = HighpassFilter(cutoff_frequency_hz=int(hpf))(y, sr)
        rep.append(f"حذف فرکانس‌های پایین (HPF {int(hpf)}Hz)")

    # کالیبراسیون به سطح کاری — رفتار کمپرسورها مستقل از بلندی ضبط
    wl = v.get("working_lufs")
    if wl is not None:
        y = calibrate_lufs(y, sr, target=float(wl))
        rep.append(f"کالیبراسیون سطح ورودی به {wl:g} LUFS")

    if v.get("gate", True):
        g = v["gate"] if isinstance(v.get("gate"), dict) else {}
        y = noise_gate(y, sr,
                       threshold_db=g.get("threshold_db", -50.0),
                       ratio=g.get("ratio", 3.0))
        rep.append("گیت نرم — حذف نویز و هیس سکوت‌ها")

    # مهار پاپ صامت‌های انفجاری (پ/ب/ت/ک/د) — قبل از کمپرسورها تا ترنزینت‌ها
    # بزرگ‌نمایی نشن
    depl = v.get("deplosive")
    if depl is not False:
        dp = depl if isinstance(depl, dict) else {}
        y = transient_tame(y, sr,
                           threshold_db=dp.get("threshold_db", -14.0),
                           ratio=dp.get("ratio", 4.0))
        rep.append("مهار پاپ صامت‌های انفجاری (پ/ب/ت/ک/د)")

    tune = v.get("tune") or {}
    if tune.get("strength", 0) > 0:
        try:
            from src.autotune import autotune
            yt, reason = autotune(
                y, sr,
                strength=tune["strength"],
                snap_cents=tune.get("snap", 50),
                scale=tune.get("scale", "chromatic"),
                vibrato_keep=float(tune.get("vibrato_keep", 0.85)))
            if yt is not None:
                y = to_stereo(yt)
                gam = reason if reason and reason != "ok" else ""
                extra = f" • گام: {gam}" if gam else ""
                rep.append(f"اصلاح نت سبک ملوداین {int(tune['strength'] * 100)}٪"
                           f" (سقف {tune.get('snap', 50)} سنت، تحریر حفظ)"
                           f"{extra}")
            else:
                rep.append(f"اصلاح نت: {reason}")
        except Exception as e:
            log.warning("autotune failed: %s", e)
            rep.append("اصلاح نت: خطا — رد شد")

    lev = v.get("leveler")
    if lev:
        y = leveler(y, sr,
                    target_rms_db=lev.get("target_db", -18.0),
                    max_gain_db=lev.get("max_db", 3.0))
        rep.append("لولر نرم — یکدستی بلندی بین جمله‌ها (بدون پامپینگ)")


    c1 = v.get("comp1")
    if c1:
        y = Compressor(threshold_db=c1["threshold_db"], ratio=c1["ratio"],
                       attack_ms=c1["attack_ms"], release_ms=c1["release_ms"])(y, sr)
        rep.append(f"کمپرسور سریع سبک ۱۱۷۶ ({c1['ratio']}:۱) — کنترل پیک‌ها")
    c2 = v.get("comp2")
    if c2:
        y = Compressor(threshold_db=c2["threshold_db"], ratio=c2["ratio"],
                       attack_ms=c2["attack_ms"], release_ms=c2["release_ms"])(y, sr)
        rep.append(f"کمپرسور نرم سبک LA-2A ({c2['ratio']}:۱) — حجم و یکدستی")

    par = v.get("parallel")
    if par:
        y = parallel_compression(y, sr, par.get("threshold_db", -28),
                                 par.get("ratio", 4), par.get("mix", 0.25))
        rep.append(f"کمپرسور موازی سبک NY — حجم و چگالی صدا "
                   f"({int(par.get('mix', 0.25) * 100)}٪)")

    ls = v.get("eq_low_shelf")
    if ls:
        y = LowShelfFilter(cutoff_frequency_hz=ls["freq"],
                           gain_db=ls["gain_db"], q=0.7)(y, sr)
        rep.append(f"گرمی بم (شلف {ls['gain_db']:+g}dB @ {ls['freq']}Hz)")
    wm = v.get("eq_warm")
    if wm:
        y = PeakFilter(cutoff_frequency_hz=wm["freq"], gain_db=wm["gain_db"],
                       q=wm.get("q", 0.8))(y, sr)
        rep.append(f"گرمای بدنه و سینه ({wm['gain_db']:+g}dB @ {wm['freq']}Hz)")
    dip = v.get("eq_dip")
    if dip:
        y = PeakFilter(cutoff_frequency_hz=dip["freq"], gain_db=dip["gain_db"],
                       q=dip.get("q", 1.4))(y, sr)
        rep.append(f"پاکسازی میدرنج ({dip['gain_db']:+g}dB @ {dip['freq']}Hz)")
    pr = v.get("eq_presence")
    if pr:
        y = PeakFilter(cutoff_frequency_hz=pr["freq"], gain_db=pr["gain_db"],
                       q=pr.get("q", 1.0))(y, sr)
        rep.append(f"حضور صدا ({pr['gain_db']:+g}dB @ {pr['freq']}Hz)")
    gl = v.get("eq_gloss")
    if gl:
        y = PeakFilter(cutoff_frequency_hz=gl["freq"], gain_db=gl["gain_db"],
                       q=gl.get("q", 0.9))(y, sr)
        rep.append(f"جلا و براقیت ({gl['gain_db']:+g}dB @ {gl['freq']}Hz)")
    airs = v.get("eq_air_shelf")
    if airs:
        y = HighShelfFilter(cutoff_frequency_hz=airs["freq"],
                            gain_db=airs["gain_db"])(y, sr)
        rep.append(f"درخشش بالا ({airs['gain_db']:+g}dB @ {airs['freq']}Hz)")

    de = v.get("deess")
    if de:
        y = block_apply(
            lambda blk: deesser(blk, sr, de["freq"], de["threshold_db"],
                                de["ratio"], makeup_db=de.get("makeup_db", 0.0)),
            y, sr)
        rep.append(f"دی‌اسر (کنترل سوت «س») @ {de['freq']}Hz")

    w = v.get("warmth")
    if w:
        y = block_apply(
            lambda blk: saturation(blk, sr, w["drive_db"], w["mix"]), y, sr,
            block_s=30.0)
        rep.append(f"گرم‌سازی صدا (اشباع لامپی {w['drive_db']}dB)")
    ai = v.get("air")
    if ai:
        y = block_apply(
            lambda blk: air_exciter(blk, sr, ai["freq"], ai["drive_db"], ai["mix"]),
            y, sr, block_s=15.0)
        rep.append(f"هوا و جزئیات ریز تیس (Air Exciter @ {ai['freq']}Hz)")

    # ابریشمی‌کردن — نرم و براق کردن ناحیهٔ بالا (بدون تیزی)
    sk = v.get("silk")
    if sk:
        y = block_apply(
            lambda blk: silk(blk, sr, sk.get("freq", 6000.0),
                             sk.get("drive_db", 1.5), sk.get("mix", 0.4)),
            y, sr, block_s=15.0)
        rep.append("ابریشمی‌کردن وکال (Silk — نرمی و شین بالا)")

    # نرم‌کردن سخت‌خوانی بعد از هوا/اشباع — لبه‌های تیز ۲.۵–۵kHz که از
    # اشباع و exciter درست می‌شن (حروف «ش/خ/ج» و حالت توییتری) اینجا مهار می‌شن
    hsh = v.get("harshness")
    if hsh is not False:
        hs = hsh if isinstance(hsh, dict) else {}
        y = block_apply(
            lambda blk: de_harshness(blk, sr, hs.get("freq", 3500.0),
                                     hs.get("threshold_db", -21.0),
                                     hs.get("ratio", 3.5)),
            y, sr, block_s=30.0)
        rep.append("نرم‌کردن سخت‌خوانی (ش/خ/ج) — بدون حالت توییتری")

    d = v.get("delay")
    if d:
        y = Delay(delay_seconds=d["time_s"], feedback=d["feedback"],
                  mix=d["mix"])(y, sr)
        rep.append(f"اکو (Delay {d['time_s']}s)")
    rv = v.get("reverb")
    if rv:
        y = Reverb(room_size=rv["room"], damping=rv["damping"],
                   wet_level=rv["wet"], dry_level=1.0, width=1.0)(y, sr)
        rep.append(f"ریورب (فضاسازی {int(rv['room'] * 100)}٪)")

    target = v.get("out_lufs", -18.0)
    y = normalize_lufs(y, sr, target=target, ceiling_db=-3.0)
    rep.append(f"نرمال‌سازی وکال → {target} LUFS")
    # گین صریح خروجی (کنترل قدرت/بلندی مستقیم، جدا از نرمال‌سازی LUFS)
    g = v.get("gain_db")
    if g:
        y = y * np.float32(db2lin(g))
        pk = float(np.max(np.abs(y)))
        if pk > 0.98:                      # گارد پیک — بدون دیستورت
            y = y * np.float32(0.98 / pk)
        rep.append(f"گین خروجی +{g:g}dB (قدرت بیشتر)")

    # ── دابل‌ترک (بک‌ویس) — استریو واقعی + حجم و گرما (اختیاری) ──
    db = v.get("double")
    if db and db.get("enabled"):
        y = add_double_layer(y, sr, db)
        rep.append("بک‌ویس (دابل‌ترک) — استریو واقعی و حجم/گرمای بیشتر")

    return y.astype(np.float32), rep


# ══════════════════ هارمونی (جابه‌جایی زیروبمی) ══════════════════

def harmonize(x, sr, semitones):
    """جابه‌جایی زیروبمی (هارمونی) بدون تغییر طول — با موتور Rubber Band
    (کیفیت بالا، بدون آرتیفکت «چیپمونک»). semitones مثبت = زیرتر (بالا).

    خروجی: استریو float32 هم‌طول ورودی.
    """
    from pedalboard import PitchShift
    y = to_stereo(x).astype(np.float32)
    return np.asarray(PitchShift(semitones=float(semitones))(y, sr),
                      dtype=np.float32)


# ══════════════════ زنجیره مستر ══════════════════

def master_chain(x, sr, m):
    """زنجیره مسترینگ بر اساس تنظیمات پریست → (سیگنال, گزارش مراحل)"""
    rep = []
    y = to_stereo(x).astype(np.float32)

    # کالیبراسیون میکس به سطح کاری — کمپرسور باس روی هر آهنگ یکسان عمل کنه
    wl = m.get("working_lufs")
    if wl is not None:
        y = calibrate_lufs(y, sr, target=float(wl))
        rep.append(f"کالیبراسیون میکس به {wl:g} LUFS")

    hpf = m.get("hpf_hz")
    if hpf:
        y = HighpassFilter(cutoff_frequency_hz=int(hpf))(y, sr)
        rep.append(f"پاکسازی ساب (زیر {int(hpf)}Hz)")

    dip = m.get("eq_dip")
    if dip:
        y = PeakFilter(cutoff_frequency_hz=dip["freq"], gain_db=dip["gain_db"],
                       q=dip.get("q", 1.2))(y, sr)
        rep.append(f"اصلاح میدرنج مستر ({dip['gain_db']:+g}dB @ {dip['freq']}Hz)")

    bc = m.get("bus_comp")
    if bc:
        y = Compressor(threshold_db=bc["threshold_db"], ratio=bc["ratio"],
                       attack_ms=bc["attack_ms"], release_ms=bc["release_ms"])(y, sr)
        rep.append(f"کمپرسور باس ({bc['ratio']}:۱) — چسبندگی کل آهنگ")

    # کمپرسور موازی باس مستر (NY) — پانچ و ضخامت بدون له شدن ترنزینت‌ها
    bp = m.get("bus_parallel")
    if bp:
        y = parallel_compression(y, sr,
                                 bp.get("threshold_db", -30.0),
                                 bp.get("ratio", 2.0),
                                 bp.get("mix", 0.2))
        rep.append(f"کمپرسور موازی مستر — پانچ و ضخامت "
                   f"({int(bp.get('mix', 0.2) * 100)}٪)")

    ls = m.get("eq_low_shelf")
    if ls:
        y = LowShelfFilter(cutoff_frequency_hz=ls["freq"],
                           gain_db=ls["gain_db"], q=0.7)(y, sr)
        rep.append(f"گرمی بم کل ({ls['gain_db']:+g}dB @ {ls['freq']}Hz)")
    hs = m.get("eq_high_shelf")
    if hs:
        y = HighShelfFilter(cutoff_frequency_hz=hs["freq"],
                            gain_db=hs["gain_db"])(y, sr)
        rep.append(f"درخشش کل ({hs['gain_db']:+g}dB @ {hs['freq']}Hz)")

    gl = m.get("eq_gloss")
    if gl:
        y = PeakFilter(cutoff_frequency_hz=gl["freq"], gain_db=gl["gain_db"],
                       q=gl.get("q", 0.9))(y, sr)
        rep.append(f"جلا و براقیت کل ({gl['gain_db']:+g}dB @ {gl['freq']}Hz)")

    sk = m.get("silk")
    if sk:
        y = block_apply(
            lambda blk: silk(blk, sr, sk.get("freq", 6000.0),
                             sk.get("drive_db", 1.5), sk.get("mix", 0.35)),
            y, sr, block_s=15.0)
        rep.append("ابریشمی‌کردن کل (Silk)")

    w = m.get("width", 1.0)
    if abs(w - 1.0) > 1e-4:
        y = width_ms(y, w)
        rep.append(f"پهنای استریو ({int(w * 100)}٪)")

    sat = m.get("sat")
    if sat:
        y = block_apply(
            lambda blk: saturation(blk, sr, sat["drive_db"], sat["mix"]), y, sr,
            block_s=30.0)
        rep.append(f"اشباع نهایی (گرمای کلی {sat['drive_db']}dB)")

    ceiling = m.get("ceiling", -1.0)
    target = m.get("lufs", -12.0)

    # ── بلندیِ بالا بدون لهیدگی: کلیپر نرم + لیمیتر در چند گذر کوتاه ──
    # نسخهٔ قدیم همهٔ فشار رو یک‌جا به یک لیمیتر می‌داد (≈۹dB GR = مچاله).
    # حالا: هر گذر حداکثر ۶dB گین می‌ده، پیک‌ها قبل از لیمیتر توسط کلیپر
    # نرم گرد می‌شن → گین‌ریداکشن لیمیتر کم می‌مونه و ترنزینت‌ها زنده‌ان.
    clip = m.get("clip")
    clip_db = None
    if isinstance(clip, dict):
        clip_db = float(clip.get("ceiling_db", -2.2))
    elif m.get("clip_ceiling_db") is not None:
        clip_db = float(m.get("clip_ceiling_db"))
    if clip_db is not None:
        y = soft_clip(y, ceiling_db=clip_db)
        rep.append(f"کلیپر نرم (گرد کردن پیک‌ها @ {clip_db:g}dB)")

    rel = m.get("limiter_release_ms", 120)
    for _ in range(4):
        l = integrated_lufs(y, sr)
        if l <= -69.0:
            break
        need = target - l
        if need <= 0.25:
            break
        y = y * np.float32(db2lin(min(need * 0.9, 6.0)))
        if clip_db is not None:
            y = soft_clip(y, ceiling_db=clip_db)
        y = Limiter(threshold_db=ceiling, release_ms=rel)(y, sr)

    # تصحیح نهایی: لیمیتر pedalboard گین جبرانی می‌ذاره و ممکنه از هدف رد کنیم
    l = integrated_lufs(y, sr)
    if -69.0 < l and l > target + 0.25:
        y = y * np.float32(db2lin(target - l))

    # سقف نهایی دقیق (حاشیهٔ ایمنی برای اورشوت انکود MP3)
    over = float(lin2db(np.max(np.abs(y)))) - (ceiling - 0.5)
    if over > 0:
        y = y * np.float32(db2lin(-over))
    rep.append(f"بلندی نهایی → {target:g} LUFS (سقف {ceiling:g}dB، چند گذر)")
    return y.astype(np.float32), rep

# ══════════════════ میکس وکال + موزیک ══════════════════

def mix_and_master(vocal, inst, sr, mx, master):
    """بالانس خودکار وکال و موزیک + مستر نهایی (بهینه برای رم کم)"""
    rep = []
    vocal = to_stereo(vocal.astype(np.float32, copy=False))
    inst = to_stereo(inst.astype(np.float32, copy=False))
    n = max(len(vocal), len(inst))
    if len(vocal) < n:
        vocal = np.pad(vocal, ((0, n - len(vocal)), (0, 0)))
    if len(inst) < n:
        inst = np.pad(inst, ((0, n - len(inst)), (0, 0)))

    duck_db = mx.get("duck_db", 0.0)
    if duck_db:
        inst = duck_under_vocal(inst, vocal, sr, duck_db)
        rep.append(f"جا باز کردن موزیک برای وکال (Ducking {duck_db}dB)")

    vg = mx.get("vocal_gain_db", 0.0)
    ig = mx.get("inst_gain_db", -2.0)
    # ضرب درجا — کپی اضافه نسازیم
    np.multiply(vocal, np.float32(db2lin(vg)), out=vocal)
    np.multiply(inst, np.float32(db2lin(ig)), out=inst)

    # ── بالانس خودکار: وکال نسبت به بیت در سطح هدف بشینه ──
    # (مثل میکس واقعی: وکال lead_db دسی‌بل بالاتر از بستر ساز — نه خیلی
    #  گم بشه، نه خیلی بزنه بیرون؛ مستقل از سطح ضبط هر فایل)
    lead_db = mx.get("vocal_lead_db")
    if lead_db is not None:
        vrms = float(lin2db(np.sqrt(np.mean(np.square(vocal))) + 1e-12))
        irms = float(lin2db(np.sqrt(np.mean(np.square(inst))) + 1e-12))
        adj = float(np.clip((irms + lead_db) - vrms, -12.0, 12.0))
        if abs(adj) > 0.1:
            np.multiply(vocal, np.float32(db2lin(adj)), out=vocal)
            rep.append(f"بالانس خودکار وکال (هدف {lead_db:+g}dB بالای بیت، "
                       f"اصلاح {adj:+.1f}dB)")

    y = vocal + inst
    rep.append(f"بالانس: وکال {vg:+g}dB / موزیک {ig:+g}dB")

    y, mrep = master_chain(y, sr, master)
    return y, rep + mrep


def mix_only(vocal, inst, sr, mx):
    """میکسِ خالص دو استمِ از قبل مسترشده — بدون مستر دوباره و بدون زنجیرهٔ وکال.

    برای وقتی که کاربر وکال و بیت رو جدا مستر کرده و فقط بالانس/میکس می‌خواد.
    خیلی سبک‌تر از mix_and_master (نه اتوتیون، نه زنجیرهٔ کامل، نه مستر) →
    روی فایل‌های بلند OOM نمی‌شه.

    مراحل: ducking (اختیاری) → بالانس خودکار وکال → چسب باس ملایم (اختیاری)
           → پهنا (اختیاری) → سقف ایمن (بدون تغییر بلندی).
    """
    from pedalboard import Compressor

    rep = []
    vocal = to_stereo(vocal.astype(np.float32, copy=False))
    inst = to_stereo(inst.astype(np.float32, copy=False))
    n = max(len(vocal), len(inst))
    if len(vocal) < n:
        vocal = np.pad(vocal, ((0, n - len(vocal)), (0, 0)))
    if len(inst) < n:
        inst = np.pad(inst, ((0, n - len(inst)), (0, 0)))

    duck_db = mx.get("duck_db", 0.0)
    if duck_db:
        inst = duck_under_vocal(inst, vocal, sr, duck_db)
        rep.append(f"جا باز کردن بیت برای وکال (Ducking {duck_db}dB)")

    vg = mx.get("vocal_gain_db", 0.0)
    ig = mx.get("inst_gain_db", 0.0)
    np.multiply(vocal, np.float32(db2lin(vg)), out=vocal)
    np.multiply(inst, np.float32(db2lin(ig)), out=inst)

    # بالانس خودکار: وکال در سطح هدف نسبت به بستر ساز (مستقل از سطح فایل‌ها)
    lead_db = mx.get("vocal_lead_db")
    if lead_db is not None:
        vrms = float(lin2db(np.sqrt(np.mean(np.square(vocal))) + 1e-12))
        irms = float(lin2db(np.sqrt(np.mean(np.square(inst))) + 1e-12))
        adj = float(np.clip((irms + lead_db) - vrms, -12.0, 12.0))
        if abs(adj) > 0.1:
            np.multiply(vocal, np.float32(db2lin(adj)), out=vocal)
            rep.append(f"بالانس خودکار وکال (هدف {lead_db:+g}dB بالای بیت، "
                       f"اصلاح {adj:+.1f}dB)")

    y = vocal + inst
    rep.append(f"بالانس: وکال {vg:+g}dB / موزیک {ig:+g}dB")

    # چسب باس ملایم (گلو — چسبندگی، بدون تغییر بلندی)
    gr = mx.get("glue_ratio")
    if gr:
        y = np.asarray(
            Compressor(threshold_db=-6.0, ratio=float(gr), attack_ms=30.0,
                       release_ms=250.0)(y, sr), dtype=np.float32)
        rep.append(f"چسب باس ملایم ({gr}:۱) — انسجام بدون مستر دوباره")

    # پهنا (اختیاری)
    w = mx.get("width", 1.0)
    if abs(w - 1.0) > 1e-4:
        y = width_ms(y, w)
        rep.append(f"پهنای استریو ({int(w * 100)}٪)")

    # سقف ایمن — بدون تغییر بلندی، فقط جلوگیری از کلیپ
    pk = float(np.max(np.abs(y)))
    if pk > 0.985:
        y = (y * np.float32(0.985 / pk)).astype(np.float32)
        rep.append("سقف ایمن (بدون تغییر بلندی مستر)")

    return y.astype(np.float32), rep
