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
    # گارد ورودی degenerate: فایل خالی یا خیلی کوتاه → پیام واضح به‌جای کرش
    # (صفر/یک نمونه باعث خطای reshape در sosfilt و خطای channel-layout
    # در فیلترهای pedalboard می‌شد؛ حداقل ۵۱۲ نمونه = ~۱۲ms کافیه)
    if data.shape[0] < 512:
        raise ValueError(
            "فایل صوتی خالی یا خیلی کوتاه است (کمتر از ۱۲ میلی‌ثانیه) — "
            "لطفاً یک فایل صوتی واقعی ارسال کنید.")
    # ورودی NaN/Inf (فایل خراب) → صفر می‌کنیم تا کل زنجیره NaN نشه
    if not np.isfinite(data).all():
        log.warning("ورودی دارای NaN/Inf — پاک‌سازی شد")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
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
    """پوش دامنه با max-pool سبک (نرخ ~۳۴۴Hz) — مصرف رم ناچیز و مستقل از طول.

    ⚠️ مهم: همیشه کپی می‌گیره (np.abs) و هرگز ورودی رو درجا دست نمی‌زنه —
    چون وقتی x مونو (۱بعدی) باشه، a همون x هست و np.abs(a, out=a) سیگنال
    رو فول‌ویو رکتیفای می‌کرد → فرکانس دو برابر (چیپمونک/سنجاب)."""
    a = x.mean(axis=1) if x.ndim == 2 else x
    a = np.abs(a)                    # کپی — ورودی دست‌نخورده می‌مونه
    if a.dtype != np.float32:
        a = a.astype(np.float32)
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
                attack_ms, release_ms, max_cut_db=None):
    """هستهٔ دی‌اسر باندی: باند lo..hi رو جدا می‌کنه و فقط همون باند رو
    بر اساس انرژی خودش فشرده می‌کنه (بدون دست‌زدن به بقیهٔ طیف).

    max_cut_db: سقف گین‌ریداکشن (Range) — جلوی کات بیش‌ازحد میدرنج رو می‌گیره.
    بهینهٔ رم: جمع/ضرب درجا + آزادسازی زودهنگام آرایه‌های واسط."""
    hp = highpass(x, sr, lo_hz, order=4)
    band = lowpass(hp, sr, hi_hz, order=4)
    del hp
    env = env_follow(band, sr, attack_ms, release_ms)
    over = np.maximum(lin2db(env) - np.float32(threshold_db), 0.0)
    gr = over * (1.0 - 1.0 / ratio)
    if max_cut_db is not None:
        gr = np.minimum(gr, np.float32(max_cut_db))  # سقف کاهش گین (Range)
    gain = db2lin(-gr).astype(np.float32)
    del env, over, gr
    if gain.ndim == 1 and x.ndim == 2:
        gain = gain[:, None]
    # out = (x - band) + band*gain = x + band*(gain - 1)  — بدون آرایهٔ rest جدا
    np.subtract(gain, np.float32(1.0), out=gain)
    np.multiply(band, gain, out=band)
    np.add(x, band, out=x)
    del band
    return x.astype(np.float32)


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


def saturation(x, sr, drive_db=3.0, mix=0.3, asym=0.0):
    """اشباع نرم لامپی — هارمونیک زوج + فرد
    اورسمپلینگ ۲× با صفرگذاری + ضد الیاس ۱۹kHz (بدون الیاسینگ/هیس).

    asym = 0  → tanh متقارن (هارمونیک فرد — تیز/روشن)
    asym > 0  → بایاس درجه‌دوم (x + a·x²) قبل از tanh → هارمونیک زوج غالب
                = «صدای لامپی گرم/مخملی» به‌جای «دیستورشن تخت». چون x²
                جریان مستقیم می‌سازه، بعدش یه DC-blocker سبک اعمال می‌شه.

    نسخهٔ بهینهٔ رم: tanh درجا روی یک آرایه (به‌جای چند آرایهٔ موازی ۲×)."""
    single = x.ndim == 1
    if single:
        x = x[:, None]
    up = _upsample_zerostuff(x, 2)
    g = float(db2lin(drive_db))
    norm = np.float32(1.0 / float(np.tanh(g)))
    np.multiply(up, g, out=up)          # مقیاس با drive (درجا)
    if asym:
        sq = up * up                    # بایاس درجه‌دوم → هارمونیک زوج
        np.multiply(sq, np.float32(asym), out=sq)
        np.add(up, sq, out=up)
        del sq
    np.tanh(up, out=up)                 # tanh درجا
    np.multiply(up, norm, out=up)       # نرمال‌سازی unity-gain (درجا)
    up = lowpass(up, sr * 2, 19000.0)   # ضد الیاس (آرایهٔ جدید ۲×)
    up = up[::2]
    if asym:
        up = highpass(up, sr, 20.0)     # حذف DC ناشی از بایاس درجه‌دوم
    n = min(len(x), len(up))
    out = (1.0 - mix) * x[:n] + mix * up[:n]
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

# ══════════════════ مولتی‌باند + باس مونو (مستر/بیت) ══════════════════

def multiband_compress(x, sr, crossover_low=150.0, crossover_high=3000.0,
                       comp_low=None, comp_mid=None, comp_high=None):
    """مولتی‌باند کمپرسور ۳ بانده (Low/Mid/High) با کراس‌اوور بازسازی-کامل.

    کراس‌اوور با «تفریق» ساخته می‌شه (high = x − low) → جمع باندها دقیقاً
    برابر ورودیه، بدون برآمدگی/ناچ و بدون اختلاف فاز (عامل قبلی «جرت جرت»).
    comp_low/mid/high: dict با threshold_db, ratio, attack_ms, release_ms.
    """
    single = x.ndim == 1
    x = to_stereo(np.asarray(x, dtype=np.float32))

    def _lp(sig, freq):
        sos = spsig.butter(4, freq, btype="low", fs=sr, output="sos")
        return spsig.sosfilt(sos, sig, axis=0).astype(np.float32)

    low = _lp(x, crossover_low)
    band = x - low                    # mid + high (بازسازی کامل)
    mid = _lp(band, crossover_high)
    high = band - mid                 # بازسازی کامل

    def _c(cfg, thr, ratio, att, rel):
        cfg = cfg or {}
        return Compressor(
            threshold_db=cfg.get("threshold_db", thr),
            ratio=cfg.get("ratio", ratio),
            attack_ms=cfg.get("attack_ms", att),
            release_ms=cfg.get("release_ms", rel))

    low = np.asarray(_c(comp_low, -18, 2.2, 15, 150)(low, sr), dtype=np.float32)
    mid = np.asarray(_c(comp_mid, -16, 2.0, 10, 110)(mid, sr), dtype=np.float32)
    high = np.asarray(_c(comp_high, -14, 1.8, 5, 90)(high, sr), dtype=np.float32)

    # گین اختیاری هر باند (makeup) — مثلاً باند High برای بیرون‌زدن های‌هت
    g_low = float((comp_low or {}).get("gain_db", 0.0))
    g_mid = float((comp_mid or {}).get("gain_db", 0.0))
    g_high = float((comp_high or {}).get("gain_db", 0.0))
    if g_low:
        low = low * np.float32(db2lin(g_low))
    if g_mid:
        mid = mid * np.float32(db2lin(g_mid))
    if g_high:
        high = high * np.float32(db2lin(g_high))

    y = low + mid + high
    del low, mid, high, band
    return (y[:, 0] if single else y).astype(np.float32)


def bass_monoize(x, sr, freq=130.0):
    """جمع‌کردن باس زیر freq به مونو (سازگاری فاز + باس متمرکز).

    با تفریق (high = x − low) → بدون ناچ/برآمدگی در crossover.
    فرکانس‌های بالای freq دست‌نخورده استریو می‌مونن.
    """
    if x.ndim != 2:
        return x
    sos = spsig.butter(4, freq, btype="low", fs=sr, output="sos")
    lp = spsig.sosfilt(sos, x, axis=0).astype(np.float32)
    hp = x - lp
    mono = lp.mean(axis=1, keepdims=True)
    mono = np.repeat(mono, 2, axis=1)
    return (mono + hp).astype(np.float32)


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



def tpdf_dither(x, bits=16, seed=0):
    """دیتر TPDF (Triangular PDF) — نویز کوانتیزیشن ۱۶-بیت رو به نویز سفید
    غیرمرتبط تبدیل می‌کنه → حذف حس «خشن/غیرصاف» در سکوت‌ها و دم‌ها.

    دامنه ≈ ۱ LSB (≈ -96dBFS) — کاملاً زیر آستانهٔ شنوایی، فقط جلوی
    آرتیفکت کوانتیزیشن رو می‌گیره (استاندارد مسترینگ دیجیتال)."""
    rng = np.random.default_rng(seed)
    lsb = 2.0 ** (1 - bits)   # 1 LSB برای 16-bit = 1/32768
    n = x.shape[0]
    # دو نویز یکنواخت مستقل → مثلثی
    u1 = rng.uniform(-1, 1, n).astype(np.float32)
    u2 = rng.uniform(-1, 1, n).astype(np.float32)
    d = (u1 + u2) * (lsb * 0.5)
    del u1, u2
    if x.ndim == 2:
        x[:, 0] += d
        x[:, 1] += d
        del d
        return x.astype(np.float32)
    x = x + d
    del d
    return x.astype(np.float32)


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


def duck_band_under_vocal(inst, vocal, sr, depth_db=2.0, lo=200.0, hi=500.0,
                          attack_ms=15.0, release_ms=140.0):
    """داکینگ باند-محدود (sidechain فرکانسی) — بیت فقط در باند lo..hi
    (ناحیهٔ برخورد وکال و بیت، ۲۰۰–۵۰۰Hz) موقع اوج وکال پایین میاد، نه کل
    طیف → وکال بدون له‌شدن بیت «روی بیت می‌نشینه».

    ⚠️ باند با bandpass واقعی ساخته می‌شه (نه تفریق دو lowpass) — تفریق
    دو lowpass به‌خاطر اختلاف فاز، فرکانس‌های خارج باند (مثل ۱۰۰Hz) رو
    هم نشت می‌داد و داکینگ به‌اشتباه کل طیف رو کم می‌کرد."""
    if inst.ndim != 2:
        return inst
    sos = spsig.butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
    band = spsig.sosfilt(sos, inst, axis=0).astype(np.float32)
    rest = inst - band                     # بقیهٔ طیف (بازسازی کامل جبری)
    env = env_follow(to_mono(vocal), sr, attack_ms, release_ms)
    norm = env / (float(np.percentile(env, 95)) + 1e-9)
    gr = depth_db * np.clip(norm, 0.0, 1.0)
    gain = db2lin(-gr).astype(np.float32)[:, None]
    del env, norm, gr
    np.multiply(band, gain, out=band)
    np.add(rest, band, out=rest)
    del band
    return rest.astype(np.float32)

# ══════════════════ بلندی صدا (LUFS) ══════════════════

# ضرایب رسمی K-weighting (ITU-R BS.1770-4) برای 48kHz — به‌جای pyloudnorm
# (که به‌تنهایی ~۱۰۷MB رم موقع import می‌گرفت و موقع اندازه‌گیری کل سیگنال
# استریو رو به float64 تبدیل می‌کرد → عامل OOM روی سرویس ۱ گیگی)
_KW_B1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
_KW_A1 = [1.0, -1.69065929318241, 0.73248077421585]
_KW_B2 = [1.0, -2.0, 1.0]
_KW_A2 = [1.0, -1.99004745483398, 0.99007225036621]
_KW_SR = 48000


def integrated_lufs(x, sr):
    """بلندی یکپارچه EBU R128 (LUFS) — سبک و کم‌مصرف.

    روی مونو و با K-weighting استاندارد اندازه‌گیری می‌شه؛ خروجی با
    pyloudnorm در بازهٔ ±۰.۱۳ LUFS یکسانه، ولی کسری از رم مصرف می‌کنه.
    """
    from fractions import Fraction

    mono = x.mean(axis=1) if x.ndim == 2 else np.asarray(x)
    if len(mono) < 2:
        return -70.0

    # ری‌سمپل به 48kHz (ضرایب K-weighting برای این نرخ تعریف شدن)
    if sr != _KW_SR:
        fr = Fraction(_KW_SR, sr)
        mono = spsig.resample_poly(mono.astype(np.float64), fr.numerator,
                                   fr.denominator)
    # K-weighting: پیش‌فیلتر (high-pass) + شلف +4dB @ 1681.97Hz
    y = spsig.lfilter(_KW_B1, _KW_A1, mono)
    y = spsig.lfilter(_KW_B2, _KW_A2, y)

    block = int(0.4 * _KW_SR)   # بلوک ۴۰۰ms
    nb = len(y) // block
    if nb < 1:
        return -70.0
    z = y[: nb * block].reshape(nb, block)
    ms = (z * z).mean(axis=1) + 1e-12
    l = -0.691 + 10.0 * np.log10(ms)

    # گیت مطلق (-70 LUFS) و گیت نسبی (-10 LU زیر میانگین)
    idx = l > -70.0
    if idx.sum() == 0:
        return -70.0
    lg = l[idx]
    mean_abs = 10.0 * np.log10((np.power(10.0, lg / 10.0)).mean())
    idx2 = lg > (mean_abs - 10.0)
    if idx2.sum() == 0:
        return float(mean_abs)
    lg2 = lg[idx2]
    return float(10.0 * np.log10((np.power(10.0, lg2 / 10.0)).mean()))

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


def add_double_layer(y, sr, cfg):
    """لایهٔ بک‌ویس به وکال → استریو واقعی + ضخامت «دوتِرَک» (ADT).

    دو حالت (بر اساس cfg):
      • detune_cents > 0 → ADT واقعی: هر کانال یک دیتونِ خیلی ریز (± سنت)
        می‌گیره. این «ضخامت» و «حجم» دابل‌ترک استودیویی می‌ده (مثل Waves
        Doubler / Little AlterBoy) بدون فلام و بدون صدای دو-لاین (چون
        اختلاف فقط در پیچ است، نه زمان).
      • detune_cents = 0 → فقط دمِ ریورب (پهنا با عمق، ایمن و مونو-سازگار).

    delay_ms اختیاری زیر ۵ms (Haas) هم می‌تونه اضافه بشه ولی پیش‌فرض ۰ است
    تا هیچ فلامی برنگرده.
    """
    from pedalboard import PitchShift, Reverb

    mid = to_mono(y)
    detune = float(cfg.get("detune_cents", 0.0))
    wet = float(cfg.get("reverb_wet", 0.1))
    room = float(cfg.get("room", 0.35))
    damping = float(cfg.get("damping", 0.5))
    back_gain = float(cfg.get("back_gain", 0.4))
    mix = float(cfg.get("mix", 0.3))
    delay_ms = float(cfg.get("delay_ms", 0.0))

    if detune > 0.5:
        # ADT واقعی: دیتون ± روی هر کانال (مونو → مونو، کم‌مصرف)
        up = np.asarray(PitchShift(semitones=detune / 100.0)(mid, sr),
                        dtype=np.float32)
        down = np.asarray(PitchShift(semitones=-detune / 100.0)(mid, sr),
                          dtype=np.float32)
        back_l, back_r = up, down
    else:
        # ریورب-فقط (سازگار با مونو — بک‌ویس در جمع مونو حذف می‌شه)
        back = np.asarray(
            Reverb(room_size=room, damping=damping,
                   wet_level=wet, dry_level=0.0, width=1.0)(mid, sr),
            dtype=np.float32)
        back_l = back_r = back

    # Haas خیلی کوتاه (اختیاری، زیر ۵ms — بدون فلام)
    if delay_ms > 0:
        d = int(delay_ms / 1000.0 * sr)
        if d > 0:
            back_l = np.concatenate([np.zeros(d, np.float32), back_l[:-d]])
            back_r = np.concatenate([np.zeros(d, np.float32), back_r[:-d]])

    # ریورب سبک روی بک‌ویس برای عمق (اگه دیتون بود، فقط برای فضا)
    if wet > 0 and detune > 0.5:
        back_l = np.asarray(
            Reverb(room_size=room, damping=damping,
                   wet_level=wet, dry_level=1.0, width=1.0)(back_l, sr),
            dtype=np.float32)
        back_r = np.asarray(
            Reverb(room_size=room, damping=damping,
                   wet_level=wet, dry_level=1.0, width=1.0)(back_r, sr),
            dtype=np.float32)

    g = np.float32(back_gain * mix)
    L = mid + back_l * g
    R = mid + back_r * g
    out = np.stack([L, R], axis=1)
    pk = float(np.max(np.abs(out)))
    if pk > 0.985:
        out = out * np.float32(0.985 / pk)
    return out.astype(np.float32)


def vocal_space(y, sr, cfg):
    """فضاسازی حرفه‌ای وکال/سلفژ: دیلی + ریورب + پره‌دلی + هوا — ضد «حمومی».

    مشکل صدای حمومی سه عامله: ۱) ریورب بدون پره‌دلی → وکال توی خودِ ریورب
    گم می‌شه؛ ۲) گل و باکس (۲۰۰–۴۰۰Hz) توی دم ریورب → صدا کدر و «توالت‌مانند»؛
    ۳) دمِ خیلی بلند که همه‌چیز رو می‌شوره.

    این ماژول:
      • پره‌دلی (۴۰–۵۰ms) → وکال جلوت‌تر، ریورب پشتش.
      • های‌پس روی ورودی ریورب و دیلی → حذف گل/باکس (بدون از دست دادن گرما).
      • هوا/درخشش فقط روی دم ریورب → دم ابریشمی و شیشه‌ای، نه تاریک.
      • گین و پیک‌گارد در پایان → بلندی درست بدون دیستورت.

    cfg کلیدها: pre_delay_ms, room, damping, verb_wet, verb_hpf_hz,
                air_mix, air_drive_db, delay_time_s, delay_feedback,
                delay_mix, delay_hpf_hz, delay_lpf_hz, gain_db
    """
    from pedalboard import Delay, HighpassFilter, LowpassFilter, Reverb

    single = y.ndim == 1
    y = to_stereo(y).astype(np.float32)

    # ── پره‌دلی: ریورب کمی بعد از وکال شروع می‌شه → وکال جلو و شفاف ──
    pd_ms = float(cfg.get("pre_delay_ms", 45.0))
    pd_n = min(int(pd_ms / 1000.0 * sr), max(len(y) - 1, 0))
    # یک آرایهٔ کامل فقط (roll) — بدون zeros_like + کپی جدا
    verb_in = np.roll(y, pd_n, axis=0) if pd_n > 0 else y
    if pd_n > 0:
        verb_in[:pd_n] = 0.0

    # ── حذف گل/باکس از ورودی ریورب ──
    verb_in = HighpassFilter(
        cutoff_frequency_hz=float(cfg.get("verb_hpf_hz", 260.0)))(verb_in, sr)

    # ── دم ریورب (فقط خیس) ──
    room = float(cfg.get("room", 0.5))
    damping = float(cfg.get("damping", 0.5))
    verb = Reverb(room_size=room, damping=damping,
                  wet_level=1.0, dry_level=0.0, width=1.0)(verb_in, sr)
    del verb_in

    # ── هوا روی دم ریورب → دم ابریشمی/شیشه‌ای نه تاریک ──
    # (با block_apply تا روی فایل‌های بلند OOM نشه — آپ‌سمپلینگ ۴× فقط بلوکی)
    air_mix = float(cfg.get("air_mix", 0.3))
    if air_mix > 0:
        verb = block_apply(
            lambda blk: air_exciter(blk, sr, freq=11000.0,
                                    drive_db=float(cfg.get("air_drive_db", 2.5)),
                                    mix=air_mix),
            verb, sr, block_s=15.0)

    # ── جمع تدریجی (ضرب درجا + جمع درجا) → حداقل آرایهٔ هم‌زمان ──
    out = np.empty_like(y)
    np.copyto(out, y)

    # دیلی (اکو) با فیلتر ضدگل — بلافاصله جمع و آزادسازی
    dmix = float(cfg.get("delay_mix", 0.12))
    if dmix > 0:
        dl = Delay(delay_seconds=float(cfg.get("delay_time_s", 0.24)),
                   feedback=float(cfg.get("delay_feedback", 0.32)),
                   mix=1.0)(y, sr)
        dl = HighpassFilter(
            cutoff_frequency_hz=float(cfg.get("delay_hpf_hz", 320.0)))(dl, sr)
        lpf = cfg.get("delay_lpf_hz")
        if lpf:
            dl = LowpassFilter(cutoff_frequency_hz=float(lpf))(dl, sr)
        np.multiply(dl, np.float32(dmix), out=dl)
        out += dl
        del dl

    # ریورب
    np.multiply(verb, np.float32(float(cfg.get("verb_wet", 0.32))), out=verb)
    out += verb
    del verb

    # ── گین درست + گارد پیک ──
    g = float(cfg.get("gain_db", 0.0))
    if g:
        np.multiply(out, np.float32(db2lin(g)), out=out)
    pk = float(np.max(np.abs(out)))
    if pk > 0.98:
        np.multiply(out, np.float32(0.98 / pk), out=out)

    return (out[:, 0] if single else out).astype(np.float32, copy=False)


# ══════════════════ معماری سه‌لایه وکال (Depth / Body / Presence) ══════════════════

def _stereo_spread(mono, sr, width=1.0):
    """ساخت استریو از مونو بدون فلام/کمب — با جفت فیلتر آل‌پس مرتبه‌اول.

    دو آل‌پس با ضریب مخالف (c و -c) روی L و R → اختلاف فاز نرم (نه تأخیر
    زمانی) → عرض واقعی که در جمعِ مونو کمب نمی‌سازه (دامنهٔ آل‌پس ثابته).
    سپس side با `width` مقیاس می‌شه (1.0=کاملاً عریض، 0=مونو).
    این‌جا از delay (Haas) استفاده نمی‌کنیم چون روی سیگنال تُنال (body)
    کمب‌فیلتر می‌سازه.
    """
    c = 0.8  # ضریب آل‌پس (عرض خوب + جمعِ مونو ~۰.۹۴، |c|<1)
    c = np.float32(c)
    bL = np.array([c, np.float32(1.0)], dtype=np.float32)
    aL = np.array([np.float32(1.0), c], dtype=np.float32)
    bR = np.array([-c, np.float32(1.0)], dtype=np.float32)
    aR = np.array([np.float32(1.0), -c], dtype=np.float32)
    L = spsig.lfilter(bL, aL, mono)
    R = spsig.lfilter(bR, aR, mono)
    mid = (L + R) * np.float32(0.5)
    side = (L - R) * np.float32(0.5) * np.float32(width)
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


def _depth_layer(dry, sr, pre_delay_ms=35.0):
    """لایهٔ Depth (عمق) — ریوربِ پلیت بلند + فیلترها برای حس فاصله.

    HPF 250Hz (حذف باس → کدر نشه) → پره‌دلی → ریورب پلیت (فقط خیس، عریض)
    → LPF 7500Hz (حذف تیزی، فقط حس فاصله) → پهن‌کردن استریو.
    پره‌دلی با np.roll (ریوربِ pedalboard پارامتر پره‌دلی نداره).
    مونو پردازش می‌شه تا مصرف رم نصف بشه؛ عرض در پایان با آل‌پس ساخته می‌شه.
    """
    from pedalboard import Reverb
    d = to_mono(dry).astype(np.float32, copy=False)
    if pre_delay_ms > 0:
        pd = int(pre_delay_ms / 1000.0 * sr)
        d = np.roll(d, pd, axis=0)
        d[:pd] = 0.0
    d = HighpassFilter(cutoff_frequency_hz=250.0)(d, sr)
    d = Reverb(room_size=0.85, damping=0.35,
               wet_level=1.0, dry_level=0.0, width=1.0)(d, sr)
    d = LowpassFilter(cutoff_frequency_hz=7500.0)(d, sr)
    return _stereo_spread(np.asarray(d, dtype=np.float32), sr, width=1.0)


def _body_layer(dry, sr, body_width=0.45, drive_db=4.5):
    """لایهٔ Body (میانی) — کمپرسور سنگین + گرمای هارمونیک برای ضخامت.

    کمپرسور 5:1 → EQ گرم (+3dB @250Hz) + کات تیزی (-2dB @5000Hz)
    → اشباع لامپی (tanh اورسمپل‌شده، گرم نه خشن) → پهنای نیمه‌عریض.
    مونو پردازش می‌شه؛ عرض در پایان با آل‌پس ساخته می‌شه (body_width).
    """
    b = to_mono(dry).astype(np.float32, copy=False)
    b = np.asarray(
        Compressor(threshold_db=-22.0, ratio=5.0,
                   attack_ms=22.0, release_ms=120.0)(b, sr), dtype=np.float32)
    # +3dB@250 قبلاً به باند ۱۵۰–۳۰۰Hz (ناحیهٔ گلآلود) میافزود → کات شد.
    b = PeakFilter(cutoff_frequency_hz=250.0, gain_db=-2.0, q=1.2)(b, sr)
    b = HighShelfFilter(cutoff_frequency_hz=5000.0, gain_db=-2.0)(b, sr)
    # اشباع نامتقارنِ ملایم (هارمونیک زوج ظریف) → مخملی بدون اکتاو/سنجابی.
    # asym زیاد (0.3) هارمونیک 2f رو غالب می‌کرد → صدای نازال/اکتاو بالا.
    # drive کمتر → هارمونیک‌های پایین (500/750Hz) که گل 300-1000 رو زیاد
    # می‌کردن کمتر تولید بشن.
    b = saturation(b, sr, drive_db=drive_db, mix=0.45, asym=0.12)
    # ریورب سبک — لایهٔ میانی دیگه خشک نباشه (حس فضا/عمق).
    # ⚠️ دیلی قبلاً ۱۳۰ms بود و حس «اکو/لگ» نسبت به لایهٔ رویی می‌داد
    # → حذف شد؛ فقط ریورب سبک مونده (بدون تکرارِ قابل‌شنیدن).
    from pedalboard import Reverb
    b = np.asarray(Reverb(room_size=0.35, damping=0.5,
                          wet_level=0.2, dry_level=1.0, width=1.0)(b, sr),
                   dtype=np.float32)
    return _stereo_spread(b, sr, width=float(body_width))


def shimmer_layer(dry, sr):
    """لایهٔ Shimmer — «هوای مرتفع» + ریورب، خیلی محو (بدون پیچ‌شیفت).

    ⚠️ قبلاً یک اکتاو پیچ‌شیفت (PitchShift +12) داشت که فرمت‌های وکال رو
    هم اکتاو-بالا می‌کرد → صدای «آلوین/سنجابی». حالا بدون پیچ‌شیفت: فقط
    باند بالای ۴kHz (هوا) از سیگنال خشک → ریورب → استریو. درخشش شیشه‌ای
    رو از EQ لایهٔ رویی می‌گیریم، نه از اکتاو اینجا."""
    from pedalboard import Reverb
    m = to_mono(dry).astype(np.float32, copy=False)
    # فقط باند هوا (بالای ۴kHz) — بدون تغییر زیروبمی، بدون فرمت جابه‌جا
    sos = spsig.butter(4, 4000.0, btype="high", fs=sr, output="sos")
    m = spsig.sosfilt(sos, m, axis=0).astype(np.float32)
    m = Reverb(room_size=0.6, damping=0.5,
               wet_level=1.0, dry_level=0.0, width=1.0)(m, sr)
    return _stereo_spread(np.asarray(m, dtype=np.float32), sr, width=1.0)


def slap_delay(y, sr, time_s=0.035, mix=0.06):
    """دیلی خیلی کوتاه (<50ms، بدون فیدبک) — ضخامت early-reflection بدون فلام."""
    from pedalboard import Delay
    single = y.ndim == 1
    yy = to_stereo(y).astype(np.float32)
    d = Delay(delay_seconds=float(time_s), feedback=0.0, mix=1.0)(yy, sr)
    out = yy * (1.0 - float(mix)) + d * float(mix)
    return (out[:, 0] if single else out).astype(np.float32)


def add_three_layer(presence, dry, sr, cfg):
    """معماری سه‌لایه: Depth (عمق) + Body (ضخامت) + Presence (شفاف).

    depth و body از dry (وکال خام) ساخته می‌شن و نسبت به سطحِ presence
    (مرجع 0dB) با -20dB و -10dB جمع می‌شن → عمق و چندبعدی بودن.
    ساخت پشت‌سرهم + جمع درجا برای مصرف رم کم (سرویس ۱ گیگ).
    """
    presence = to_stereo(presence).astype(np.float32)
    p_rms = float(np.sqrt(np.mean(np.square(presence)) + 1e-12))
    n = len(presence)

    # ── لایهٔ Depth ──
    depth = _depth_layer(dry, sr, float(cfg.get("depth_pre_delay_ms", 35.0)))
    if len(depth) > n:
        depth = depth[:n]
    elif len(depth) < n:
        depth = np.pad(depth, ((0, n - len(depth)), (0, 0)))
    d_rms = float(np.sqrt(np.mean(np.square(depth)) + 1e-12))
    if d_rms > 1e-9:
        depth = depth * np.float32(db2lin(cfg.get("depth_level_db", -20.0)) * p_rms / d_rms)
    np.add(presence, depth, out=presence)
    del depth

    # ── لایهٔ Body ──
    body = _body_layer(dry, sr, body_width=float(cfg.get("body_width", 0.45)))
    if len(body) > n:
        body = body[:n]
    elif len(body) < n:
        body = np.pad(body, ((0, n - len(body)), (0, 0)))
    b_rms = float(np.sqrt(np.mean(np.square(body)) + 1e-12))
    if b_rms > 1e-9:
        body = body * np.float32(db2lin(cfg.get("body_level_db", -10.0)) * p_rms / b_rms)
    np.add(presence, body, out=presence)
    del body

    # ── لایهٔ Shimmer (درخشش شیشه‌ای — خیلی محو) ──
    sh = shimmer_layer(dry, sr)
    if len(sh) > n:
        sh = sh[:n]
    elif len(sh) < n:
        sh = np.pad(sh, ((0, n - len(sh)), (0, 0)))
    s_rms = float(np.sqrt(np.mean(np.square(sh)) + 1e-12))
    if s_rms > 1e-9:
        sh = sh * np.float32(db2lin(cfg.get("shimmer_level_db", -29.0)) * p_rms / s_rms)
    np.add(presence, sh, out=presence)
    del sh

    # گارد پیک
    pk = float(np.max(np.abs(presence)))
    if pk > 0.985:
        presence = presence * np.float32(0.985 / pk)
    return presence.astype(np.float32)


# ══════════════════ ماژول‌های تکمیلی وکال (شیشه‌ای/ابریشمی/مخملی) ══════════════════

def _sos_band(lo, hi, fs, btype="bandpass", order=4):
    """فیلتر butter با ضرایب float32 (خروجی sosfilt float32 بمونه — نصف رم)."""
    sos = spsig.butter(order, [lo, hi], btype=btype, fs=fs, output="sos")
    return sos.astype(np.float32)


def dynamic_resonance_eq(x, sr, cfg=None):
    """Dynamic EQ روی رزونانس‌های فردی (۳۰۰/۱۲۰۰/۳۵۰۰Hz) — کات فقط وقتی
    انرژی باند از آستانه رد بشه، نه همیشه → صدا در حالت عادی طبیعی می‌مونه.

    بر پایهٔ _band_deess (همون هستهٔ دی‌اسر باندی) — برداری و کم‌مصرف."""
    cfg = cfg or {}
    freqs = cfg.get("freqs", ((300.0, 2.0, -3.5), (1200.0, 2.5, -2.5), (3500.0, 2.0, -2.0)))
    thr = float(cfg.get("threshold_db", -18.0))
    att = float(cfg.get("attack_ms", 8.0))
    rel = float(cfg.get("release_ms", 100.0))
    y = x
    for fc, q, maxcut in freqs:
        bw = fc / q
        lo = max(20.0, fc - bw / 2.0)
        hi = fc + bw / 2.0
        y = _band_deess(y, sr, lo, hi, thr, 6.0, att, rel, max_cut_db=abs(maxcut))
    return y


def formant_aware_warmth(x, sr, cfg=None):
    """گرمای حفظ‌کنندهٔ فرمنت — گرما @240Hz + ناچ دور فرمنت‌های تقریبی
    (۷۰۰/۱۲۰۰/۲۵۰۰Hz) تا تیمبر/فرمنت جابه‌جا نشه. (تقریب عملی، نه LPC)."""
    cfg = cfg or {}
    warmth_db = float(cfg.get("warmth_db", 2.5))
    warmth_freq = float(cfg.get("warmth_freq", 240.0))
    notches = cfg.get("notch_freqs", (700.0, 1200.0, 2500.0))
    width = float(cfg.get("notch_width_hz", 150.0))
    warmed = PeakFilter(cutoff_frequency_hz=warmth_freq, gain_db=warmth_db,
                        q=1.2)(x, sr)
    added = warmed - x
    del warmed
    for f in notches:
        sos = _sos_band(f - width / 2.0, f + width / 2.0, sr,
                        btype="bandstop", order=2)
        added = spsig.sosfilt(sos, added, axis=0).astype(np.float32)
    return (x + added).astype(np.float32)




def multiband_harmonic_exciter(x, sr, cfg=None):
    """اکسایتر چندباندی — به هر باند جدا هارمونیک زوج اضافه می‌کنه:
    پایین (گرما)، میانی (وضوح)، بالا (شیشه‌ای). فقط هارمونیک اضافه‌شده
    جمع می‌شه، نه کل باند دوباره.

    باندهای پایین/میانی (۲× فرکانس < ۱۹kHz) مستقیم مربع می‌شن (بدون
    اورسمپل → نصف رم)؛ فقط باند بالا اورسمپل ۲× می‌گیره (ضد الیاس)."""
    cfg = cfg or {}
    bands = cfg.get("bands", ((200.0, 800.0, 0.08), (2000.0, 5000.0, 0.03),
                              (8000.0, 16000.0, 0.18)))
    single = x.ndim == 1
    y = to_stereo(x).astype(np.float32)
    for lo, hi, drive in bands:
        sos = _sos_band(lo, hi, sr, btype="bandpass", order=4)
        band = spsig.sosfilt(sos, y, axis=0).astype(np.float32)
        p = float(np.max(np.abs(band))) + 1e-9
        # مربع مستقیم (هارمونیک زوج) + حذف DC؛ برای باند بالا یک LPF ملایم
        # ضد الیاس (بدون اورسمپل ۲× → نصف رم).
        harm = band * band
        np.multiply(harm, np.float32(drive / p), out=harm)
        if hi * 2.0 >= 19000.0:
            harm = lowpass(harm, sr, 18000.0)
        harm = highpass(harm, sr, 20.0)
        y = y + harm
        del band, harm
    return (y[:, 0] if single else y).astype(np.float32)


def vocal_transient_designer(x, sr, cfg=None):
    """نرم‌کردن Attack کانسوننت‌های بی‌صدا (پ/ت/ک/چ) فقط در باند ۲–۸kHz
    بدون دست‌زدن به واکه‌ها → مستقیم حس مخملی. برداری با env_follow."""
    cfg = cfg or {}
    red = float(cfg.get("attack_reduction", 0.35))
    lo, hi = cfg.get("detect_band", (2000.0, 8000.0))
    att = float(cfg.get("attack_ms", 3.0))
    rel = float(cfg.get("release_ms", 40.0))
    sos = _sos_band(lo, hi, sr, btype="bandpass", order=4)
    band = spsig.sosfilt(sos, x, axis=0).astype(np.float32)
    e_fast = env_follow(band, sr, attack_ms=att, release_ms=rel)
    e_slow = env_follow(band, sr, attack_ms=att * 4.0, release_ms=rel * 4.0)
    trans = np.maximum(e_fast - e_slow, 0.0)
    del e_fast, e_slow, band
    peak = float(np.percentile(trans, 99)) + 1e-9
    gr = 1.0 - red * (trans / peak)
    del trans
    if x.ndim == 2:
        gr = gr[:, None]
    return (x * gr).astype(np.float32)


def micro_pitch_vibrato(x, sr, cfg=None):
    """لرزش ریز پیچ (±۴ سنت @5Hz) روی لایهٔ موازی محو (-24dB) — حس زنده
    بدون بی‌ثباتی محسوس. با resampling نرم (np.interp).

    بهینهٔ رم: فقط phase سراسری (float32) full-length می‌مونه؛ interp
    بلوکی انجام می‌شه تا آرایه‌های موقت float64 (خروجی np.interp) فقط
    به اندازهٔ بلوک ۳۰ ثانیه‌ای باشن، نه کل فایل."""
    cfg = cfg or {}
    rate = float(cfg.get("rate_hz", 5.0))
    depth_cents = float(cfg.get("depth_cents", 4.0))
    level_db = float(cfg.get("mix_level_db", -24.0))
    single = x.ndim == 1
    xx = to_stereo(x)
    if xx.dtype != np.float32:
        xx = xx.astype(np.float32)
    n = len(xx)
    nch = xx.shape[1]
    # phase سراسری (مونو) — ساخته‌شده مرحله‌ای درجا
    ph = np.arange(n, dtype=np.float32)
    np.multiply(ph, np.float32(2.0 * np.pi * rate / sr), out=ph)
    np.sin(ph, out=ph)                              # LFO
    np.multiply(ph, np.float32(depth_cents / 1200.0), out=ph)  # نیم‌پرده تقریبی
    np.add(ph, np.float32(1.0), out=ph)
    np.cumsum(ph, out=ph)                           # فاز تجمعی
    np.multiply(ph, np.float32((n - 1) / (float(ph[-1]) + 1e-9)), out=ph)
    g = np.float32(db2lin(level_db))
    out = np.empty_like(xx)
    blk = int(30.0 * sr)
    margin = 8  # جابه‌جایی حداکثر <۴ نمونه؛ margin برای interp امن
    for s in range(0, n, blk):
        e = min(s + blk, n)
        s0 = max(0, s - margin)
        e0 = min(n, e + margin)
        base = np.arange(s0, e0, dtype=np.float32)
        for ch in range(nch):
            out[s:e, ch] = np.interp(ph[s:e], base, xx[s0:e0, ch]).astype(np.float32)
    del ph
    np.multiply(out, g, out=out)
    return (out[:, 0] if single else out).astype(np.float32, copy=False)


def dynamic_stereo_width(x, sr, cfg=None):
    """پهنای استریو پویا — نوسان خیلی محو (±۷٪) side بر اساس انرژی لحظه‌ای
    → فضا استاتیک/مکانیکی حس نشه."""
    cfg = cfg or {}
    base_w = float(cfg.get("base_width", 1.0))
    depth = float(cfg.get("modulation_depth", 0.07))
    smooth_ms = float(cfg.get("smoothing_ms", 150.0))
    if x.ndim != 2:
        return x
    mid = (x[:, 0] + x[:, 1]) * np.float32(0.5)
    side = (x[:, 0] - x[:, 1]) * np.float32(0.5)
    env = env_follow(mid, sr, attack_ms=smooth_ms, release_ms=smooth_ms)
    norm = env / (float(np.percentile(env, 99)) + 1e-9)
    del env
    curve = np.float32(base_w) + np.float32(depth) * (norm - float(norm.mean()))
    del norm
    side = side * curve
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


# ══════════════════ زنجیره وکال ══════════════════

def vocal_chain(x, sr, v):
    """زنجیره کامل وکال بر اساس تنظیمات پریست → (سیگنال, گزارش مراحل)"""
    rep = []
    # بدون کپی اضافه (to_stereo روی ورودی ۲ کاناله خودش رو برمی‌گردونه)
    y = to_stereo(np.asarray(x, dtype=np.float32))
    mono_mode = False

    # ── ترمیم استریو: وکال جدا‌شده یک کانالش ضعیفه → کانال قوی مبنای هر دو ──
    if v.get("stereo_repair"):
        y = stereo_repair(y, sr)
        # بعد از ترمیم، L==R (مونوی واقعی) → کل زنجیره رو مونو پردازش کن تا
        # مصرف رم نصف بشه (روی سرویس ۱ گیگی عامل جلوگیری از OOM). پهنا رو
        # لایهٔ بک‌ویس در پایان دوباره می‌سازه.
        y = to_mono(y)
        mono_mode = True
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

    # ── ذخیرهٔ خشکِ خام (برای لایه‌های Depth/Body در معماری سه‌لایه) ──
    # مونو ذخیره می‌شه چون لایه‌های Depth/Body از همین منبع مونو ساخته می‌شن
    # (عرض رو خودِ لایه‌ها در پایان با آل‌پس می‌سازن) → نصف مصرف رم.
    # بعد از این خط y فقط reassign می‌شه و آرایهٔ استریوی قبلی آزاد می‌شه.
    dry_ref = to_mono(y)

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
                # در حالت مونو (بعد از stereo_repair) خروجی اتوتیون هم مونوست
                y = yt if mono_mode else to_stereo(yt)
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
    treb = v.get("eq_high_shelf")
    if treb:
        y = HighShelfFilter(cutoff_frequency_hz=treb["freq"],
                            gain_db=treb["gain_db"])(y, sr)
        rep.append(f"تریبل (شلف بالا {treb['gain_db']:+g}dB @ {treb['freq']}Hz)")
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
        # باند دوم ملایم (رزونانس میانی ۴–۵kHz) — ابریشمی‌تر، بدون خفه‌کردن
        f2 = de.get("freq2")
        if f2:
            y = block_apply(
                lambda blk: deesser(blk, sr, f2, de.get("threshold_db", -24.0),
                                    de.get("ratio2", 3.0)),
                y, sr)
            rep.append(f"دی‌اسر باند دوم @ {int(f2)}Hz (ابریشمی)")

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
    # هوای فوق‌بالا (>14kHz) — درخشش شیشه‌ای از ناحیه‌های خیلی بالاتر
    ai2 = v.get("air2")
    if ai2:
        y = block_apply(
            lambda blk: air_exciter(blk, sr, ai2["freq"], ai2["drive_db"], ai2["mix"]),
            y, sr, block_s=15.0)
        rep.append(f"هوای فوق‌بالا (Air @ {ai2['freq']}Hz) — شیشه‌ای")

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

    # ── ماژول‌های تکمیلی (شیشه‌ای/ابریشمی/مخملی) — همیشه روشن ──
    # ترتیب: اول تمیزکاری رزونانس → گرما/اکسایتر (رنگ) → ترانزینت (فرم‌دهی)
    # نکته: formant_aware_warmth حذف شد چون گرما @240Hz رو دوباره روی
    # eq_warm (+2.5dB @240) می‌چید → باند 240Hz دوبار بوست می‌خورد (+5.5dB)
    # و ناچ‌های 1200/2500Hz هم وضوح ناحیهٔ ۱.۲–۳.۵kHz رو می‌بریدن.
    y = dynamic_resonance_eq(y, sr)
    rep.append("داینامیک EQ رزونانس‌های فردی (۳۰۰/۱۲۰۰/۳۵۰۰Hz)")
    y = multiband_harmonic_exciter(y, sr)
    rep.append("اکسایتر چندباندی (گرما/وضوح/شیشه‌ای)")
    y = vocal_transient_designer(y, sr)
    rep.append("نرم‌کردن اتک کانسوننت‌ها (پ/ت/ک/چ) — مخملی")

    # ── فضاسازی ──
    # پریست‌های اصلی (معماری سه‌لایه): ریورب بلند به لایهٔ Depth منتقل شد؛
    # اینجا فقط دیلی خیلی کوتاه (<50ms، بدون فیدبک) برای ضخامت early-reflection
    # می‌مونه — بدون tail بلند، بدون فلام.
    slap = v.get("slap_delay")
    if slap and slap.get("enabled"):
        y = slap_delay(y, sr,
                       time_s=slap.get("time_s", 0.035),
                       mix=slap.get("mix", 0.06))
        rep.append("دیلی کوتاه (slap) — ضخامت early-reflection")

    # سازگاری با پریست‌های قدیمی که هنوز از delay/reverb جدا (یا space) استفاده
    # می‌کنن — رفتار فضاسازی کامل قبلی براشون حفظ می‌شه.
    sp = v.get("space")
    if sp is None and (v.get("delay") or v.get("reverb")):
        sp = {"enabled": True}
        if v.get("delay"):
            sp["delay_time_s"] = v["delay"].get("time_s", 0.24)
            sp["delay_feedback"] = v["delay"].get("feedback", 0.3)
            sp["delay_mix"] = v["delay"].get("mix", 0.12)
        if v.get("reverb"):
            sp["room"] = v["reverb"].get("room", 0.5)
            sp["damping"] = v["reverb"].get("damping", 0.5)
            sp["verb_wet"] = v["reverb"].get("wet", 0.32)
    if sp and sp.get("enabled", True):
        y = vocal_space(y, sr, sp)
        rep.append("فضاسازی حرفه‌ای (پره‌دلی + دیلی + ریورب + هوای دم)")

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

    # ── معماری سه‌لایه (Depth/Body/Presence) — عمق و چندبعدی بودن ──
    # جایگزین بک‌ویسِ دیتون‌دار قبلی؛ فضاسازی و ضخامت از لایه‌های Depth/Body
    # میان و لایهٔ Presence (همین زنجیره) شفافیت اصلی رو نگه می‌داره.
    tl = v.get("three_layer")
    if tl and tl.get("enabled"):
        y = add_three_layer(y, dry_ref, sr, tl)
        rep.append("سه‌لایه وکال (Depth + Body + Shimmer + Presence) — عمق و ضخامت")

    # ── ویبراتو ریز پیچ (لایه موازی محو) + پهنای استریو پویا ──
    y = y + micro_pitch_vibrato(y, sr)
    rep.append("لرزش ریز پیچ (موازی محو) — حس زنده")
    y = dynamic_stereo_width(y, sr)
    rep.append("پهنای استریو پویا (نفس‌کشیدن فضا)")

    # گارد پیک نهایی (بعد از جمع ویبراتو و پهنای پویا — بدون دیستورت)
    pk = float(np.max(np.abs(y)))
    if pk > 0.985:
        y = y * np.float32(0.985 / pk)

    # خروجی همیشه استریو (اگه زنجیره مونو بود، اینجا پهن می‌شه)
    if y.ndim == 1:
        y = to_stereo(y)

    return y.astype(np.float32), rep


# ══════════════════ هارمونی (جابه‌جایی زیروبمی) ══════════════════

def harmonize(x, sr, semitones):
    """جابه‌جایی زیروبمی (هارمونی) حافظِ فرمت — با واکودر WORLD (pyworld).

    WORLD پیت (F0) و پوش طیفی (فرمت‌ها) رو جدا تحلیل می‌کنه؛ اینجا فقط F0
    جابه‌جا می‌شه و پوش طیفی ثابت می‌مونه → همون خواننده، فقط نت عوض شده
    (بدون اثر میکی‌موس/سنجاب). برخلاف PitchShift ساده که فرمت‌ها رو هم
    همراه پیت جابه‌جا می‌کنه.

    semitones مثبت = زیرتر (بالا). خروجی: استریو float32 هم‌طول ورودی.
    پردازش بلوکی (۶۰s + هم‌پوشانی نرم) برای مصرف رم کم روی سرویس ۱ گیگی."""
    try:
        import pyworld as pw
    except Exception as e:
        log.warning("pyworld در دسترس نیست — fallback به PitchShift ساده: %s", e)
        from pedalboard import PitchShift
        y = to_stereo(x).astype(np.float32)
        return np.asarray(PitchShift(semitones=float(semitones))(y, sr),
                          dtype=np.float32)

    semis = float(semitones)
    if abs(semis) < 0.05:
        return to_stereo(x).astype(np.float32)

    single = x.ndim == 1
    mono = to_mono(x).astype(np.float64)
    n = len(mono)

    def _shift_block(blk, s_ratio):
        # WORLD: تحلیل F0 + پوش طیفی + ناپریودیک → فقط F0 جابه‌جا → سنتز
        _f0, t_est = pw.dio(blk, sr)
        f0 = pw.stonemask(blk, _f0, t_est, sr)
        sp = pw.cheaptrick(blk, f0, t_est, sr)
        ap = pw.d4c(blk, f0, t_est, sr)
        f0_s = f0 * s_ratio
        out = pw.synthesize(f0_s, sp, ap, sr)
        if len(out) < len(blk):
            out = np.pad(out, (0, len(blk) - len(out)))
        return out[: len(blk)]

    # بلوک‌بندی با هم‌پوشانی نرم (بدون کلیک در مرزها)
    blk_s = 60.0
    ov_s = 2.0
    blen = int(blk_s * sr)
    ov = int(ov_s * sr)
    r = 2.0 ** (semis / 12.0)
    if n <= blen:
        out = _shift_block(mono, r).astype(np.float32)
    else:
        out = np.zeros(n, dtype=np.float64)
        w = np.zeros(n, dtype=np.float64)
        pos = 0
        while pos < n:
            end = min(pos + blen, n)
            seg = _shift_block(mono[pos:end], r)
            fade = np.hanning(end - pos)
            out[pos:end] += seg * fade
            w[pos:end] += fade
            if end >= n:
                break
            pos += blen - ov
        out = (out / np.maximum(w, 1e-8)).astype(np.float32)

    return to_stereo(out).astype(np.float32)


# ══════════════════ داینامیک EQ (تفکیک سازها) ══════════════════

def _eq_band(x, sr, band):
    """اعمال یک باند EQ (peak / low_shelf / high_shelf)."""
    t = band.get("type", "peak")
    f = float(band["freq"])
    g = float(band.get("gain_db", 0.0))
    q = float(band.get("q", 0.9))
    if t == "peak":
        return PeakFilter(cutoff_frequency_hz=f, gain_db=g, q=q)(x, sr)
    if t == "low_shelf":
        return LowShelfFilter(cutoff_frequency_hz=f, gain_db=g, q=q)(x, sr)
    if t == "high_shelf":
        return HighShelfFilter(cutoff_frequency_hz=f, gain_db=g)(x, sr)
    return x


def dynamic_eq(x, sr, bands):
    """داینامیک EQ — هر باند فقط وقتی بلندتر از آستانه شد کات می‌شه (نه همیشه).

    برخلاف EQ استاتیک (که همیشه بوست/کات می‌کنه)، اینجا کات فقط موقع شلوغی
    باند اعمال می‌شه → سازهای مختلف روی هم سوار نمی‌شن و «تفکیک» واضح‌تر
    می‌شه، بدون اینکه بدنهٔ صدا وقتی لازم نیست کم بشه.

    bands: لیست dict با lo_hz, hi_hz, threshold_db, ratio, attack_ms, release_ms.
    """
    single = x.ndim == 1
    y = to_stereo(x).astype(np.float32)
    for b in bands:
        y = _band_deess(
            y, sr,
            lo_hz=float(b.get("lo_hz", 300.0)),
            hi_hz=float(b.get("hi_hz", 500.0)),
            threshold_db=float(b.get("threshold_db", -24.0)),
            ratio=float(b.get("ratio", 2.0)),
            attack_ms=float(b.get("attack_ms", 8.0)),
            release_ms=float(b.get("release_ms", 80.0)),
            max_cut_db=b.get("max_cut_db"))
    return (y[:, 0] if single else y).astype(np.float32)


def ms_eq(x, sr, cfg):
    """EQ جداگانه Mid/Side — مرکز واضح / اطراف باز (فراتر از پهنای صرف).

    mid:  بوست/کات روی مرکز (کیک، باس، وکال اصلی) → فوکوس‌تر
    side: بوست/کات روی کناره‌ها (سازهای اطراف، فضا) → هوا/عمق بدون شلوغی مرکز
    """
    if x.ndim != 2:
        return x
    mid = (x[:, 0] + x[:, 1]) * 0.5
    side = (x[:, 0] - x[:, 1]) * 0.5
    for b in (cfg.get("mid") or []):
        mid = np.asarray(_eq_band(mid, sr, b), dtype=np.float32)
    for b in (cfg.get("side") or []):
        side = np.asarray(_eq_band(side, sr, b), dtype=np.float32)
    out = np.empty_like(x)
    np.add(mid, side, out=out[:, 0])      # L = mid + side (درجا)
    np.subtract(mid, side, out=out[:, 1])  # R = mid - side (درجا)
    return out.astype(np.float32)


def transient_shaper(x, sr, low_punch_db=1.2, split_hz=2500.0):
    """Transient Shaper جدا از کمپرسور — شکل ترنزینت رو تغییر می‌ده بدون
    بلندترکردن کل حجم.

    باند لو: تقویت اتک (پانچ کیک/باس) — فقط لبهٔ ترنزینت بلند می‌شه،
             نه کل باس → پانچ واضح‌تر بدون له‌شدگی.
    باند های: نرم‌کردن sustain (پایداری های‌هت/سنج) → شفافیت بیشتر، کمتر انبوه.
    """
    single = x.ndim == 1
    y = to_stereo(x).astype(np.float32)

    low = lowpass(y, sr, split_hz)
    high = y - low

    # ── باند لو: پانچ اتک (تفاوت پوش سریع/کند = موقع ترنزینت) ──
    e_fast = env_follow(low, sr, attack_ms=2.0, release_ms=25.0)
    e_slow = env_follow(low, sr, attack_ms=30.0, release_ms=160.0)
    trans = e_fast / np.maximum(e_slow, 1e-4)
    amt = float(db2lin(low_punch_db) - 1.0)
    punch = (1.0 + amt * np.clip(trans - 1.0, 0.0, 4.0) / 4.0).astype(np.float32)
    del e_fast, e_slow, trans
    punch = np.clip(punch, 1.0, float(db2lin(low_punch_db)))
    if y.ndim == 2:
        punch = punch[:, None]
    np.multiply(low, punch, out=low)          # درجا
    del punch

    # ── باند های: کم‌کردن sustain با پوش کند (فقط بخش پایداری) ──
    e_hi = env_follow(high, sr, attack_ms=12.0, release_ms=140.0)
    e_hi_db = lin2db(np.maximum(e_hi, 1e-6))
    over = np.maximum(e_hi_db - np.float32(-30.0), 0.0)
    gain = db2lin(-over * 0.4).astype(np.float32)   # زانوی نرم، کات ملایم
    del e_hi, e_hi_db, over
    if y.ndim == 2:
        gain = gain[:, None]
    np.multiply(high, gain, out=high)         # درجا
    del gain

    np.add(low, high, out=low)                # درجا
    del high
    return (low[:, 0] if single else low).astype(np.float32)


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

    # گیت نرم — پاک‌سازی هیس/نویز «شششش» سکوت‌ها (پایان آهنگ و بین فریزها)
    # بدون دست‌زدن به موزیک (آستانه خیلی پایین، فقط کف نویز رو می‌بنده)
    g = m.get("gate")
    if g is not False:
        gc = g if isinstance(g, dict) else {}
        y = noise_gate(y, sr,
                       threshold_db=gc.get("threshold_db", -55.0),
                       ratio=gc.get("ratio", 3.0),
                       release_ms=gc.get("release_ms", 180.0))
        rep.append("گیت نرم — پاک‌سازی هیس/نویز سکوت‌ها")

    dip = m.get("eq_dip")
    if dip:
        y = PeakFilter(cutoff_frequency_hz=dip["freq"], gain_db=dip["gain_db"],
                       q=dip.get("q", 1.2))(y, sr)
        rep.append(f"اصلاح میدرنج مستر ({dip['gain_db']:+g}dB @ {dip['freq']}Hz)")

    # داینامیک EQ — کات فقط موقع شلوغی باند (تفکیک سازها، نه کات همیشگی)
    deq = m.get("dyn_eq")
    if deq:
        y = dynamic_eq(y, sr, deq)
        rep.append("داینامیک EQ (میدرنج گل‌آلود + حضور) — تفکیک سازها")

    bc = m.get("bus_comp")
    if bc:
        y = Compressor(threshold_db=bc["threshold_db"], ratio=bc["ratio"],
                       attack_ms=bc["attack_ms"], release_ms=bc["release_ms"])(y, sr)
        rep.append(f"کمپرسور باس ({bc['ratio']}:۱) — چسبندگی کل آهنگ")

    # مولتی‌باند ۳ بانده — پانچ کیک/باس و وضوح سازها (جدا از میدرنج)
    mb = m.get("multiband")
    if mb:
        y = multiband_compress(
            y, sr,
            crossover_low=float(mb.get("crossover_low", 150.0)),
            crossover_high=float(mb.get("crossover_high", 3000.0)),
            comp_low=mb.get("low"), comp_mid=mb.get("mid"), comp_high=mb.get("high"))
        rep.append("مولتی‌باند (Low/Mid/High) — پانچ باس و وضوح سازها")

    # ترنزینت‌شیپر — شکل ترنزینت جدا از کمپرسور (پانچ کیک + شفافیت های‌هت)
    tr = m.get("transient")
    if tr:
        y = transient_shaper(
            y, sr,
            low_punch_db=float(tr.get("low_punch_db", 1.2)),
            split_hz=float(tr.get("split_hz", 2500.0)))
        rep.append("ترنزینت‌شیپر — پانچ کیک و شفافیت های‌هت")

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
    # اشباع/گرمی قبل از EQ بالا — تا هارمونیک‌های تولیدشده توسط EQ بالا
    # کنترل/شکل‌گیری بشن (نه اینکه بی‌کنترل روی باند حساس ۶–۱۱kHz بشینن)
    sat = m.get("sat")
    if sat:
        y = block_apply(
            lambda blk: saturation(blk, sr, sat["drive_db"], sat["mix"]), y, sr,
            block_s=30.0)
        rep.append(f"اشباع/گرمی (قبل از EQ بالا — {sat['drive_db']}dB)")

    hs = m.get("eq_high_shelf")
    if hs:
        y = HighShelfFilter(cutoff_frequency_hz=hs["freq"],
                            gain_db=hs["gain_db"])(y, sr)
        rep.append(f"درخشش کل ({hs['gain_db']:+g}dB @ {hs['freq']}Hz)")

    pr = m.get("eq_presence")
    if pr:
        y = PeakFilter(cutoff_frequency_hz=pr["freq"], gain_db=pr["gain_db"],
                       q=pr.get("q", 1.0))(y, sr)
        rep.append(f"حضور سازها ({pr['gain_db']:+g}dB @ {pr['freq']}Hz)")

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

    # EQ جداگانه Mid/Side — مرکز واضح / اطراف باز (تفکیک تن رنگ)
    mse = m.get("ms_eq")
    if mse:
        y = ms_eq(y, sr, mse)
        rep.append("Mid/Side EQ — فوکوس مرکز + هوای اطراف")

    # باس مونو — متمرکزکردن زیر بم (سازگاری فاز + باس تمیز)
    bm = m.get("bass_mono")
    if bm:
        y = bass_monoize(y, sr, freq=float(bm.get("freq", 130.0)))
        rep.append(f"باس مونو (زیر {int(bm.get('freq', 130.0))}Hz)")

    ceiling = m.get("ceiling", -1.0)
    target = m.get("lufs", -12.0)

    # ── بلندیِ بالا بدون لهیدگی: کلیپر نرم + لیمیتر در چند گذر کوتاه ──
    clip = m.get("clip")
    clip_db = None
    if isinstance(clip, dict):
        clip_db = float(clip.get("ceiling_db", -2.2))
    elif m.get("clip_ceiling_db") is not None:
        clip_db = float(m.get("clip_ceiling_db"))
    if clip_db is not None:
        y = soft_clip(y, ceiling_db=clip_db)
        rep.append(f"کلیپر نرم (گرد کردن پیک‌ها @ {clip_db:g}dB)")

    # ── رساندن به LUFS هدف با گین پلکانی + لیمیتر ──
    # ⚠️ لیمیتر pedalboard «دو کمپرسور + کلیپر سخت در 0dBFS» داره؛ یعنی خروجی
    # همیشه سقف 0dBFS (نه ceiling). پس حلقه تا (target - ceiling) بالا می‌بره
    # و بعد با گین ceiling پیک نهایی به ceiling و بلندی به target می‌رسه.
    # آستانهٔ لیمیتر باید از ceiling پایین‌تر باشه (این‌جا ceiling - 1.5dB) تا
    # کمپرسورها کرست رو به‌اندازهٔ کافی کم کنن و به بلندی هدف برسیم.
    rel = m.get("limiter_release_ms", 120)
    lim_thr = ceiling - 2.0          # آستانهٔ فشرده‌سازی لیمیتر (زیر سقف نهایی)
    loop_target = target - ceiling   # بلندیِ هدف در سقف 0dBFS (قبل از گین ceiling)
    for _ in range(5):
        l = integrated_lufs(y, sr)
        if l <= -69.0:
            break
        need = loop_target - l
        if need <= 0.25:
            break
        y = y * np.float32(db2lin(min(need * 0.85, 5.0)))
        y = Limiter(threshold_db=lim_thr, release_ms=rel)(y, sr)

    # تصحیح دقیق نهایی: لیمیتر (کلیپر سخت 0dBFS) غیرخطیه و ممکنه از هدف رد بشیم
    # → اگه بلندتر از هدف شدیم، با گین منفی دقیقاً به loop_target برمی‌گردونیم
    # (کاهش گین هیچ کلیپی نمی‌سازه، پس دقیق و امنه).
    l = integrated_lufs(y, sr)
    if -69.0 < l and l > loop_target + 0.1:
        y = y * np.float32(db2lin(loop_target - l))

    # گین نهایی: پیک → ceiling (بلندی → target). لیمیتر کلیپر سخت در 0dBFS
    # گذاشته، پس ضرب در ceiling دقیقاً پیک رو به ceiling می‌رسونه (بدون overshoot).
    y = y * np.float32(db2lin(ceiling))

    # حاشیهٔ ایمنی کوچک برای اورشوت انکود MP3 (فقط اگه چیزی رد شده باشه)
    over = float(lin2db(np.max(np.abs(y)))) - (ceiling - 0.2)
    if over > 0:
        y = y * np.float32(db2lin(-over))
    rep.append(f"بلندی نهایی → {target:g} LUFS (سقف {ceiling:g}dB، چند گذر)")

    # دیتر TPDF — حذف آرتیفکت کوانتیزیشن ۱۶-بیت (استاندارد مسترینگ)
    if m.get("dither", False):
        y = tpdf_dither(y, bits=16)
        rep.append("دیتر TPDF (نویز کوانتیزیشن → سفید غیرمرتبط)")
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
        inst = duck_band_under_vocal(inst, vocal, sr, duck_db,
                                     lo=200.0, hi=500.0)
        rep.append(f"داکینگ باند ۲۰۰–۵۰۰Hz برای وکال ({duck_db}dB)")

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
        inst = duck_band_under_vocal(inst, vocal, sr, duck_db,
                                     lo=200.0, hi=500.0)
        rep.append(f"داکینگ باند ۲۰۰–۵۰۰Hz برای وکال ({duck_db}dB)")

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
