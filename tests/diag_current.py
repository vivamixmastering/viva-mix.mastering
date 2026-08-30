# -*- coding: utf-8 -*-
"""تشخیص تجربی: هر مرحله از زنجیرهٔ پاپ ایرانی چقدر صدا رو «مچاله» می‌کنه؟"""
import numpy as np
import pyloudnorm as pyln

SR = 44100


def lufs(x, sr=SR):
    try:
        return pyln.Meter(sr).integrated_loudness(x)
    except Exception:
        return -70.0


def db(x):
    """تبدیل dB به ضریب دامنه (برای ساخت سیگنال تست با سطح مشخص)."""
    return 10 ** (x / 20.0)


def peak_db(x):
    return 20 * np.log10(np.max(np.abs(x)) + 1e-12)


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12)


def crest_db(x):
    return peak_db(x) - rms_db(x)


def make_vocal(dur=20.0, sr=SR):
    """وکال مصنوعی: ۵ نوت با هارمونیک + ویبراتو ۵.۵Hz + کمی بی‌کوکی (۳۰-۴۵ سنت)
    + انروپلویپ طبیعی (بالا و پایین رفتن دامنه بین کلمات)"""
    n = int(dur * sr)
    t = np.arange(n) / sr
    # نت‌ها: A3, C4, D4, E4, G4 (لا مینور) با دیتیون خطایی
    notes = [(220.0, 0.35), (261.6 * 2 ** (0.035), 1.0), (293.7 * 2 ** (-0.028), 1.0),
             (329.6 * 2 ** (0.042), 1.0), (392.0 * 2 ** (-0.03), 1.0)]
    seg = int(n / len(notes))
    f0 = np.zeros(n)
    for i, (f, amp) in enumerate(notes):
        s, e = i * seg, min((i + 1) * seg, n)
        f0[s:e] = f
    # گلیساندو بین نوت‌ها + ویبراتو ۵.۵ هرتز با عمق ۳۵ سنت (مثل تحریر ملایم)
    f0 = np.convolve(f0, np.hanning(2200) / np.hanning(2200).sum(), mode="same")
    vib = 2 ** (0.03 * np.sin(2 * np.pi * 5.5 * t) / 12 * 12)  # ±35 cents
    f0v = f0 * (2 ** (0.035 * np.sin(2 * np.pi * 5.5 * t)))
    # انروپلویپ: بالا پایین رفتن ۸ دسی‌بل مثل کلمات + نفسی بین نوت‌ها
    env = 0.55 + 0.45 * np.abs(np.convolve(np.sin(2 * np.pi * 0.4 * t),
                                            np.hanning(8820) / np.hanning(8820).sum(), mode="same"))
    # ویبراتو ±۴۲ سنت با ۵.۵ هرتز (تحریر ملایم)
    f0v = f0 * (2 ** (0.035 * np.sin(2 * np.pi * 5.5 * t)))
    # سنتز با انباشتگر فاز (روش صحیح — f(t)*t مدولاسیون کاذب می‌سازه)
    y = np.zeros(n)
    phase = np.zeros(8)
    for i in range(n):
        w = 2 * np.pi * f0v[i] / sr
        phase += w * np.arange(1, 9)
        y[i] = np.sum((0.75 ** np.arange(8)) * np.sin(phase))

    y *= env / np.max(np.abs(y))
    # کمی نویز نفس
    y += 0.0015 * np.random.default_rng(7).standard_normal(n)
    y = y / np.max(np.abs(y)) * 0.85  # پیک حدود -1.4dB مثل فایل‌های واقعی
    return y.astype(np.float32)


def make_vocal_slow(dur=20.0, sr=SR):
    """نسخهٔ برداری (سریع) با انباشتگر فاز — همان make_vocal ولی بدون حلقهٔ نمونه‌به‌نمونه"""
    n = int(dur * sr)
    t = np.arange(n) / sr
    notes = [(220.0, 0.0), (261.6, 0.035), (293.7, -0.028), (329.6, 0.042), (392.0, -0.03)]
    seg = int(n / len(notes))
    f0 = np.zeros(n)
    for i, (f, det) in enumerate(notes):
        s, e = i * seg, min((i + 1) * seg, n)
        f0[s:e] = f * 2 ** det
    f0 = np.convolve(f0, np.hanning(2200) / np.hanning(2200).sum(), mode="same")
    vib = 2 ** (0.035 * np.sin(2 * np.pi * 5.5 * t))
    f0v = f0 * vib
    env = 0.55 + 0.45 * np.abs(np.convolve(np.sin(2 * np.pi * 0.4 * t),
                                            np.hanning(8820) / np.hanning(8820).sum(), mode="same"))
    dphi = 2 * np.pi * f0v / sr
    y = np.zeros(n)
    amps = 0.75 ** np.arange(8)
    for h in range(1, 9):
        y += amps[h - 1] * np.sin(np.cumsum(dphi * h))
    y *= env / np.max(np.abs(y))
    y += 0.0015 * np.random.default_rng(7).standard_normal(n)
    return (y / np.max(np.abs(y)) * 0.85).astype(np.float32)


def make_inst(dur=20.0, sr=SR):
    n = int(dur * sr)
    t = np.arange(n) / sr
    y = np.zeros(n)
    # آکورد لا مینور + کیک ساده
    for f in (110.0, 164.8, 220.0, 329.6):
        y += 0.25 * np.sin(2 * np.pi * f * t)
    kick = np.zeros(n)
    for k in range(0, n, sr // 2):
        e = np.exp(-np.arange(sr // 8) / (sr * 0.03))
        kick[k:k + len(e)] += 0.7 * e * np.sin(2 * np.pi * 55 * np.arange(len(e)) / sr)
    y = y * 0.4 + kick
    y = y / np.max(np.abs(y)) * 0.7
    return y.astype(np.float32)


def report(name, x):
    print(f"{name:<38} LUFS {lufs(x):7.2f} | peak {peak_db(x):7.2f}dB | "
          f"crest {crest_db(x):5.2f}dB | rms {rms_db(x):6.2f}dB")
    return x


if __name__ == "__main__":
    np.random.seed(0)
    v = make_vocal()
    i = make_inst()
    print("=== ورودی‌ها ===")
    report("وکال خام", v)
    report("موزیک خام", i)

    from src.audio_engine import vocal_chain, mix_and_master
    import yaml
    presets = yaml.safe_load(open("presets/mastering_presets.yaml", encoding="utf-8"))["presets"]
    p = [q for q in presets if q["id"] == "persian_pop"][0]

    out, rep = vocal_chain(v[:, None], SR, p["vocal"])
    print("\n=== وکال بعد از زنجیرهٔ فعلی پاپ ایرانی ===")
    report("وکال پردازش‌شده", out)

    y, mrep = mix_and_master(out, i[:, None], SR, p["mix"], p["master"])
    print("\n=== خروجی نهایی فعلی (میکس+مستر) ===")
    report("آهنگ کامل", y)
    print("\n--- گزارش مراحل ---")
    for r in rep + mrep:
        print(" •", r)

    # ── تست جداگانهٔ لیمیتر pedalboard: آیا makeup gain می‌ذاره؟ ──
    from pedalboard import Limiter
    pre = y / np.max(np.abs(y)) * db(0.5)
    print("\n=== رفتار Limiter پدال‌بورد ===")
    lim = Limiter(threshold_db=-6.0, release_ms=100)(pre, SR)
    print(f"ورودی peak {peak_db(pre):.2f} → خروجی peak {peak_db(lim):.2f} "
          f"(اگه بیشتر از آستانه بود یعنی makeup می‌ذاره)")
