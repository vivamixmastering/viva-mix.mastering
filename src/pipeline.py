# -*- coding: utf-8 -*-
"""
pipeline.py — اجرای کامل پردازش بر اساس حالت و پریست

حالت‌ها:
  vocal : فقط زنجیره وکال
  full  : فقط زنجیره مسترینگ روی آهنگ کامل
  two   : وکال + بیت (دو فایل) → میکس و مستر نهایی
  smart : جداسازی خودکار وکال با Demucs → مثل حالت two
"""
import logging
import os
import time
from pathlib import Path

import numpy as np
import yaml

from config import PRESETS_FILE, TMP_DIR
from src.audio_engine import (
    SR, encode_mp3, load_audio, master_chain, mix_and_master,
    save_wav, vocal_chain,
)
from src.match_eq import apply_match_eq

log = logging.getLogger("pipeline")


def load_presets():
    with open(PRESETS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)["presets"]


PRESETS = load_presets()


def get_preset(pid):
    for p in PRESETS:
        if p["id"] == pid:
            return p
    return PRESETS[0]


# ══════════════════ جداسازی هوشمند (Demucs) ══════════════════

def smart_available():
    """آیا حالت هوشمند روی این سرور قابل اجراست؟"""
    if os.getenv("SMART_MODE", "0") != "1":
        return False, "حالت هوشمند روی این سرور غیرفعاله (SMART_MODE=1 لازمه)."
    try:
        import demucs  # noqa: F401
        import torch  # noqa: F401
        return True, ""
    except ImportError:
        return False, "کتابخانه‌های جداسازی (demucs/torch) نصب نیستن."


def separate_stems(path, workdir):
    """جداسازی وکال و موزیک با Demucs — پردازش تکه‌تکه (پنجره با هم‌پوشانی
    ۵۰٪ و اتصال نرم هانینگ) تا حافظه رم پر نشه و فایل‌های طولانی (آهنگ کامل)
    هم بدون OOM شدن پردازش بشن.

    ⚠️ htdemucs روی CPU خیلی حافظه می‌خوره (توجه transformer):
    - سرور ۲ گیگ: قطعه پیشنهادی ۵-۶ ثانیه
    - سرور ۴+ گیگ: می‌تونی DEMUCS_CHUNK=15 یا بیشتر بدی
    (با متغیر محیطی DEMUCS_CHUNK و DEMUCS_THREADS قابل تنظیمه)

    → (vocal_path, inst_path)
    """
    import gc

    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    x, sr = load_audio(path)
    log.info("demucs: %d samples (%.1fs)", len(x), len(x) / sr)

    chunk_s = float(os.getenv("DEMUCS_CHUNK", "3"))
    threads = int(os.getenv("DEMUCS_THREADS", "1"))
    torch.set_num_threads(max(1, threads))

    # مدل قابل تنظیم — پیش‌فرض mdx_extra_q (کوانتیزه و سبک: با ~۱.۵ گیگ رم هم
    # اجرا می‌شه). اگه رم سرور بالاست (۴ گیگ+) می‌تونی DEMUCS_MODEL=htdemucs
    # بذاری که کیفیت جداسازی بهتری داره.
    model_name = os.getenv("DEMUCS_MODEL", "mdx_extra_q")
    model = get_model(model_name)
    model.eval()

    ref = x.mean(0)
    xm = ((x - ref.mean()) / (ref.std() + 1e-8)).astype(np.float32)
    del x
    gc.collect()

    chunk_len = int(chunk_s * sr)
    hop = chunk_len // 2    # هم‌پوشانی ۵۰٪ → بازسازی کامل با پنجره هانینگ
    vi = model.sources.index("vocals")

    out_vocal = np.zeros_like(xm, dtype=np.float32)
    out_inst = np.zeros_like(xm, dtype=np.float32)
    weight = np.zeros(len(xm), dtype=np.float32)

    n_chunks = max(1, int(np.ceil(len(xm) / hop)) + 1)
    pos = 0
    while pos < len(xm):
        done = int(pos / hop) + 1
        end = min(pos + chunk_len, len(xm))
        seg = xm[pos:end]
        wav = torch.from_numpy(seg.T.copy())[None]
        with torch.no_grad():
            sources = apply_model(model, wav, device="cpu", progress=False)[0]
        voc = sources[vi].numpy().T.astype(np.float32)
        ins = (sources.sum(0) - sources[vi]).numpy().T.astype(np.float32)
        fade = np.hanning(end - pos).astype(np.float32)[:, None]
        out_vocal[pos:end] += voc * fade
        out_inst[pos:end] += ins * fade
        weight[pos:end] += fade[:, 0]
        log.info("demucs chunk %d/%d done", done, n_chunks)
        if end >= len(xm):
            break
        pos += hop
        del sources, wav, voc, ins, seg
        gc.collect()

    w = np.maximum(weight, 1e-8)[:, None]
    out_vocal /= w
    out_inst /= w

    vp = Path(workdir) / f"sep_vocal_{int(time.time())}.wav"
    ip = Path(workdir) / f"sep_inst_{int(time.time())}.wav"
    save_wav(vp, out_vocal, sr)
    save_wav(ip, out_inst, sr)
    return vp, ip


# ══════════════════ پردازش اصلی ══════════════════

def process_mode(paths, mode, preset, workdir=None, match=None):
    """paths: {'vocal':..., 'inst':..., 'full':...} بر اساس حالت
    match: پروفایل تُنال مرجع (dict از match_eq.analyze) — اگه داده بشه،
           تُنالِ عنصر اصلی هر حالت به منحنی مرجع نزدیک می‌شه (Match EQ).
    خروجی: (مسیر فایل نهایی, لیست گزارش, ثانیه زمان پردازش)
    """
    workdir = Path(workdir or TMP_DIR)
    t0 = time.time()
    header = f"🎛️ پریست: {preset['name']}"

    if mode == "vocal":
        x, sr = load_audio(paths["vocal"])
        y, vrep = vocal_chain(x, sr, preset.get("vocal", {}))
        if match:
            y = apply_match_eq(y, sr, match)
            vrep = vrep + ["🎯 تطبیق تُنال با مرجع (Match EQ)"]
        rep = [header, "🎤 زنجیره وکال:"] + vrep
    elif mode == "full":
        x, sr = load_audio(paths["full"])
        if match:
            x = apply_match_eq(x, sr, match)
        y, mrep = master_chain(x, sr, preset.get("master", {}))
        rep = [header, "🎵 زنجیره مسترینگ:"]
        if match:
            rep.append("🎯 تطبیق تُنال با مرجع (Match EQ)")
        rep += mrep
    elif mode in ("two", "smart", "two_bleed"):
        if mode == "smart" and "vocal" not in paths:
            # اگه فقط آهنگ کامل داده شده، خودمان جداسازی می‌کنیم
            vp, ip = separate_stems(paths["full"], workdir)
            paths = {"vocal": str(vp), "inst": str(ip)}
        vx, sr = load_audio(paths["vocal"])
        ix, _ = load_audio(paths["inst"])
        if match:
            # بیت رو به منحنی مرجع نزدیک می‌کنیم تا وکال «سوار» همون بافت بشه
            ix = apply_match_eq(ix, sr, match)
        vcfg = dict(preset.get("vocal", {}))
        if mode in ("two_bleed", "smart"):
            # وکال جداسازی‌شده (ابزار کاربر یا Demucs) همیشه کمی نشت موزیک
            # داره — پاکسازی ضدانشتت فعال
            vcfg["bleed_safe"] = True
        v, vrep = vocal_chain(vx, sr, vcfg)
        del vx  # وکال خام دیگه لازم نیست — رم آزاد شه
        import gc as _gc
        _gc.collect()
        y, mixrep = mix_and_master(v, ix, sr, preset.get("mix", {}),
                                   preset.get("master", {}))
        rep = ([header, "🎤 زنجیره وکال:"] + vrep
               + ["🎧 میکس و مستر نهایی:"]
               + (["🎯 تطبیق تُنال با مرجع (Match EQ)"] if match else [])
               + mixrep)
    else:
        raise ValueError(f"حالت ناشناخته: {mode}")

    wav = workdir / f"out_{int(time.time())}.wav"
    save_wav(wav, y, sr)
    try:
        mp3 = wav.with_suffix(".mp3")
        encode_mp3(wav, mp3)
        out = mp3
    except Exception as e:
        log.warning("MP3 encode failed: %s", e)
        out = wav
    return out, rep, time.time() - t0
