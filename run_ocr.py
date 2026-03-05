#!/usr/bin/env python3
"""ClearScan — PDF-to-text OCR pipeline using Qwen3-VL-2B via llama.cpp."""

import argparse
import gc
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata

import cv2
import fitz
import numpy as np
from PIL import Image

# ============================================================
# DEFAULTS  (overridden by CLI args → env vars → these values)
# ============================================================
_DEFAULTS = {
    "input_dir":   "data",
    "output_dir":  "output",
    "llama_cli":   os.path.join("llama_bin", "llama-mtmd-cli.exe")
                   if sys.platform == "win32"
                   else os.path.join("llama_bin", "llama-mtmd-cli"),
    "model_path":  os.path.join("models", "Qwen3VL-2B-Instruct-Q4_K_M.gguf"),
    "mmproj_path": os.path.join("models", "mmproj-Qwen3VL-2B-Instruct-F16.gguf"),
    "dpi":         200,
    "max_size":    1120,
    "threads":     4,
    "max_tokens":  1500,
    "ctx_size":    4096,
    "temp":        0.7,
}

DEGRADED_CONTRAST_THRESHOLD = 60

SYSTEM_PROMPT = (
    "/no_think\n"
    "You are an OCR system. Your only job is to transcribe text from document images exactly as it appears. "
    "You output the transcribed text and nothing else. "
    "You never describe, summarize, or comment on the document. "
    "You never add formatting that is not present in the original."
)

USER_PROMPT = (
    "Transcribe every word of text visible in this document image.\n"
    "Rules:\n"
    "- Output the text exactly as it appears, line by line\n"
    "- Preserve Swedish characters: Å Ä Ö å ä ö\n"
    "- Include handwritten text\n"
    "- Write [REDACTED] for any blacked-out or marker-covered text\n"
    "- Write [?] for individual words you cannot read\n"
    "- Do not repeat any line or phrase\n"
    "- Stop at the end of the visible text"
)


# ============================================================
# CLI / ENV CONFIGURATION
# ============================================================
def _env(name, default=None, cast=None):
    """Read an env var with optional type casting."""
    val = os.environ.get(name, default)
    if val is not None and cast is not None:
        val = cast(val)
    return val


def parse_args(argv=None):
    """Build config from CLI flags → environment variables → built-in defaults."""
    p = argparse.ArgumentParser(
        description="ClearScan OCR — transcribe scanned PDFs to text",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment variables (used when CLI flags are omitted):\n"
            "  CLEARSCAN_INPUT_DIR     Input directory with PDFs\n"
            "  CLEARSCAN_OUTPUT_DIR    Output directory for .txt files\n"
            "  CLEARSCAN_LLAMA_CLI     Path to llama-mtmd-cli binary\n"
            "  CLEARSCAN_MODEL_PATH    Path to main GGUF model\n"
            "  CLEARSCAN_MMPROJ_PATH   Path to mmproj GGUF model\n"
            "  CLEARSCAN_THREADS       CPU threads  (default: 4)\n"
            "  CLEARSCAN_CTX_SIZE      Context size  (default: 4096)\n"
            "  CLEARSCAN_MAX_TOKENS    Max output tokens  (default: 1500)\n"
            "  CLEARSCAN_TEMP          Sampling temperature  (default: 0.7)\n"
        ),
    )
    p.add_argument("-i", "--input",      dest="input_dir",
                   default=_env("CLEARSCAN_INPUT_DIR", _DEFAULTS["input_dir"]),
                   help="Directory containing PDF files  (default: %(default)s)")
    p.add_argument("-o", "--output",     dest="output_dir",
                   default=_env("CLEARSCAN_OUTPUT_DIR", _DEFAULTS["output_dir"]),
                   help="Directory for OCR text output  (default: %(default)s)")
    p.add_argument("--llama-cli",        dest="llama_cli",
                   default=_env("CLEARSCAN_LLAMA_CLI", _DEFAULTS["llama_cli"]),
                   help="Path to llama-mtmd-cli binary")
    p.add_argument("--model",           dest="model_path",
                   default=_env("CLEARSCAN_MODEL_PATH", _DEFAULTS["model_path"]),
                   help="Path to Qwen3-VL GGUF model")
    p.add_argument("--mmproj",          dest="mmproj_path",
                   default=_env("CLEARSCAN_MMPROJ_PATH", _DEFAULTS["mmproj_path"]),
                   help="Path to mmproj GGUF model")
    p.add_argument("-t", "--threads",    type=int,
                   default=_env("CLEARSCAN_THREADS", _DEFAULTS["threads"], int),
                   help="CPU threads  (default: %(default)s)")
    p.add_argument("--ctx-size",         type=int,
                   default=_env("CLEARSCAN_CTX_SIZE", _DEFAULTS["ctx_size"], int),
                   help="Context window size  (default: %(default)s)")
    p.add_argument("--max-tokens",       type=int,
                   default=_env("CLEARSCAN_MAX_TOKENS", _DEFAULTS["max_tokens"], int),
                   help="Max output tokens  (default: %(default)s)")
    p.add_argument("--temp",             type=float,
                   default=_env("CLEARSCAN_TEMP", _DEFAULTS["temp"], float),
                   help="Sampling temperature  (default: %(default)s)")
    return p.parse_args(argv)

def safe_stem(filename_stem):
    """
    Convert a filename stem to ASCII-safe version for temp file paths.
    Only used for temp image files passed to subprocess — not for output files.
    Examples:
        Åke-Malmström-hörd  →  Ake-Malmstrom-hord
        Östergård           →  Ostergard
    """
    normalized = unicodedata.normalize("NFD", filename_stem)
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    safe       = re.sub(r"[^\w\-]", "_", ascii_only)
    return safe


# ============================================================
# PREPROCESSING
# ============================================================
def preprocess_image(pil_img):
    cv_img   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray     = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())

    if contrast < DEGRADED_CONTRAST_THRESHOLD:
        clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray      = clahe.apply(gray)
        kernel    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        denoised  = cv2.fastNlMeansDenoising(sharpened, h=10)
        result    = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB))
        mode      = "ENHANCED"
    else:
        kernel    = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        sharpened = cv2.filter2D(cv_img, -1, kernel)
        result    = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
        mode      = "CLEAN"

    return result, mode, contrast


def render_page(page, out_path, dpi, max_size):
    zoom = dpi / 72.0
    mat  = fitz.Matrix(zoom, zoom)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    arr  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    img  = Image.fromarray(arr, "RGB")
    del pix, arr

    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    img, mode, contrast = preprocess_image(img)
    img.save(out_path, "JPEG", quality=92)
    size = img.size
    del img
    return size, mode, contrast


# ============================================================
# OCR — raw output, no post-processing
# ============================================================
def ocr_image(image_path, cfg):
    """Run llama-mtmd-cli on a single image and return (text, had_error)."""
    cmd = [
        cfg.llama_cli,
        "-m",                 cfg.model_path,
        "--mmproj",           cfg.mmproj_path,
        "--image",            image_path,
        "--jinja",
        "-sys",               SYSTEM_PROMPT,
        "-p",                 USER_PROMPT,
        "-n",                 str(cfg.max_tokens),
        "--ctx-size",         str(cfg.ctx_size),
        "-ngl",               "0",
        "--temp",             str(cfg.temp),
        "--top-p",            "0.8",
        "--top-k",            "20",
        "--min-p",            "0.0",
        "--presence-penalty", "1.5",
        "--repeat-penalty",   "1.1",
        "--repeat-last-n",    "128",
        "-t",                 str(cfg.threads),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=False,
            timeout=600,
        )
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]", True

    stdout = result.stdout.decode("utf-8", errors="replace")
    stderr = result.stderr.decode("utf-8", errors="replace")

    if result.returncode != 0:
        return f"[OCR FAILED (code {result.returncode}):\n{stderr}]", True

    # Strip only llama.cpp internal diagnostic lines from stdout.
    skip_prefixes = (
        "llama_", "ggml_", "load_", "build:", "system_info",
        "sampling:", "generate:", "clip_", "encode_", "Log ",
        "main:", "common_", "llm_load", "print_info", "sched_",
        "<|im_start|>", "<|im_end|>", "mtmd_", "alloc_",
        "warmup", "encoding ", "decoding ", "image slice",
        "image decoded", "WARN:",
    )
    lines = stdout.splitlines()
    raw_lines = [
        ln for ln in lines
        if ln.strip() and not any(ln.lstrip().startswith(p) for p in skip_prefixes)
    ]

    return "\n".join(raw_lines), False


# ============================================================
# MAIN
# ============================================================
def main(argv=None):
    cfg = parse_args(argv)

    dpi      = _DEFAULTS["dpi"]
    max_size = _DEFAULTS["max_size"]
    temp_dir = "temp_pages"

    # Create directories
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # Validate required files
    for path, label in [
        (cfg.llama_cli,   "llama-mtmd-cli binary"),
        (cfg.model_path,  "model GGUF"),
        (cfg.mmproj_path, "mmproj GGUF"),
    ]:
        if not os.path.isfile(path):
            print(f"ERROR: Cannot find {label}: {path}", file=sys.stderr)
            sys.exit(1)

    print("=" * 60)
    print("Qwen3-VL-2B OCR  |  llama-mtmd-cli  |  CPU")
    print(f"  Input : {cfg.input_dir}")
    print(f"  Output: {cfg.output_dir}")
    print(f"  Ctx: {cfg.ctx_size}  |  Max tokens: {cfg.max_tokens}  |  Threads: {cfg.threads}")
    print(f"  DPI: {dpi}  |  Max size: {max_size}px  |  Temp: {cfg.temp}")
    print(f"  Jinja: ON  |  Think: OFF  |  Post-processing: OFF")
    print("=" * 60)

    # Discover PDFs
    pdf_files = sorted([
        f for f in os.listdir(cfg.input_dir)
        if f.lower().endswith(".pdf")
    ])

    if not pdf_files:
        print(f"ERROR: No PDFs found in '{cfg.input_dir}/'", file=sys.stderr)
        sys.exit(1)

    print(f"\nFound {len(pdf_files)} PDF(s):\n")
    for f in pdf_files:
        size_mb = os.path.getsize(os.path.join(cfg.input_dir, f)) / (1024 * 1024)
        print(f"  - {f}  ({size_mb:.1f} MB)")
    print()

    grand_start = time.time()

    for pdf_filename in pdf_files:
        pdf_path = os.path.join(cfg.input_dir, pdf_filename)
        stem     = os.path.splitext(pdf_filename)[0]

        print(f"\n{'='*60}")
        print(f"Processing: {pdf_filename}")
        print(f"{'='*60}")

        out_path = os.path.join(cfg.output_dir, stem + ".txt")
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(f"OCR OUTPUT: {pdf_filename}\n")
            fh.write(f"Model    : Qwen3-VL-2B (Q4_K_M GGUF)\n")
            fh.write(f"Started  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.write(f"Settings : ctx={cfg.ctx_size}, temp={cfg.temp}, presence_penalty=1.5, "
                     f"jinja=ON, no_think=ON, max_size={max_size}px\n")
            fh.write("=" * 60 + "\n\n")

        doc       = fitz.open(pdf_path)
        num_pages = len(doc)
        print(f"  Pages: {num_pages}")

        page_errors = 0

        for page_num in range(num_pages):
            t0 = time.time()
            print(f"\n  Page {page_num+1}/{num_pages}  ", end="", flush=True)

            page     = doc[page_num]
            img_path = os.path.join(temp_dir, f"{safe_stem(stem)}_p{page_num+1}.jpg")

            try:
                size, mode, contrast = render_page(page, img_path, dpi, max_size)
                print(f"({size[0]}x{size[1]}px {mode} c={contrast:.0f})  ",
                      end="", flush=True)
            except Exception as e:
                print(f"RENDER ERROR: {e}")
                with open(out_path, "a", encoding="utf-8") as fh:
                    fh.write(f"--- PAGE {page_num+1}/{num_pages} ---\n")
                    fh.write(f"[RENDER ERROR: {e}]\n\n")
                page_errors += 1
                continue

            text, had_error = ocr_image(img_path, cfg)
            elapsed = time.time() - t0

            if had_error:
                status = f"FAIL  {elapsed:.1f}s"
                page_errors += 1
            else:
                status = f"OK  {elapsed:.1f}s  (~{len(text.split())} words)"
            print(status)

            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(f"--- PAGE {page_num+1}/{num_pages}  "
                         f"[{elapsed:.1f}s | {mode} | c={contrast:.0f}] ---\n")
                fh.write(text)
                fh.write("\n\n")
                fh.flush()

            print(f"     -> {out_path}")

            try:
                os.remove(img_path)
            except OSError:
                pass
            gc.collect()

        doc.close()

        elapsed_total = time.time() - grand_start
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write("=" * 60 + "\n")
            fh.write(f"Completed : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            fh.write(f"Pages     : {num_pages}  (errors: {page_errors})\n")
            fh.write(f"Total time: {elapsed_total/60:.1f} minutes\n")

        print(f"\n  Done: {out_path}  ({page_errors}/{num_pages} errors)")

    try:
        shutil.rmtree(temp_dir)
    except OSError:
        pass

    grand_total = time.time() - grand_start
    print(f"\n{'='*60}")
    print(f"ALL DONE  —  {grand_total/60:.1f} minutes")
    print(f"Results in: '{cfg.output_dir}/'")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()