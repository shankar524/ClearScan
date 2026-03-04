import os
import gc
import time
import subprocess
import fitz
import numpy as np
import cv2
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_DIR   = "data"
OUTPUT_DIR  = "output"

LLAMA_CLI   = os.path.join("llama_bin", "llama-mtmd-cli.exe")
MODEL_PATH  = os.path.join("models", "Qwen3VL-2B-Instruct-Q4_K_M.gguf")
MMPROJ_PATH = os.path.join("models", "mmproj-Qwen3VL-2B-Instruct-F16.gguf")

DPI      = 200
MAX_SIZE = 1120

THREADS    = 4
MAX_TOKENS = 1500
CTX_SIZE   = "4096"

TEMP_DIR = "temp_pages"
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
# SETUP
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR,   exist_ok=True)

for path, label in [
    (LLAMA_CLI,   "llama-mtmd-cli binary"),
    (MODEL_PATH,  "model GGUF"),
    (MMPROJ_PATH, "mmproj GGUF"),
]:
    if not os.path.isfile(path):
        print(f"❌ Cannot find {label}: {path}")
        exit(1)

print("=" * 60)
print("Qwen3-VL-2B OCR  |  llama-mtmd-cli  |  CPU")
print(f"  Ctx: {CTX_SIZE}  |  Max tokens: {MAX_TOKENS}  |  Threads: {THREADS}")
print(f"  DPI: {DPI}  |  Max size: {MAX_SIZE}px")
print(f"  Jinja: ON  |  Think: OFF  |  Post-processing: OFF")
print("=" * 60)

import unicodedata
import re

def safe_stem(filename_stem):
    """
    Convert a filename stem to ASCII-safe version for temp file paths.
    Only used for temp image files passed to subprocess — not for output files.
    Examples:
        Åke-Malmström-hörd  →  Ake-Malmstrom-hord
        Östergård           →  Ostergard
    """
    # Decompose unicode characters (Å → A + combining ring)
    normalized = unicodedata.normalize("NFD", filename_stem)
    # Keep only ASCII characters
    ascii_only  = normalized.encode("ascii", "ignore").decode("ascii")
    # Replace any remaining non-alphanumeric/dash/underscore with underscore
    safe        = re.sub(r"[^\w\-]", "_", ascii_only)
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


def render_page(page, out_path, dpi=DPI):
    zoom = dpi / 72.0
    mat  = fitz.Matrix(zoom, zoom)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    arr  = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
    img  = Image.fromarray(arr, "RGB")
    del pix, arr

    w, h = img.size
    if max(w, h) > MAX_SIZE:
        scale = MAX_SIZE / max(w, h)
        img   = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    img, mode, contrast = preprocess_image(img)
    img.save(out_path, "JPEG", quality=92)
    size = img.size
    del img
    return size, mode, contrast


# ============================================================
# OCR — raw output, no post-processing
# ============================================================
def ocr_image(image_path):
    cmd = [
        LLAMA_CLI,
        "-m",                 MODEL_PATH,
        "--mmproj",           MMPROJ_PATH,
        "--image",            image_path,
        "--jinja",
        "-sys",               SYSTEM_PROMPT,
        "-p",                 USER_PROMPT,
        "-n",                 str(MAX_TOKENS),
        "--ctx-size",         CTX_SIZE,
        "-ngl",               "0",
        "--temp",             "0.7",
        "--top-p",            "0.8",
        "--top-k",            "20",
        "--min-p",            "0.0",
        "--presence-penalty", "1.5",
        "--repeat-penalty",   "1.1",
        "--repeat-last-n",    "128",
        "-t",                 str(THREADS),
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
    # Everything else is written to the file as-is.
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
# MAIN LOOP
# ============================================================
pdf_files = sorted([
    f for f in os.listdir(INPUT_DIR)
    if f.lower().endswith(".pdf")
])

if not pdf_files:
    print(f"❌ No PDFs found in '{INPUT_DIR}/'")
    exit(1)

print(f"\nFound {len(pdf_files)} PDF(s):\n")
for f in pdf_files:
    size_mb = os.path.getsize(os.path.join(INPUT_DIR, f)) / (1024 * 1024)
    print(f"  • {f}  ({size_mb:.1f} MB)")
print()

grand_start = time.time()

for pdf_filename in pdf_files:
    pdf_path = os.path.join(INPUT_DIR, pdf_filename)
    stem     = os.path.splitext(pdf_filename)[0]

    print(f"\n{'='*60}")
    print(f"Processing: {pdf_filename}")
    print(f"{'='*60}")

    out_path = os.path.join(OUTPUT_DIR, stem + ".txt")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(f"OCR OUTPUT: {pdf_filename}\n")
        fh.write(f"Model    : Qwen3-VL-2B (Q4_K_M GGUF)\n")
        fh.write(f"Started  : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"Settings : ctx={CTX_SIZE}, temp=0.7, presence_penalty=1.5, "
                 f"jinja=ON, no_think=ON, max_size={MAX_SIZE}px\n")
        fh.write("=" * 60 + "\n\n")

    doc       = fitz.open(pdf_path)
    num_pages = len(doc)
    print(f"  Pages: {num_pages}")

    page_errors = 0

    for page_num in range(num_pages):
        t0 = time.time()
        print(f"\n  Page {page_num+1}/{num_pages}  ", end="", flush=True)

        page     = doc[page_num]
        img_path = os.path.join(TEMP_DIR, f"{safe_stem(stem)}_p{page_num+1}.jpg")

        try:
            size, mode, contrast = render_page(page, img_path)
            print(f"({size[0]}×{size[1]}px {mode} c={contrast:.0f})  ",
                  end="", flush=True)
        except Exception as e:
            print(f"❌ RENDER ERROR: {e}")
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(f"--- PAGE {page_num+1}/{num_pages} ---\n")
                fh.write(f"[RENDER ERROR: {e}]\n\n")
            page_errors += 1
            continue

        text, had_error = ocr_image(img_path)
        elapsed = time.time() - t0

        if had_error:
            status = f"❌  {elapsed:.1f}s"
            page_errors += 1
        else:
            status = f"✅  {elapsed:.1f}s  (~{len(text.split())} words)"
        print(status)

        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(f"--- PAGE {page_num+1}/{num_pages}  "
                     f"[{elapsed:.1f}s | {mode} | c={contrast:.0f}] ---\n")
            fh.write(text)
            fh.write("\n\n")
            fh.flush()

        print(f"     → {out_path}")

        try:
            os.remove(img_path)
        except Exception:
            pass
        gc.collect()

    doc.close()

    elapsed_total = time.time() - grand_start
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write("=" * 60 + "\n")
        fh.write(f"Completed : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"Pages     : {num_pages}  (errors: {page_errors})\n")
        fh.write(f"Total time: {elapsed_total/60:.1f} minutes\n")

    print(f"\n  ✅ {out_path}  ({page_errors}/{num_pages} errors)")

import shutil
try:
    shutil.rmtree(TEMP_DIR)
except Exception:
    pass

grand_total = time.time() - grand_start
print(f"\n{'='*60}")
print(f"ALL DONE  —  {grand_total/60:.1f} minutes")
print(f"Results in: '{OUTPUT_DIR}/'")
print(f"{'='*60}")