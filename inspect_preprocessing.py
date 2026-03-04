import os
import fitz
import numpy as np
import cv2
from PIL import Image

# ============================================================
# CONFIGURATION
# ============================================================
INPUT_DIR  = "data"
OUTPUT_DIR = "debug_preprocessed"

# ↓↓↓ SET YOUR PDF FILENAME HERE ↓↓↓
PDF_FILENAME = "pol-2017-10-03-E63-05-B.pdf"
# ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

DPI      = 200
MAX_SIZE = 1120

DEGRADED_CONTRAST_THRESHOLD = 45   # lowered from 60

# ============================================================
# SETUP
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

pdf_path = os.path.join(INPUT_DIR, PDF_FILENAME)
if not os.path.isfile(pdf_path):
    print(f"❌ Cannot find PDF: {pdf_path}")
    exit(1)

# ============================================================
# PREPROCESSING — updated pipeline
# ============================================================
def preprocess_image(pil_img):
    cv_img   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    gray     = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    contrast = float(gray.std())

    if contrast < DEGRADED_CONTRAST_THRESHOLD:
        # FIXED ORDER: denoise first → then CLAHE → then sharpen
        # Old order was CLAHE → sharpen → denoise which amplified noise before removing it
        denoised  = cv2.fastNlMeansDenoising(gray, h=10)
        clahe     = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))  # reduced from 2.0 to avoid amplifying background grain
        enhanced  = clahe.apply(denoised)
        kernel    = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        result    = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))
        mode      = "ENHANCED"
    else:
        # FIXED: operate on grayscale instead of BGR color image
        # removes color scan noise and yellow paper tint, focuses model on pure luminance contrast
        kernel    = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        result    = Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB))
        mode      = "CLEAN"

    return result, mode, contrast


def render_page(page, dpi=DPI):
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
    return img, mode, contrast


# ============================================================
# MAIN — render + preprocess + save, no model involved
# ============================================================
stem      = os.path.splitext(PDF_FILENAME)[0]
doc       = fitz.open(pdf_path)
num_pages = len(doc)

print("=" * 60)
print(f"Preprocessing inspection: {PDF_FILENAME}")
print(f"Pages     : {num_pages}")
print(f"DPI       : {DPI}  |  Max size: {MAX_SIZE}px")
print(f"Threshold : contrast < {DEGRADED_CONTRAST_THRESHOLD} → ENHANCED")
print(f"Output dir: {OUTPUT_DIR}/")
print("=" * 60)
print("Changes vs original run_ocr.py:")
print("  [1] ENHANCED pipeline order: denoise → CLAHE → sharpen  (was CLAHE → sharpen → denoise)")
print("  [2] CLAHE clipLimit: 1.0  (was 2.0)")
print("  [3] CLEAN path: grayscale sharpening  (was BGR color sharpening)")
print("  [4] Contrast threshold: 45  (was 60)")
print("=" * 60)

for page_num in range(num_pages):
    page = doc[page_num]

    img, mode, contrast = render_page(page)
    w, h = img.size

    out_name = f"{stem}_page{page_num+1:02d}_{mode}_c{contrast:.0f}.jpg"
    out_path = os.path.join(OUTPUT_DIR, out_name)
    img.save(out_path, "JPEG", quality=92)

    print(f"  Page {page_num+1:2d}/{num_pages}  |  {mode:8s}  |  c={contrast:5.1f}  "
          f"|  {w}×{h}px  →  {out_name}")

doc.close()

print("=" * 60)
print(f"Done. All {num_pages} preprocessed images saved to: {OUTPUT_DIR}/")
print("=" * 60)