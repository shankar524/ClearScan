
## ClearScan OCR

PDF-to-text OCR pipeline using **Qwen3-VL-2B** (quantized GGUF) via **llama.cpp**. Transcribes scanned documents — including handwritten text and Swedish characters — to plain text files.

---

## Docker (Recommended)

The simplest way to use ClearScan. No Python setup, no model downloads — everything is bundled in the image.

### Build

```bash
docker build -t clearscan .
```

> First build downloads the model (~1.6 GB) and llama.cpp binary. Subsequent builds use Docker cache.
> The Dockerfile is multi-platform: on x86_64 hosts it downloads a pre-built llama.cpp binary; on ARM64 (e.g. Apple Silicon) it compiles from source. The first ARM64 build takes ~5 minutes for the compilation step.

### Run

```bash
docker run --rm \
  -v /path/to/your/pdfs:/input \
  -v /path/to/results:/output \
  clearscan
```

This processes all `.pdf` files in the input directory and writes `.txt` files to the output directory.

### Custom Settings

```bash
# Use 8 threads, larger context window
docker run --rm \
  -v "$(pwd)/data":/input \
  -v "$(pwd)/output":/output \
  clearscan --threads 8 --ctx-size 8192

# Adjust temperature and max tokens
docker run --rm \
  -v "$(pwd)/data":/input \
  -v "$(pwd)/output":/output \
  clearscan --temp 0.5 --max-tokens 2000
```

### All Options

| Flag            | Env Variable            | Default | Description                     |
|-----------------|-------------------------|---------|---------------------------------|
| `-t, --threads` | `CLEARSCAN_THREADS`     | 4       | CPU threads                     |
| `--ctx-size`    | `CLEARSCAN_CTX_SIZE`    | 4096    | Context window size             |
| `--max-tokens`  | `CLEARSCAN_MAX_TOKENS`  | 1500    | Max output tokens per page      |
| `--temp`        | `CLEARSCAN_TEMP`        | 0.7     | Sampling temperature            |

### Environment Variable Overrides

```bash
docker run --rm \
  -e CLEARSCAN_THREADS=8 \
  -v "$(pwd)/data":/input \
  -v "$(pwd)/output":/output \
  clearscan
```

### Image Details

- **Size**: ~2.5 GB (1.6 GB models + base image + dependencies)
- **RAM**: ~2 GB minimum at runtime
- **CPU only** — no GPU required
- **Base**: `python:3.11-slim`
- **Multi-platform**: native builds for both `linux/amd64` and `linux/arm64`
- **llama.cpp**: release b8198 (pre-built binary on amd64, compiled from source on arm64)

---

## Local Setup (Without Docker)

```bash
# Go to your main project folder
cd path/to/your/project_folder

# Create a completely separate folder for this model test
mkdir qwen3vl_test
cd qwen3vl_test

# Create subfolders
mkdir data output models
```

Your structure:
```
project_folder/
├── (your existing paddle code)
├── nanonets_test/         ← previous test
└── qwen3vl_test/
    ├── data/              ← put your test PDFs here
    ├── output/            ← text results will appear here
    └── models/            ← GGUF weights download here
```

---

### STEP 2 — Create the Virtual Environment

```bash
python -m venv venv_qwen3vl
```

Activate it:

**Windows:**
```bash
venv_qwen3vl\Scripts\activate
```

**Linux/Mac:**
```bash
source venv_qwen3vl/bin/activate
```

You should see `(venv_qwen3vl)` in your terminal.

---

### STEP 3 — Upgrade pip

```bash
python -m pip install --upgrade pip
```

---

### STEP 4 — Install Python Dependencies

```bash
# PDF processing
pip install pymupdf
pip install Pillow
pip install numpy

# Model download utility
pip install huggingface_hub hf_transfer
```

---
**On Windows**
STEP 1 — Download the Pre-Built llama.cpp Windows Binary
Go to this URL in your browser:
https://github.com/ggml-org/llama.cpp/releases/latest
Look for a file named something like:
llama-...-x64.zip
Download that zip. 
Extract it. You will get a folder with many .exe and .dll files inside. Copy that entire extracted folder into your qwen3vl_test/ directory and rename it llama_bin:
qwen3vl_test/
├── data/
├── output/
├── models/
│   ├── Qwen3VL-2B-Instruct-Q4_K_M.gguf
│   └── mmproj-Qwen3VL-2B-Instruct-F16.gguf
├── llama_bin/              ← paste the extracted folder contents here
│   ├── llama-mtmd-cli.exe  ← this is the one we need
│   ├── llama.dll
│   ├── ggml.dll
│   └── (many other files)
├── run_ocr.py
└── venv_qwen3vl/



---

### STEP 6 — Download the GGUF Model Files

Create a file called `download_model.py` inside `qwen3vl_test/` and run it:

```python
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download

print("Downloading Qwen3-VL-2B GGUF files...")
print("Total download: ~1.6GB\n")

# Download the quantized LLM weights (~1.3GB)
print("Downloading LLM weights (Q4_K_M ~1.3GB)...")
hf_hub_download(
    repo_id="Qwen/Qwen3-VL-2B-Instruct-GGUF",
    filename="Qwen3VL-2B-Instruct-Q4_K_M.gguf",
    local_dir="models"
)

# Download the vision encoder (~300MB)
print("Downloading vision encoder (~300MB)...")
hf_hub_download(
    repo_id="Qwen/Qwen3-VL-2B-Instruct-GGUF",
    filename="mmproj-Qwen3VL-2B-Instruct-F16.gguf",
    local_dir="models"
)

print("\n✅ All files downloaded to models/ folder")
print("Files:")
for f in os.listdir("models"):
    size_mb = os.path.getsize(f"models/{f}") / (1024*1024)
    print(f"  {f}  ({size_mb:.0f} MB)")
```

Run it:
```bash
python download_model.py
```

After this your `models/` folder should contain:
```
models/
├── Qwen3VL-2B-Instruct-Q4_K_M.gguf     (~1300 MB)
└── mmproj-Qwen3VL-2B-Instruct-F16.gguf  (~300 MB)
```

---

### STEP 7 — Put Your PDFs in the Data Folder

Copy your test PDFs into `qwen3vl_test/data/`. Start with one small PDF to verify.

---

### STEP 8 — Create the OCR Script

Create `run_ocr.py` inside `qwen3vl_test/` and paste the code provided
---

### STEP 9 — Run It

```bash
# Default (reads from data/, writes to output/)
python run_ocr.py

# Custom input/output directories
python run_ocr.py --input /path/to/pdfs --output /path/to/results

# Adjust performance settings
python run_ocr.py --threads 8 --ctx-size 8192 --max-tokens 2000 --temp 0.5
```

Run `python run_ocr.py --help` for all options.

You will see:
```
Loading Qwen3-VL-2B via llama.cpp (CPU, Q4 quantized)
Model RAM usage: ~1.6GB
✅ Model loaded successfully!

Found 1 PDF(s) to process:
  - your_file.pdf

Processing: your_file.pdf
  Page 1/3  (size: 1654x2339)  ✅ done in 142.3s
```

---
