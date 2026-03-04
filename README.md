
## FULL STEP-BY-STEP GUIDE: Qwen3-VL-2B via llama.cpp

---

### STEP 1 — Create the Isolated Folder

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
python run_ocr.py
```

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
