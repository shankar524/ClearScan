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