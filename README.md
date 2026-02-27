# ClearScan: Softwerk OCR Challenge

## Environment Setup

### 1. Create a Virtual Environment
Follow the steps for your specific Operating System:

#### **Windows**
```powershell
# Create the environment
python -m venv .venv
# Activate the environment
.\.venv\Scripts\activate
```


#### **MacOS / Linux**
```
# Create the environment
python3 -m venv .venv
# Activate the environment
source .venv/bin/activate
```


#### Install Dependencies
```
pip install --upgrade pip
pip install -r requirements.txt
```


## The "Forensic" Pipeline

### 1. Pre-processing Pipeline

To ensure maximum accuracy on faded scans, we developed a custom pre-processing pipeline before passing data to the AI:

    300 DPI Rendering: High-resolution PDF-to-Image conversion to capture fine typewriter details.

    Adaptive Thresholding: Handles local lighting variations and shadows in old photocopies.

    Deskewing (Straightening): Automatically detects the text angle and rotates the document to a perfect 0° horizon.

    Median Denoising: A 3x3 kernel filter that "washes away" scanner dust and salt-and-pepper noise while preserving text edges.

    Memory Management: Explicit garbage collection and pointer deletion to handle 30+ page documents on consumer hardware.


## OCR Model

We use PaddleOCR with the following configuration:

    Language: Swedish and French (dynamically selected based on filename).

    Angle Classification: Enabled to handle rotated documents.

    Model Caching: Models are downloaded during the build process to ensure offline functionality.