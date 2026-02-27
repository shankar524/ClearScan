import os
# Disable MKL-DNN to prevent segfaults during model download
os.environ['FLAGS_use_mkldnn'] = '0'

from paddleocr import PaddleOCR

# This triggers the model download for offline use
print("Downloading Swedish and French models...")
PaddleOCR(use_angle_cls=True, lang='sv')
PaddleOCR(use_angle_cls=True, lang='fr')
print("Models baked in successfully.")