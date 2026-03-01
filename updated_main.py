import os
import cv2
import numpy as np
import fitz  # PyMuPDF (i removes teh pdf2image bcoz of "poppler" issues)
from paddleocr import PaddleOCR
import gc  # Garbage collector to free up memory after processing each document


# =====================================
# 1. INITIALIZING THE OCR MODEL
# =====================================
# use_angle_cls=True: Turns on the automatic rotation correction
# lang='sv': Sets the language to Swedish for the Palme archives. 
# show_log=False: Stops PaddleOCR from flooding your terminal with diagnostic text.

ocr = PaddleOCR(use_angle_cls=True, lang='sv', show_log=False)

# Initiating our main folders
INPUT_DIR = 'data'
OUTPUT_DIR = 'output'
DEBUG_DIR = 'debug'

# Creating the output and debug folders if they don't already exist on your computer
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)


