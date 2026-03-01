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



# =================================================
# 2. DOCUMENT INGESTION (LOOPING THROUGH FILES)
# =================================================
for filename in os.listdir(INPUT_DIR):
    # Skip any files in the folder that aren't PDFs
    if not filename.lower().endswith('.pdf'):
        continue 

    print(f"Processing: {filename}...")
    
    # Getting the exact full path to the current PDF
    pdf_path = os.path.join(INPUT_DIR, filename)
    
    # Converts the PDF into a list of high-quality images (300 dots-per-inch) using PyMuPDF (fitz)
    pdf_document = fitz.open(pdf_path)

    # An empty list to hold all the text we find in this specific document
    document_text = []



    # ===================================
    # 3. PAGE BY PAGE PROCESSING
    # ===================================
    for page_num in range(len(pdf_document)):
        print(f"  -> Reading page {page_num + 1}...")
        
        # Grabs the current page
        page = pdf_document[page_num]
        
        # PDFs default to 72 DPI. W're zooming in to 250 DPI to get a clearer image for OCR.
        zoom = 250 / 72
        mat = fitz.Matrix(zoom, zoom)
        
        # Renders the page to a pixel map (image)
        pix = page.get_pixmap(matrix=mat)

        # Converting the PyMuPDF image directly into a NumPy array for OpenCV
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        # PyMuPDF might output RGB (3 channels) or RGBA (4 channels). We need BGR for OpenCV.
        if pix.n == 4:
            doc_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:
            doc_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:
            doc_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

        # Saving the very first page as a debug image
        if page_num == 0:
            cv2.imwrite(f'{DEBUG_DIR}/{filename}_page1.png', doc_bgr)



        # ============================================
        # 4. RUNNING INFERENCE (EXTRACTING THE TEXT)
        # ============================================
        # cls=True tells PaddleOCR to actively check and fix the rotation of the text.
        result = ocr.ocr(doc_bgr, cls=True)

        # PaddleOCR returns a complex list of boxes, text, and confidence scores. 
        # We're just gonna pull out just the text strings.
        page_text = ""
        
        # 'result' can be None if the page is completely blank, so we're checking if it exists first.
        if result and result[0]: 
            for line in result[0]:
                text_string = line[1][0]
                page_text += text_string + "\n"
        
        # Adding extracted text from this page to our document-level list.
        document_text.append(f"--- PAGE {page_num + 1} ---\n{page_text}")

        # Using Python's GC for freeing up memory after processing each page
        del pix
        del img_array
        del doc_bgr
        del result
        gc.collect()


    # ============================
    # 5. SAVING THE OUTPUT
    # ============================
    # Changing the file extension from .pdf to .txt for saving
    if filename.endswith('.pdf'):
        output_filename = filename.replace('.pdf', '.txt')
    
    if filename.endswith('.PDF'):
        output_filename = filename.replace('.PDF', '.txt')

    output_path = os.path.join(OUTPUT_DIR, output_filename)

    # Writing all the collected text into the text file and saving it in the output folder
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(document_text))
        
    print(f"Finished! Saved text to {output_path}\n")
    
    pdf_document.close()  # Closing the PDF file to free up memory


