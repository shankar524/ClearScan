# from pdf2image import convert_from_path
# import os
# import numpy as np
# import cv2

# def deskew(image):
#     inverted = cv2.bitwise_not(image)

#     coords = cv2.findNonZero(inverted)
#     angle = cv2.minAreaRect(coords)[-1]

#     # Adjust angle logic
#     if angle < -45:
#         angle += 90

#     # Rotate the image to straighten it
#     (h, w) = image.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     straightened = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#     return straightened
    

# path = 'data'
# filenames = []

# for filename in os.listdir('data'):
#     filenames.append(filename)

# datasets = np.array(filenames)

# os.makedirs('debug', exist_ok=True)

# for i in range(len(filenames)):
#     pdf_path = os.path.join(path, datasets[i])
#     pages = convert_from_path(pdf_path, dpi=300)

#     for page_num, pil_image in enumerate(pages):
#         doc = np.array(pil_image)

#         # Convert RGB to BGR for OpenCV
#         doc = cv2.cvtColor(doc, cv2.COLOR_RGB2BGR)
#         gray = cv2.cvtColor(doc, cv2.COLOR_BGR2GRAY)

#         # Adaptive thresholding to handle local light variations
#         thresh = cv2.adaptiveThreshold(
#             gray,
#             maxValue=255,
#             adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             thresholdType=cv2.THRESH_BINARY,
#             blockSize=11,
#             C=2
#         )

#         # Deskew (straighten) the thresholded image
#         straightened = deskew(thresh)

#         # Remove salt and pepper noise with median blur
#         cleaned = cv2.medianBlur(straightened, 3)

#         # Save a sample debug image from the first page of the first document (for my testing)
#         if i == 0 and page_num == 0:
#             cv2.imwrite('debug/sample_thresh.png', cleaned)

#         # TODO: Insert OCR model inference here

#         print(f"File: {datasets[i]} | Page: {page_num + 1} | Shape: {cleaned.shape}")

#         # Free up memory before the next page or file
#         del doc
#         del gray
#         del thresh
#         del straightened
#         del cleaned

# # Pre-processing pipeline 