import os
import jiwer
import unicodedata

GT_DIR = 'ground_truth_data'
OUTPUT_DIR = 'output'

wer_list = []
cer_list = []

all_references = []
all_hypotheses = []

print("--- Starting OCR Evaluation ---\n")

# 1. Pre-scan to find valid files
valid_files = []
if os.path.exists(GT_DIR):
    for filename in os.listdir(GT_DIR):
        if filename.endswith('.txt'):
            output_path = os.path.join(OUTPUT_DIR, filename)
            if os.path.exists(output_path):
                valid_files.append(filename)

if not valid_files:
    print("No matching file pairs found in the directories.")
    exit()

# FIX: Normalize filenames so 'å', 'ä', 'ö' are correctly counted as 1 character
normalized_names = [unicodedata.normalize('NFC', f) for f in valid_files]

# Find the maximum visual length
max_name_len = max([len(f) for f in normalized_names] + [len("Filename")])
col_width = max_name_len + 2  # Add 2 spaces of padding

# 2. Print the dynamic table header
header = f"{'Filename':<{col_width}} | {'CER (%)':>8} | {'WER (%)':>8}"
print(header)
print("-" * len(header))

# 3. Process the files
for filename in valid_files:
    gt_path = os.path.join(GT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    with open(gt_path, 'r', encoding='utf-8') as f:
        reference = f.read().strip()
        
    with open(output_path, 'r', encoding='utf-8') as f:
        hypothesis = f.read().strip()

    if not reference:
        continue
        
    # Calculate File-Level Accuracy
    wer_score = jiwer.wer(reference, hypothesis)
    cer_score = jiwer.cer(reference, hypothesis)

    wer_accuracy = max(0, (1 - wer_score) * 100)
    cer_accuracy = max(0, (1 - cer_score) * 100)

    wer_list.append(wer_accuracy)
    cer_list.append(cer_accuracy)

    all_references.append(reference)
    all_hypotheses.append(hypothesis)

    # Normalize the filename for printing so it visually aligns in the terminal
    display_name = unicodedata.normalize('NFC', filename)

    print(f"{display_name:<{col_width}} | {cer_accuracy:>7.2f}% | {wer_accuracy:>7.2f}%")

# ==========================================
# FINAL CALCULATIONS & COMPARISON
# ==========================================
if all_references:
    overall_wer_score = jiwer.wer(all_references, all_hypotheses)
    overall_cer_score = jiwer.cer(all_references, all_hypotheses)

    overall_wer_accuracy = max(0, (1 - overall_wer_score) * 100)
    overall_cer_accuracy = max(0, (1 - overall_cer_score) * 100)
    
    avg_cer = sum(cer_list) / len(cer_list)
    avg_wer = sum(wer_list) / len(wer_list)
    
    print(f"\n{'=' * len(header)}")
    print(f"🏆 FINAL PIPELINE ACCURACY COMPARISON 🏆")
    print(f"{'=' * len(header)}")
    
    # Align the final output labels with the dynamic filename column
    print(f"{'Average of Files (Macro)':<{col_width}} | CER: {avg_cer:>5.2f}% | WER: {avg_wer:>5.2f}%")
    print(f"{'Overall Corpus (Micro)':<{col_width}} | CER: {overall_cer_accuracy:>5.2f}% | WER: {overall_wer_accuracy:>5.2f}%")
    print(f"{'=' * len(header)}")