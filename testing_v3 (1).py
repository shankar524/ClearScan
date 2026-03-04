import os
import re
import jiwer

# Determine the directory of this script and define the output file path
script_dir = os.path.dirname(os.path.abspath(__file__))
results_file_path = os.path.join(script_dir, 'evaluation_results.txt')

# Open the text file in write mode ('w')
results_file = open(results_file_path, 'w', encoding='utf-8')

def log_print(msg=""):
    """Helper function to print to the console AND write to the text file."""
    print(msg)
    results_file.write(str(msg) + "\n")

GT_DIR = 'ground_truth_data'
OUTPUT_DIR = 'output'

# Ensure the output directory exists so os.listdir does not fail
os.makedirs(OUTPUT_DIR, exist_ok=True)

log_print("--- Starting Advanced Page-Level OCR Evaluation ---\n")

def split_into_pages(text):
    """
    Split text into pages using the page markers.

    Supports both:
    - Ground truth style:  "--- PAGE 1 ---"
    - Qwen output style:   "--- PAGE 1/3  [..meta..] ---"
    """
    pattern = r'--- PAGE \d+(?:/\d+)?(?:\s+\[.*?\])?\s*---'

    # Split on page markers
    pages = re.split(pattern, text)

    # Drop everything before the first page marker (headers, metadata, etc.)
    if len(pages) > 1:
        pages = pages[1:]
    else:
        pages = []

    return [p.strip() for p in pages]

for filename in os.listdir(GT_DIR):
    if not filename.endswith('.txt'):
        continue

    gt_path = os.path.join(GT_DIR, filename)

    # Find all output files that correspond to this ground truth file.
    # This supports multiple variants like "filename.txt" and "v1-filename.txt".
    matching_outputs = [
        out_name
        for out_name in os.listdir(OUTPUT_DIR)
        if out_name == filename or out_name.endswith(filename)
    ]

    if not matching_outputs:
        log_print(f"⚠️ Warning: Missing output file for {filename}")
        continue

    # Load the ground truth text once
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_text = f.read()

    for output_name in matching_outputs:
        output_path = os.path.join(OUTPUT_DIR, output_name)

        # Load the OCR output text
        with open(output_path, 'r', encoding='utf-8') as f:
            out_text = f.read()

        # Split texts into lists of pages
        gt_pages = split_into_pages(gt_text)
        out_pages = split_into_pages(out_text)

        log_print(f"\n=======================================================")
        log_print(f"📄 Analyzing File: {filename}")
        log_print(f"   Using output: {output_name}")
        log_print(f"=======================================================")

        # Find out how many pages we can safely compare
        num_pages = min(len(gt_pages), len(out_pages))

        if num_pages == 0:
            log_print("  ⚠️ Could not detect '--- PAGE X ---' markers. Ensure your ground truth uses them!")
            continue

        for i in range(num_pages):
            gt_page_text = gt_pages[i]
            out_page_text = out_pages[i]

            # Skip if the ground truth page is completely blank
            if not gt_page_text:
                continue

            # -----------------------------------------
            # 1. MISSING WORDS CALCULATION
            # -----------------------------------------
            # We convert both pages to lowercase and turn them into sets of unique words.
            # Subtracting the output set from the GT set leaves only the words the OCR missed.
            gt_words = set(gt_page_text.lower().split())
            out_words = set(out_page_text.lower().split())

            missing_words = gt_words - out_words
            missing_count = len(missing_words)

            # -----------------------------------------
            # 2. PAGE-LEVEL ACCURACY (CER & WER)
            # -----------------------------------------
            cer_score = jiwer.cer(gt_page_text, out_page_text)
            wer_score = jiwer.wer(gt_page_text, out_page_text)

            cer_acc = max(0, (1 - cer_score) * 100)
            wer_acc = max(0, (1 - wer_score) * 100)

            # -----------------------------------------
            # 3. PRINTING THE ANALYSIS
            # -----------------------------------------
            log_print(f"  -> Page {i + 1}:")
            log_print(f"     • Accuracy: {cer_acc:.1f}% CER | {wer_acc:.1f}% WER")
            log_print(f"     • Missing Words: {missing_count} (out of {len(gt_words)} unique GT words)")

            # If words are missing, print up to 5 examples so you can see exactly what it's failing on
            if missing_count > 0:
                example_missing = list(missing_words)[:5]
                log_print(f"     • Example misses: {example_missing}")

# Close the file safely when the loop is done
log_print(f"\n✅ Evaluation complete. Results saved locally to: evaluation_results.txt")
results_file.close()