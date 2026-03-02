import os
from pathlib import Path
from rapidfuzz import fuzz
import unicodedata

# Folders (make sure these match your local names)
ground_truth_dir = Path('ground_truth_data')
extracted_dir = Path('output')

# 1. Gather files and find the longest filename to anchor the table
all_files = sorted([f for f in ground_truth_dir.glob('*.txt')])

if not all_files:
    print("No files found!")
else:
    # FIX: Normalize filenames so special characters are counted as 1 visual character
    normalized_names = [unicodedata.normalize('NFC', f.name) for f in all_files]
    
    # This finds the exact length needed for the longest normalized file
    # We add 2 extra spaces for breathing room
    max_name_len = max(len(name) for name in normalized_names)
    col_width = max(max_name_len, 20) + 2 

    # 2. Print Header
    # :<{col_width} ensures the filename column is ALWAYS the same width
    header = f"{'FILENAME':<{col_width}} | {'ACCURACY':>10}"
    print("\n")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    all_scores = []

    # 3. Process and Print
    for gt_path in all_files:
        filename = gt_path.name
        # Normalize the filename specifically for printing so the terminal aligns it perfectly
        display_name = unicodedata.normalize('NFC', filename)
        ex_path = extracted_dir / filename

        if ex_path.exists():
            with open(gt_path, 'r', encoding='utf-8') as f:
                gt_text = f.read().strip()
            with open(ex_path, 'r', encoding='utf-8') as f:
                ex_text = f.read().strip()

            score = fuzz.ratio(gt_text, ex_text)
            all_scores.append(score)
            
            # This is the magic line: the {col_width} keeps the | perfectly aligned
            print(f"{display_name:<{col_width}} | {score:>9.2f}%")
        else:
            print(f"{display_name:<{col_width}} | {'MISSING':>10}")

    # 4. Final Footer
    if all_scores:
        avg_accuracy = sum(all_scores) / len(all_scores)
        print("-" * len(header))
        print(f"{'OVERALL AVERAGE':<{col_width}} | {avg_accuracy:>9.2f}%")
    print("-" * len(header))