#!/bin/sh
set -e

# ─── Pass-through for --help ─────────────────────────────────
case "$1" in
    -h|--help)
        exec python main.py --help
        ;;
esac

# ─── Validate mount points ───────────────────────────────────
if [ ! -d "/input" ]; then
    echo "ERROR: /input directory not found."
    echo "Mount your PDF directory:  docker run -v /path/to/pdfs:/input ..."
    exit 1
fi

if [ ! -d "/output" ]; then
    echo "ERROR: /output directory not found."
    echo "Mount your output directory:  docker run -v /path/to/results:/output ..."
    exit 1
fi

# Count PDFs
pdf_count=$(find /input -maxdepth 1 -iname '*.pdf' | wc -l)
if [ "$pdf_count" -eq 0 ]; then
    echo "ERROR: No PDF files found in /input."
    exit 1
fi

echo "============================================================"
echo " ClearScan OCR"
echo " Input:  /input  ($pdf_count PDF(s))"
echo " Output: /output"
echo "============================================================"

# Forward all extra CLI arguments (e.g. --threads 8, --ctx-size 8192)
exec python main.py --input /input --output /output "$@"
