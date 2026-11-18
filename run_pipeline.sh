#!/usr/bin/env sh
# Rebuild everything from the raw data pack.  Usage: DATA_DIR="/path/to/Data Pack/" sh run_pipeline.sh
set -e
ROOT="$(pwd)"
export DATA_DIR="${DATA_DIR:-$ROOT/raw_data/Data Pack/}"
export OUT_DIR="${OUT_DIR:-$ROOT/pipeline/out/}"
mkdir -p "$OUT_DIR"
cd pipeline
for s in 01_build_state 02_pressure_model 03_ratings 04_robustness 05_consensus 07_variants 06_export; do echo "== $s"; python3 $s.py; done
python3 08_report.py
cd ..
python3 pipeline/add_names.py
