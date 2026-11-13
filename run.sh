#!/usr/bin/env bash
set -e

mkdir -p artifacts

echo "Step 1: Compiling CUDA image processor..."
make build

echo "Step 2: Generating 200 noisy input images (256x256)..."
python3 scripts/generate_dataset.py --mode generate --input_dir input_data --count 200 --width 256 --height 256

echo "Step 3: Running GPU shared-memory pipeline (Gaussian + Sobel) on all 200 images..."
./bin/image_processor --input_dir input_data --output_dir output_data --filter pipeline | tee artifacts/execution.log

echo "Step 4: Exporting sample before/after PNGs and packaging proof_of_execution.tar.gz..."
python3 scripts/generate_dataset.py --mode export_png --input_dir input_data --output_dir output_data --artifact_dir artifacts
tar -czvf proof_of_execution.tar.gz artifacts/

echo "Done! Upload 'proof_of_execution.tar.gz' to Coursera."
