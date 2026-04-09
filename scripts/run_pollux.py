import argparse
import os
import sys
import yaml
import glob
from tqdm import tqdm
from pollux.pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser(description="Pollux: Automated Roman Photometry Pipeline")
    parser.add_argument("--config", help="Path to the YAML pipeline configuration file")
    parser.add_argument("input", nargs="*", help="Optional: Override input path(s) or glob pattern")
    parser.add_argument("--model", help="Optional: Override ONNX model path")
    parser.add_argument("--output", help="Optional: Override output path")
    
    args = parser.parse_args()

    if not args.config:
        print("Error: --config <path.yaml> is required.")
        sys.exit(1)

    if not os.path.exists(args.config):
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    pipeline = Pipeline(args.config)
    batch_cfg = pipeline.config.get('batch')
    
    # 1. Determine input files
    input_files = []
    if args.input:
        for pattern in args.input:
            expanded = glob.glob(pattern)
            if expanded: input_files.extend(expanded)
            else: input_files.append(pattern)
    elif batch_cfg:
        input_dir = batch_cfg.get('input_dir', '.')
        pattern = batch_cfg.get('pattern', '*.fits')
        search_path = os.path.join(input_dir, pattern)
        input_files = glob.glob(search_path)
    else:
        for step in pipeline.config['pipeline']:
            if step['type'] == 'load_image' and step.get('path'):
                input_files = [step['path']]
                break

    if not input_files:
        print("Error: No input files found.")
        sys.exit(1)

    # 2. Global Overrides & Batch Preparation
    if args.model:
        for step in pipeline.config['pipeline']:
            if step['type'] == 'photometry':
                step['model_path'] = args.model

    output_dir = batch_cfg.get('output_dir') if batch_cfg else None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # If batch mode, make internal photometry step quiet to avoid messy output
    is_batch = len(input_files) > 1
    if is_batch:
        for step in pipeline.config['pipeline']:
            if step['type'] == 'photometry':
                step['quiet'] = True

    # 3. Process files with progress bar
    print(f"Processing {len(input_files)} images...")
    pbar = tqdm(input_files, desc="Batch Progress") if is_batch else input_files

    for input_path in pbar:
        if not os.path.exists(input_path):
            continue

        if is_batch:
            pbar.set_description(f"Processing {os.path.basename(input_path)}")

        # Determine output path
        current_output = None
        if args.output:
            current_output = args.output
        elif output_dir:
            basename = os.path.splitext(os.path.basename(input_path))[0]
            current_output = os.path.join(output_dir, f"{basename}_catalog.asdf")
        
        try:
            pipeline.run(input_path=input_path, output_path=current_output)
        except Exception as e:
            if is_batch:
                print(f"\nError processing {input_path}: {e}")
            else:
                raise e

    pipeline.finalize()
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
