import argparse
import os
import sys
from pollux.pipeline import PhotometryPipeline

def main():
    parser = argparse.ArgumentParser(description="Pollux: Automated Roman Photometry Pipeline")
    parser.add_argument("input", help="Path to the input Roman Level 2 ASDF file")
    parser.add_argument("--model", default="models/stage0/stage0_epoch_12.pth", help="Path to the model weights")
    parser.add_argument("--output", default="output_catalog.asdf", help="Path to save the output catalog")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        sys.exit(1)

    print(f"Starting Pollux pipeline...")
    print(f"Input: {args.input}")
    print(f"Model: {args.model}")
    print(f"Threshold: {args.threshold}")

    pipeline = PhotometryPipeline(args.model)
    catalog = pipeline.process_image(args.input, threshold=args.threshold)
    
    print(f"Detected {len(catalog)} stars.")
    
    pipeline.save_catalog(catalog, args.output)
    print(f"Catalog saved to {args.output}")

if __name__ == "__main__":
    main()
