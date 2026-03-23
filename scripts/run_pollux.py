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
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--no_calibrate", action="store_true", help="Skip automated Gaia calibration")
    
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
    print(f"Batch Size: {args.batch_size}")

    pipeline = PhotometryPipeline(args.model)
    catalog = pipeline.process_image(
        args.input, 
        threshold=args.threshold, 
        batch_size=args.batch_size,
        auto_calibrate=not args.no_calibrate
    )
    
    print(f"Detected {len(catalog)} stars.")
    pipeline.save_catalog(catalog, args.output)
    print(f"Catalog saved to {args.output}")

if __name__ == "__main__":
    main()
