import argparse
import os
import sys
import yaml
from pollux.pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser(description="Pollux: Automated Roman Photometry Pipeline")
    parser.add_argument("--config", help="Path to the YAML pipeline configuration file")
    
    # Keep some old arguments for convenience, but they will override or be merged into config
    parser.add_argument("input", nargs="?", help="Path to the input Roman Level 2 ASDF file")
    parser.add_argument("--model", help="Path to the ONNX model file")
    parser.add_argument("--output", help="Path to save the output catalog")
    
    args = parser.parse_args()

    if args.config:
        config_path = args.config
        if not os.path.exists(config_path):
            print(f"Error: Config file not found at {config_path}")
            sys.exit(1)
        
        print(f"Running Pollux with config: {config_path}")
        pipeline = Pipeline(config_path)
        
        # Override config with CLI arguments if provided
        if args.input:
            for step in pipeline.config['pipeline']:
                if step['type'] == 'load_image':
                    step['path'] = args.input
        
        if args.model:
            for step in pipeline.config['pipeline']:
                if step['type'] == 'photometry':
                    step['model_path'] = args.model

        if args.output:
            for step in pipeline.config['pipeline']:
                if step['type'] == 'save_catalog':
                    step['path'] = args.output

    elif args.input and args.model:
        # Create a default config if no config file is provided but input/model are
        print("No config file provided. Using default pipeline with CLI arguments.")
        default_config = {
            'pipeline': [
                {'type': 'load_image', 'path': args.input},
                {'type': 'photometry', 'model_path': args.model, 'threshold': 0.5, 'batch_size': 16},
                {'type': 'calibrate', 'radius_arcsec': 1.0},
                {'type': 'save_catalog', 'path': args.output or "output_catalog.asdf"}
            ]
        }
        
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
            yaml.dump(default_config, f)
            config_path = f.name
        
        try:
            pipeline = Pipeline(config_path)
        finally:
            os.remove(config_path)
    else:
        parser.print_help()
        sys.exit(1)

    catalog = pipeline.run()
    if catalog is not None:
        print(f"Pipeline complete. Detected {len(catalog)} stars.")
    else:
        print("Pipeline complete. No catalog generated.")

if __name__ == "__main__":
    main()
