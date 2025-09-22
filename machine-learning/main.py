import sys
import argparse
import processing
import training
import evaluation

def main():
    parser = argparse.ArgumentParser(description="Molecular Dynamics ML Pipeline")
    parser.add_argument('stage', choices=['process', 'train', 'evaluate'], help='Pipeline stage to run')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()

    if args.stage == 'process':
        processing.run(args.config)
    elif args.stage == 'train':
        training.run(args.config)
    elif args.stage == 'evaluate':
        evaluation.run(args.config)
    else:
        print("Unknown stage.")
        sys.exit(1)

if __name__ == "__main__":
    main()