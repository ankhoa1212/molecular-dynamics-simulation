import sys
import os
import argparse

# Add current directory to path so imports work when run from elsewhere
sys.path.append(os.path.dirname(__file__))

try:
    import process
    import train
    import evaluate
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Molecular Dynamics YOLOv12 Pipeline")
    parser.add_argument(
        "stage", choices=["process", "train", "evaluate"], help="Pipeline stage to run"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    # Ensure config path is absolute or relative to this script
    if not os.path.isabs(args.config):
        args.config = os.path.join(os.path.dirname(__file__), args.config)

    if args.stage == "process":
        process.run(args.config)
    elif args.stage == "train":
        train.run(args.config)
    elif args.stage == "evaluate":
        evaluate.run(args.config)
    else:
        print("Unknown stage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
