"""test_only.py

Run evaluation on the test set using a trained checkpoint (default: checkpoints/best_model.pth).

Usage (from project root):
    python -m MultimodalEmotionRecognition.test_only \
        --config config.yaml \
        --dataset IEMOCAP \
        --checkpoint checkpoints/best_model.pth

The script loads the config, initialises the dataset loader (test split only),
reconstructs the MERHAN model, loads the checkpoint weights, and computes
WAR, UAR, macro-F1, weighted-F1, and the confusion matrix.
"""

import os
import argparse
import random
import yaml
import numpy as np
import torch

# Local imports (assumes PYTHONPATH includes project root)
from dataset_loader import DatasetLoader
from model import MERHAN
from evaluation import Evaluation


def load_config(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found at {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser("Evaluate a trained MER-HAN checkpoint on the test set.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dataset", type=str, choices=["IEMOCAP", "MELD"], default="IEMOCAP", help="Dataset to use")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to the model checkpoint (.pth)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Configuration & reproducibility
    # ------------------------------------------------------------------
    config = load_config(args.config)
    set_seed(config.get("random_seed", 42))

    # ------------------------------------------------------------------
    # Prepare data (only need the test loader)
    # ------------------------------------------------------------------
    dataset_type = args.dataset.upper()
    loader = DatasetLoader(config=config, dataset_type=dataset_type)
    _, _, test_loader = loader.load_data()

    # ------------------------------------------------------------------
    # Build model skeleton
    # ------------------------------------------------------------------
    classification_cfg = config.get("model", {}).get("classification", {})
    if dataset_type == "IEMOCAP":
        num_classes = classification_cfg.get("num_classes_IEMOCAP", 4)
    else:
        num_classes = classification_cfg.get("num_classes_MELD", 7)

    model = MERHAN(config=config, num_classes=num_classes)

    # ------------------------------------------------------------------
    # Load checkpoint weights
    # ------------------------------------------------------------------
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    print(f"Loaded checkpoint from '{args.checkpoint}' (epoch {checkpoint.get('epoch', 'N/A')})")

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    evaluator = Evaluation(model=model, test_loader=test_loader, config=config)
    metrics = evaluator.evaluate()

    # Pretty print metrics.
    print("\nTest-set Metrics:")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            print(f"{k}:\n{v}")
        else:
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main() 