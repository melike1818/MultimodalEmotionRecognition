"""main.py

Entry point for the MER-HAN multimodal emotion recognition project.
This script loads the configuration from config.yaml, sets up the random seeds and device,
prepares the datasets using the DatasetLoader, initializes the MERHAN model,
trains the model using the Trainer, and finally evaluates it using the Evaluation module.
"""

import os
import sys
import argparse
import random
import yaml
import numpy as np
import torch

# Import submodules from the project.
from dataset_loader import DatasetLoader
from model import MERHAN
from trainer import Trainer
from evaluation import Evaluation


def load_config(config_path: str) -> dict:
    """
    Loads the configuration from a YAML file.

    Args:
        config_path: Path to the config.yaml file.

    Returns:
        A configuration dictionary.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_global_seed(seed: int) -> None:
    """
    Sets the random seed for reproducibility in Python, NumPy, and PyTorch.

    Args:
        seed: The random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_arguments() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        An argparse.Namespace containing parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="MER-HAN: Multimodal Emotion Recognition based on Audio and Text using Hybrid Attention Networks."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["IEMOCAP", "MELD"],
        default="IEMOCAP",
        help="Dataset to use: IEMOCAP or MELD (default: IEMOCAP)."
    )
    return parser.parse_args()


def main():
    # Parse command-line arguments.
    args = parse_arguments()

    # Load configuration.
    config_path: str = args.config
    config: dict = load_config(config_path)

    # Set global random seed.
    seed: int = config.get("random_seed", 42)
    set_global_seed(seed)

    # Determine the device.
    hardware_config: dict = config.get("hardware", {})
    device_str: str = hardware_config.get("device", "cuda")
    device: torch.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize DatasetLoader with the chosen dataset type.
    dataset_type: str = args.dataset.upper()
    dataset_loader = DatasetLoader(config=config, dataset_type=dataset_type)
    train_loader, val_loader, test_loader = dataset_loader.load_data()

    # Determine number of classes based on dataset type.
    classification_config: dict = config.get("model", {}).get("classification", {})
    if dataset_type == "IEMOCAP":
        num_classes: int = classification_config.get("num_classes_IEMOCAP", 4)
    elif dataset_type == "MELD":
        num_classes: int = classification_config.get("num_classes_MELD", 7)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    # Initialize the MERHAN model.
    model = MERHAN(config=config, num_classes=num_classes)
    model.to(device)  # Ensure model is on the correct device.

    # Instantiate Trainer and start training.
    trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config)
    trainer.train()

    # After training, evaluate the model on the test set.
    evaluator = Evaluation(model=model, test_loader=test_loader, config=config)
    metrics = evaluator.evaluate()

    # Print evaluation metrics.
    print("\nEvaluation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")


if __name__ == "__main__":
    main()
