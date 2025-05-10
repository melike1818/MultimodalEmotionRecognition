"""
evaluation.py

This module implements the Evaluation class for the MER-HAN multimodal emotion recognition model.
It iterates over the test dataset, collects predictions and ground truth labels, and computes evaluation
metrics including Weighted Average Recall (WAR), Unweighted Average Recall (UAR), macro F1-score, weighted F1-score,
and the confusion matrix using scikit-learn's metrics. These metrics follow the evaluation protocol specified 
in the paper "Multimodal emotion recognition based on audio and text by using hybrid_att".

Usage:
    evaluation = Evaluation(model, test_loader, config)
    metrics_dict = evaluation.evaluate()
"""

import torch
import numpy as np
from typing import Dict, Any
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
import logging

# Configure logging for evaluation module.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluation:
    """
    Evaluation class for the MER-HAN model.

    Attributes:
        model (torch.nn.Module): The MER-HAN model instance.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        config (Dict[str, Any]): Configuration dictionary loaded from config.yaml.
        device (torch.device): Device for evaluation (cpu or cuda).
    """
    def __init__(self, model: torch.nn.Module, test_loader: torch.utils.data.DataLoader, config: Dict[str, Any]) -> None:
        """
        Initialize the Evaluation instance.

        Args:
            model: An instance of the MERHAN model.
            test_loader: A DataLoader for the test dataset.
            config: A configuration dictionary loaded from config.yaml.
        """
        self.model = model
        self.test_loader = test_loader
        self.config = config

        # Retrieve and set device from configuration. Default to "cuda" if available.
        device_str: str = config.get("hardware", {}).get("device", "cuda")
        self.device: torch.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Evaluation will run on device: {self.device}")

    def evaluate(self) -> Dict[str, any]:
        """
        Evaluate the model on the test dataset and compute metrics.

        Returns:
            A dictionary containing the following keys:
                - "WAR": Weighted Average Recall (accuracy score).
                - "UAR": Unweighted Average Recall (macro average recall).
                - "F1_score": Macro F1-score.
                - "weighted_F1_score": Weighted F1-score.
                - "confusion_matrix": Confusion matrix as a numpy array.
        """
        self.model.eval()  # Set model to evaluation mode
        ground_truth_labels = []  # type: list
        predicted_labels = []     # type: list

        # Disable gradient computations for evaluation.
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating", unit="batch"):
                # Move audio tensor to the device.
                audio: torch.Tensor = batch["audio"].to(self.device)  # shape: (batch, seq_len, feature_dim)

                # Prepare text inputs and move to device.
                text_input_ids: torch.Tensor = batch["text_input_ids"].to(self.device)
                text_attention_mask: torch.Tensor = batch["text_attention_mask"].to(self.device)
                text_inputs: Dict[str, torch.Tensor] = {
                    "text_input_ids": text_input_ids,
                    "text_attention_mask": text_attention_mask
                }

                # Move labels to device.
                labels: torch.Tensor = batch["label"].to(self.device)

                # Forward pass: obtain output logits from the model.
                outputs: torch.Tensor = self.model(audio, text_inputs)  # shape: (batch, num_classes)

                # Compute predicted class using argmax over the class dimension.
                preds: torch.Tensor = torch.argmax(outputs, dim=1)

                # Append current batch predictions and ground truth to lists.
                ground_truth_labels.extend(labels.cpu().numpy().tolist())
                predicted_labels.extend(preds.cpu().numpy().tolist())

        # Convert collected labels to numpy arrays.
        gt_labels_np: np.ndarray = np.array(ground_truth_labels)
        pred_labels_np: np.ndarray = np.array(predicted_labels)

        # Compute evaluation metrics.
        # WAR: overall accuracy, equivalent to weighted average recall.
        war: float = accuracy_score(gt_labels_np, pred_labels_np)
        # UAR: unweighted average recall (macro average recall).
        uar: float = recall_score(gt_labels_np, pred_labels_np, average="macro", zero_division=0)
        # Macro F1-score.
        macro_f1: float = f1_score(gt_labels_np, pred_labels_np, average="macro", zero_division=0)
        # Weighted F1-score.
        weighted_f1: float = f1_score(gt_labels_np, pred_labels_np, average="weighted", zero_division=0)
        # Confusion Matrix.
        conf_matrix: np.ndarray = confusion_matrix(gt_labels_np, pred_labels_np)

        # Organize metrics into a dictionary.
        metrics_dict: Dict[str, any] = {
            "WAR": war,
            "UAR": uar,
            "F1_score": macro_f1,
            "weighted_F1_score": weighted_f1,
            "confusion_matrix": conf_matrix
        }

        # Log computed metrics.
        logger.info("Evaluation Metrics:")
        logger.info(f"WAR (Accuracy): {war:.4f}")
        logger.info(f"UAR (Macro Recall): {uar:.4f}")
        logger.info(f"Macro F1 score: {macro_f1:.4f}")
        logger.info(f"Weighted F1 score: {weighted_f1:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")

        return metrics_dict
