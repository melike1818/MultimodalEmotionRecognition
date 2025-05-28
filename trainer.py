"""trainer.py

This module implements the Trainer class for the MER-HAN multimodal emotion recognition model.
The Trainer is responsible for training the model using the training DataLoader and evaluating it on
the validation DataLoader. It follows the configuration specified in config.yaml, using the Adam 
optimizer (lr=0.0001), CrossEntropyLoss, and performs training for a specified number of epochs (default 64).

Usage:
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import Dict, Any
import wandb  # Weights & Biases for experiment tracking

class Trainer:
    """Trainer class that manages the training loop for the MER-HAN model."""
    def __init__(self, model: nn.Module, train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader, config: Dict[str, Any]) -> None:
        """
        Initializes the Trainer.

        Args:
            model: An instance of the MERHAN model.
            train_loader: DataLoader for training data.
            val_loader: DataLoader for validation data.
            config: Dictionary containing configuration settings from config.yaml.
        """
        self.config = config

        # Set device from config; use cuda if available and requested, else cpu.
        device_str = config.get("hardware", {}).get("device", "cuda")
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Training hyperparameters.
        self.learning_rate: float = float(config["training"].get("learning_rate", 0.0001))
        self.epochs: int = int(config["training"].get("epochs", 64))
        self.batch_size: int = int(config["training"].get("batch_size", 32))
        self.random_seed: int = int(config.get("random_seed", 42))

        # Early stopping: number of epochs with no improvement after which training stops.
        self.early_stop_patience: int = int(config["training"].get("early_stop_patience", 10))
        self.epochs_no_improve: int = 0  # Counter for epochs without validation loss improvement.

        # Set loss function and optimizer.
        self.criterion = nn.CrossEntropyLoss()
        # Optimizer should receive only parameters that require gradients
        trainable_params = (p for p in self.model.parameters() if p.requires_grad)
        self.optimizer = optim.Adam(trainable_params, lr=self.learning_rate)

        # DataLoaders.
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Directory for saving checkpoints.
        self.checkpoint_dir = "checkpoints"
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize best validation loss for checkpointing.
        self.best_val_loss = float('inf')

        # --------------------
        # Weights & Biases setup
        # --------------------
        # Enable wandb logging only if requested in the config.
        self.use_wandb: bool = self.config.get("wandb", {}).get("use", False)
        if self.use_wandb:
            wandb_init_kwargs = {
                "project": self.config.get("wandb", {}).get("project", "MER-HAN"),
                "name": self.config.get("wandb", {}).get("run_name", None),
                "config": self.config,
            }
            self.wandb_run = wandb.init(**wandb_init_kwargs)
            # Track gradients and model parameters.
            wandb.watch(self.model, log="all")

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """
        Saves the model checkpoint if the validation loss improves.

        Args:
            epoch: Current epoch number.
            val_loss: Validation loss for the current epoch.
        """
        checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pth")
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss
        }
        torch.save(state, checkpoint_path)
        # Save (and optionally upload) checkpoint to wandb.
        if getattr(self, "use_wandb", False):
            wandb.save(checkpoint_path)
        print(f"Checkpoint saved at epoch {epoch} with validation loss: {val_loss:.4f}")

    def validate(self) -> float:
        """
        Evaluates the model on the validation set.

        Returns:
            The average validation loss.
        """
        self.model.eval()
        val_loss_total: float = 0.0
        total_samples: int = 0
        with torch.no_grad():
            for batch in self.val_loader:
                # Move audio data and labels to device.
                audio = batch["audio"].to(self.device)  # shape: (B, seq_len, feature_dim)
                text = {
                    "text_input_ids": batch["text_input_ids"].to(self.device),
                    "text_attention_mask": batch["text_attention_mask"].to(self.device)
                }
                labels = batch["label"].to(self.device)
                outputs = self.model(audio, text)  # shape: (B, num_classes)
                loss = self.criterion(outputs, labels)
                batch_size = labels.size(0)
                val_loss_total += loss.item() * batch_size
                total_samples += batch_size
        avg_val_loss = val_loss_total / total_samples if total_samples > 0 else 0.0
        return avg_val_loss

    def train(self) -> None:
        """
        Executes the full training loop, iterating over epochs and training batches.
        After each epoch, performs validation and saves the best performing model based on validation loss.
        """
        print(f"Starting training on device: {self.device}")
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            epoch_train_loss: float = 0.0
            total_batches: int = 0

            # Use tqdm progress bar for training batches.
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}", leave=False)
            for batch in progress_bar:
                # Prepare inputs.
                audio = batch["audio"].to(self.device)  # (B, seq_len, feature_dim)
                text = {
                    "text_input_ids": batch["text_input_ids"].to(self.device),
                    "text_attention_mask": batch["text_attention_mask"].to(self.device)
                }
                labels = batch["label"].to(self.device)

                # Zero the gradients.
                self.optimizer.zero_grad()

                # Forward pass.
                outputs = self.model(audio, text)  # (B, num_classes)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Accumulate loss.
                epoch_train_loss += loss.item()
                total_batches += 1
                progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

            # Compute average training loss for the epoch.
            avg_train_loss = epoch_train_loss / total_batches if total_batches > 0 else 0.0

            # Perform validation.
            avg_val_loss = self.validate()

            # Log epoch statistics.
            print(f"Epoch [{epoch}/{self.epochs}] - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

            # Log to Weights & Biases.
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                })

            # Checkpoint: Save model if validation loss improved.
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_checkpoint(epoch, avg_val_loss)
                # Reset counter when improvement occurs.
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            # Early stopping check.
            if self.epochs_no_improve >= self.early_stop_patience:
                print(f"Early stopping triggered after {self.early_stop_patience} epochs with no improvement.")
                break
        print("Training complete.")

        # Finalize the wandb run.
        if self.use_wandb:
            wandb.finish()
