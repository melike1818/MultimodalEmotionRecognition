"""dataset_loader.py

This module implements the DatasetLoader class responsible for loading and preprocessing
the IEMOCAP and MELD datasets for the MER-HAN multimodal emotion recognition project.
The module includes custom Dataset classes for IEMOCAP and MELD, a collate function
to pad variable length audio and text sequences, and helper functions to load metadata.
"""

import os
import csv
import random
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchaudio
torchaudio.set_audio_backend("soundfile")
from transformers import BertTokenizer

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

label_map = {
    "neu": 0,
    "hap": 1,
    "ang": 2,
    "sad": 3,
    # add other labels as needed
}

def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
    """
    Collate function to pad audio and text sequences in the batch.
    
    For audio:
      - Each audio sample is expected to have shape [n_mfcc, time_steps].
      - The audio tensor is transposed to [time_steps, n_mfcc] and then zero-padded to the maximum time_steps.
    
    For text:
      - Text tokenized sequences (input_ids and attention_mask) are padded to the maximum sequence length in the batch.
    
    Returns:
        A dictionary with keys:
         - "audio": Tensor of shape (batch_size, max_time_steps, feature_dim)
         - "text_input_ids": Tensor of shape (batch_size, max_text_length)
         - "text_attention_mask": Tensor of shape (batch_size, max_text_length)
         - "label": Tensor of labels (batch_size,)
    """
    # Process audio: transpose each tensor to (time_steps, feature_dim) for padding.
    audio_list = []
    for sample in batch:
        # sample["audio"] shape: [n_mfcc, time_steps]
        audio_tensor = sample["audio"].transpose(0, 1)  # now (time_steps, n_mfcc)
        audio_list.append(audio_tensor)
    padded_audio = pad_sequence(audio_list, batch_first=True, padding_value=0.0)  # (B, max_time, n_mfcc)

    # Process text input_ids and attention_mask. They may already be fixed length from tokenizer,
    # but in case variable lengths occur, pad them.
    text_ids_list = [sample["text_input_ids"] for sample in batch]
    text_mask_list = [sample["text_attention_mask"] for sample in batch]
    padded_text_ids = pad_sequence(text_ids_list, batch_first=True, padding_value=0)
    padded_text_mask = pad_sequence(text_mask_list, batch_first=True, padding_value=0)

    # Process labels.
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)

    return {
        "audio": padded_audio,  # (B, max_time_steps, feature_dim)
        "text_input_ids": padded_text_ids,  # (B, max_seq_length)
        "text_attention_mask": padded_text_mask,  # (B, max_seq_length)
        "label": labels  # (B,)
    }


# -----------------------------------------------------------------------------
#  Dataset classes
# -----------------------------------------------------------------------------
# All datasets enforce a COMMON fixed sequence length (`max_seq_length`).
#   • If an audio MFCC sequence is shorter than this length, it is zero-padded
#     on the time-axis; if it is longer, it is truncated.
#   • The same rule is applied to the text token sequence (input_ids and
#     attention_mask) by telling the tokenizer to produce `max_seq_length`
#     tokens.  This guarantees that every sample arrives at the model with
#     audio and text tensors whose first (time) dimension equals
#     `max_seq_length`, which solves the size-mismatch error during
#     cross-modal fusion.


class IEMOCAPDataset(Dataset):
    """
    Custom Dataset for the IEMOCAP dataset.
    Each sample is expected to be a dict with keys:
        - "audio_path": Path to the audio file.
        - "transcript": The text transcription.
        - "label": Emotion label (as an integer).
        - "session": Session number (used for splitting training and testing).
    """
    def __init__(self,
                 samples: List[Dict[str, Any]],
                 mfcc_transform: torchaudio.transforms.MFCC,
                 tokenizer: BertTokenizer,
                 max_seq_length: int = 128) -> None:
        """Initialise the dataset.

        Args:
            samples:        Metadata entries.
            mfcc_transform: Pre-configured MFCC extractor.
            tokenizer:      BERT tokenizer.
            max_seq_length: Fixed length shared by BOTH modalities.
        """
        self.samples = samples
        self.mfcc_transform = mfcc_transform
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load audio using torchaudio.
        audio_path = sample["audio_path"]
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} not found.")
        waveform, sample_rate = torchaudio.load(audio_path)  # waveform shape: [channels, num_samples]
        # If waveform has more than one channel, average the channels.
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Apply MFCC transform -> (n_mfcc, T)
        mfcc = self.mfcc_transform(waveform).squeeze(0)  # [n_mfcc, T]

        # --- Align audio length to max_seq_length --------------------------------
        if mfcc.shape[1] > self.max_seq_length:
            mfcc = mfcc[:, : self.max_seq_length]
        elif mfcc.shape[1] < self.max_seq_length:
            pad_len = self.max_seq_length - mfcc.shape[1]
            mfcc = torch.nn.functional.pad(mfcc, (0, pad_len))  # pad time axis

        # Tokenize the transcript using the BERT tokenizer.
        transcript = sample["transcript"]
        if transcript.strip() == "":
            transcript = "unknown"
        tokenized = self.tokenizer(
            transcript,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        input_ids = tokenized['input_ids'].squeeze(0)          # shape: [max_seq_length]
        attention_mask = tokenized['attention_mask'].squeeze(0)  # shape: [max_seq_length]

        label_str = sample["label"]
        label = label_map.get(label_str, 0)

        return {
            "audio": mfcc,  # shape: [n_mfcc, max_seq_length]
            "text_input_ids": input_ids,  # shape: [max_seq_length]
            "text_attention_mask": attention_mask,  # shape: [max_seq_length]
            "label": label
        }


class MELDDataset(Dataset):
    """
    Custom Dataset for the MELD dataset.
    Each sample is expected to be a dict with keys:
        - "audio_path": Path to the audio file.
        - "transcript": The text transcription.
        - "label": Emotion label (as an integer).
    """
    def __init__(self,
                 samples: List[Dict[str, Any]],
                 mfcc_transform: torchaudio.transforms.MFCC,
                 tokenizer: BertTokenizer,
                 max_seq_length: int = 128) -> None:
        self.samples = samples
        self.mfcc_transform = mfcc_transform
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Load audio using torchaudio.
        audio_path = sample["audio_path"]
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file {audio_path} not found.")
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Apply MFCC transform -> (n_mfcc, T)
        mfcc = self.mfcc_transform(waveform).squeeze(0)

        # --- Align audio length --------------------------------------------------
        if mfcc.shape[1] > self.max_seq_length:
            mfcc = mfcc[:, : self.max_seq_length]
        elif mfcc.shape[1] < self.max_seq_length:
            pad_len = self.max_seq_length - mfcc.shape[1]
            mfcc = torch.nn.functional.pad(mfcc, (0, pad_len))

        # Tokenize transcript.
        transcript = sample["transcript"]
        if transcript.strip() == "":
            transcript = "unknown"
        tokenized = self.tokenizer(
            transcript,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        input_ids = tokenized['input_ids'].squeeze(0)
        attention_mask = tokenized['attention_mask'].squeeze(0)

        label_str = sample["label"]
        label = label_map.get(label_str, 0)  # default to 0 if not found

        return {
            "audio": mfcc,
            "text_input_ids": input_ids,
            "text_attention_mask": attention_mask,
            "label": label
        }


class DatasetLoader:
    """
    DatasetLoader class handles loading and preprocessing of the IEMOCAP and MELD datasets.
    It initializes the audio MFCC transform, the BERT tokenizer, loads metadata from CSV files,
    splits the data according to the configuration, and returns DataLoader objects for training,
    validation, and testing.
    """
    def __init__(self, config: Dict[str, Any], dataset_type: str = "IEMOCAP") -> None:
        self.config = config
        self.dataset_type = dataset_type.upper()  # Supported values: "IEMOCAP" or "MELD"

        # Set random seeds for reproducibility.
        seed = config.get("random_seed", 42)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.batch_size: int = config["training"].get("batch_size", 32)

        # Unified sequence length across modalities ------------------------------
        self.max_seq_length: int = config.get("dataset", {}).get("max_seq_length", 128)

        # Setup MFCC transform for audio preprocessing.
        sample_rate: int = 16000  # Default sample rate
        frame_width_ms: int = config["model"]["audio_encoder"].get("frame_width_ms", 25)
        frame_stride_ms: int = config["model"]["audio_encoder"].get("frame_stride_ms", 10)
        win_length: int = int(sample_rate * frame_width_ms / 1000)
        hop_length: int = int(sample_rate * frame_stride_ms / 1000)
        n_mfcc: int = config["model"]["audio_encoder"].get("input_feature_dim", 40)
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                'n_fft': 512,
                'win_length': win_length,
                'hop_length': hop_length,
                'n_mels': n_mfcc
            }
        )

        # Initialize BERT tokenizer for text preprocessing.
        pretrained_model: str = config["model"]["text_encoder"].get("pretrained_model", "bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    def load_metadata(self, metadata_path: str) -> List[Dict[str, Any]]:
        """
        Loads metadata from a CSV file located at metadata_path.

        The CSV file is expected to have a header and contains one sample per row.
        Returns a list of dictionaries representing samples.
        """
        samples: List[Dict[str, Any]] = []
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError(f"Metadata file {metadata_path} not found.")
        with open(metadata_path, mode="r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                samples.append(row)
        logger.info(f"Loaded {len(samples)} samples from {metadata_path}")
        return samples

    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Loads the dataset based on the dataset_type (IEMOCAP or MELD), splits the data into training,
        validation, and testing sets, and returns corresponding DataLoader objects.
        
        Returns:
            A tuple (train_loader, val_loader, test_loader)
        """
        if self.dataset_type == "IEMOCAP":
            metadata_path: str = os.path.join("data", "iemocap_metadata.csv")
            all_samples: List[Dict[str, Any]] = self.load_metadata(metadata_path)
            # Filter samples based on session information.
            train_sessions: List[int] = self.config["dataset"]["IEMOCAP"].get("train_sessions", [1, 2, 3, 4])
            test_sessions: List[int] = self.config["dataset"]["IEMOCAP"].get("test_sessions", [5])
            train_samples = [s for s in all_samples if int(s.get("session", 0)) in train_sessions]
            test_samples = [s for s in all_samples if int(s.get("session", 0)) in test_sessions]

            # Create a validation split from training samples (e.g., 10% for validation).
            random.shuffle(train_samples)
            val_size: int = max(1, int(0.1 * len(train_samples)))
            val_samples = train_samples[:val_size]
            train_samples = train_samples[val_size:]
            logger.info(f"IEMOCAP: {len(train_samples)} training, {len(val_samples)} validation, {len(test_samples)} testing samples.")

            # Initialize Dataset objects.
            train_dataset = IEMOCAPDataset(train_samples, self.mfcc_transform, self.tokenizer,
                                           max_seq_length=self.max_seq_length)
            val_dataset = IEMOCAPDataset(val_samples, self.mfcc_transform, self.tokenizer,
                                         max_seq_length=self.max_seq_length)
            test_dataset = IEMOCAPDataset(test_samples, self.mfcc_transform, self.tokenizer,
                                          max_seq_length=self.max_seq_length)

        elif self.dataset_type == "MELD":
            metadata_path = os.path.join("data", "meld_metadata.csv")
            all_samples = self.load_metadata(metadata_path)
            # Use pre-defined splits from configuration.
            train_split: int = self.config["dataset"]["MELD"].get("train_split", 1039)
            val_split: int = self.config["dataset"]["MELD"].get("val_split", 114)
            test_split: int = self.config["dataset"]["MELD"].get("test_split", 280)
            if len(all_samples) < (train_split + val_split + test_split):
                raise ValueError("Not enough samples in MELD metadata for the specified splits.")
            train_samples = all_samples[:train_split]
            val_samples = all_samples[train_split: train_split + val_split]
            test_samples = all_samples[train_split + val_split: train_split + val_split + test_split]
            logger.info(f"MELD: {len(train_samples)} training, {len(val_samples)} validation, {len(test_samples)} testing samples.")

            train_dataset = MELDDataset(train_samples, self.mfcc_transform, self.tokenizer,
                                        max_seq_length=self.max_seq_length)
            val_dataset = MELDDataset(val_samples, self.mfcc_transform, self.tokenizer,
                                      max_seq_length=self.max_seq_length)
            test_dataset = MELDDataset(test_samples, self.mfcc_transform, self.tokenizer,
                                       max_seq_length=self.max_seq_length)

        else:
            raise ValueError(f"Unsupported dataset type: {self.dataset_type}. Supported types: IEMOCAP, MELD.")

        # Create DataLoader objects.
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size,
                                  shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(dataset=val_dataset, batch_size=self.batch_size,
                                shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size,
                                 shuffle=False, collate_fn=collate_fn)

        return train_loader, val_loader, test_loader
