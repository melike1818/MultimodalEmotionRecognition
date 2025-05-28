"""
model.py

This module implements the MERHAN model for multimodal emotion recognition based on audio and text using
hybrid attention networks, as described in the paper "Multimodal emotion recognition based on audio and text by using hybrid_att". 

The MERHAN model is composed of four main submodules:
    - AudioEncoder: 2-layer Bi-LSTM with local intra-modal MHSA on MFCC features.
    - TextEncoder: Pretrained BERT with local intra-modal MHSA on token embeddings.
    - CrossModalAttention: Projects audio and text features into a common space via 1D-CNN and applies bi-directional
      cross-attention with residual connections and layer normalization.
    - GlobalInterModalAttention: Computes a global attention weighted aggregation over the concatenated cross-modal features 
      followed by a classification FC layer.

All model hyperparameters are derived from the configuration file (config.yaml).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from typing import Dict, Any

# ----------------------------
# Audio Encoder Module
# ----------------------------
class AudioEncoder(nn.Module):
    """Audio encoder module using a two-layer Bi-LSTM followed by a multi-head self-attention mechanism."""
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the AudioEncoder.

        Args:
            config: Dictionary containing the audio encoder configuration parameters.
                    Expected keys:
                        - "input_feature_dim": Dimension of input MFCC features (default: 40).
                        - "hidden_size": Hidden size of LSTM (default: 256).
                        - "num_layers": Number of LSTM layers (default: 2).
                        - "mhsa_heads": Number of attention heads for local intra-modal attention (default: 4).
        """
        super(AudioEncoder, self).__init__()
        input_feature_dim: int = config.get("input_feature_dim", 40)
        hidden_size: int = config.get("hidden_size", 256)
        num_layers: int = config.get("num_layers", 2)
        mhsa_heads: int = config.get("mhsa_heads", 4)

        # BiLSTM: bidirectional LSTM, batch_first=True.
        # Output dimension per time step becomes hidden_size*2.
        self.bilstm: nn.LSTM = nn.LSTM(
            input_size=input_feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        embed_dim_audio: int = hidden_size * 2  # 512

        # Local intra-modal self-attention via MultiheadAttention.
        # Using batch_first=True so input and output shapes are (batch, seq_len, embed_dim_audio).
        self.mhsa: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=embed_dim_audio,
            num_heads=mhsa_heads,
            batch_first=True
        )
        self.layernorm: nn.LayerNorm = nn.LayerNorm(embed_dim_audio)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for AudioEncoder.

        Args:
            audio: Tensor of shape (batch, seq_len, input_feature_dim).

        Returns:
            Tensor of shape (batch, seq_len, embed_dim_audio) with refined audio features.
        """
        # Pass through Bi-LSTM
        lstm_out, _ = self.bilstm(audio)  # shape: (batch, seq_len, hidden_size*2)
        
        # Apply local intra-modal self-attention.
        # nn.MultiheadAttention with batch_first=True accepts input shape (batch, seq_len, embed_dim).
        attn_output, _ = self.mhsa(lstm_out, lstm_out, lstm_out)
        # Residual connection and layer normalization.
        refined_audio = self.layernorm(lstm_out + attn_output)
        return refined_audio  # shape: (batch, seq_len, 512)

# ----------------------------
# Text Encoder Module
# ----------------------------
class TextEncoder(nn.Module):
    """Text encoder module using a pretrained BERT model followed by a multi-head self-attention mechanism."""
    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the TextEncoder.

        Args:
            config: Dictionary containing text encoder parameters.
                    Expected keys:
                        - "pretrained_model": Name of the pretrained BERT model (default: "bert-base-uncased").
                        - "output_dim": Dimension of BERT output (default: 768).
                        - "mhsa_heads": Number of attention heads for local intra-modal attention (default: 4).
        """
        super(TextEncoder, self).__init__()
        pretrained_model: str = config.get("pretrained_model", "bert-base-uncased")
        output_dim: int = config.get("output_dim", 768)
        mhsa_heads: int = config.get("mhsa_heads", 4)

        # Load pretrained BERT model.
        self.bert: BertModel = BertModel.from_pretrained(pretrained_model)
        # Local intra-modal self-attention module.
        self.mhsa: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=mhsa_heads,
            batch_first=True
        )
        self.layernorm: nn.LayerNorm = nn.LayerNorm(output_dim)

    def forward(self, text_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for TextEncoder.

        Args:
            text_inputs: Dictionary with keys "text_input_ids" and "text_attention_mask".
                         Each has shape (batch, seq_len).

        Returns:
            Tensor of shape (batch, seq_len, output_dim) with refined text features.
        """
        input_ids: torch.Tensor = text_inputs["text_input_ids"]
        attention_mask: torch.Tensor = text_inputs["text_attention_mask"]
        
        # Get BERT outputs; use the last hidden state.
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output: torch.Tensor = bert_outputs.last_hidden_state  # shape: (batch, seq_len, output_dim)

        # Apply local intra-modal self-attention.
        attn_output, _ = self.mhsa(sequence_output, sequence_output, sequence_output)
        refined_text = self.layernorm(sequence_output + attn_output)
        return refined_text  # shape: (batch, seq_len, 768)

# ----------------------------
# Cross Modal Attention Module
# ----------------------------
class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention block that projects audio and text features into a common subspace
    and applies bidirectional cross-attention (A->T and T->A) with residual connections and layer normalization.
    """
    def __init__(self, config: Dict[str, Any], audio_in_dim: int, text_in_dim: int) -> None:
        """
        Initializes the CrossModalAttention block.

        Args:
            config: Dictionary containing cross-modal attention configuration.
                    Expected keys:
                        - "cnn_kernel_size": Kernel size for the 1D CNN projection (default: 1).
                        - "projection_dim": Output dimension (d) for the 1D CNN projection (default: 256).
                        - "mhsa_heads": Number of heads for cross-modal MHSA (default: 4).
            audio_in_dim: Dimension of the input audio features (e.g., 512 from AudioEncoder).
            text_in_dim: Dimension of the input text features (e.g., 768 from TextEncoder).
        """
        super(CrossModalAttention, self).__init__()
        cnn_kernel_size: int = config.get("cnn_kernel_size", 1)
        projection_dim: int = config.get("projection_dim", 256)
        mhsa_heads: int = config.get("mhsa_heads", 4)

        # 1D CNN projection for audio and text features.
        # nn.Conv1d expects input shape (batch, channels, seq_len), so we'll transpose accordingly.
        self.audio_proj: nn.Conv1d = nn.Conv1d(
            in_channels=audio_in_dim,
            out_channels=projection_dim,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2
        )
        self.text_proj: nn.Conv1d = nn.Conv1d(
            in_channels=text_in_dim,
            out_channels=projection_dim,
            kernel_size=cnn_kernel_size,
            padding=cnn_kernel_size // 2
        )

        # Cross-modal attention modules.
        # For Audio→Text: query from text, key/value from audio.
        self.attn_a2t: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=mhsa_heads,
            batch_first=True
        )
        # For Text→Audio: query from audio, key/value from text.
        self.attn_t2a: nn.MultiheadAttention = nn.MultiheadAttention(
            embed_dim=projection_dim,
            num_heads=mhsa_heads,
            batch_first=True
        )

        # Layer normalization for residual connections.
        self.layernorm_a2t: nn.LayerNorm = nn.LayerNorm(projection_dim)
        self.layernorm_t2a: nn.LayerNorm = nn.LayerNorm(projection_dim)

    def forward(self, audio_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        Performs cross-modal attention to fuse audio and text features.

        Args:
            audio_features: Tensor of shape (batch, seq_len, audio_in_dim) from AudioEncoder.
            text_features: Tensor of shape (batch, seq_len, text_in_dim) from TextEncoder.

        Returns:
            Fused feature tensor of shape (batch, seq_len, 2 * projection_dim).
        """
        batch_size, seq_len, _ = audio_features.size()
        projection_dim: int = self.audio_proj.out_channels  # Expected to be 256

        # Project audio features: from (batch, seq_len, audio_in_dim) -> (batch, seq_len, projection_dim)
        audio_proj = self.audio_proj(audio_features.transpose(1, 2)).transpose(1, 2)
        # Project text features: from (batch, seq_len, text_in_dim) -> (batch, seq_len, projection_dim)
        text_proj = self.text_proj(text_features.transpose(1, 2)).transpose(1, 2)

        # Cross-modal attention: Audio -> Text (A→T)
        # Query: text_proj, Key and Value: audio_proj.
        a2t_attn, _ = self.attn_a2t(query=text_proj, key=audio_proj, value=audio_proj)
        # Residual connection and layer normalization.
        a2t_out = self.layernorm_a2t(text_proj + a2t_attn)

        # Cross-modal attention: Text -> Audio (T→A)
        # Query: audio_proj, Key and Value: text_proj.
        t2a_attn, _ = self.attn_t2a(query=audio_proj, key=text_proj, value=text_proj)
        t2a_out = self.layernorm_t2a(audio_proj + t2a_attn)

        # Concatenate the results along the feature dimension.
        fusion: torch.Tensor = torch.cat([a2t_out, t2a_out], dim=-1)  # shape: (batch, seq_len, 2*projection_dim)
        return fusion

# ----------------------------
# Global Inter-Modal Attention Module (MEC Block)
# ----------------------------
class GlobalInterModalAttention(nn.Module):
    """
    Global Inter-Modal Attention (MEC Block) that computes a weighted aggregation of the fused
    cross-modal features and outputs class logits through a fully connected layer.
    """
    def __init__(self, num_classes: int, projection_dim: int) -> None:
        """
        Initializes the GlobalInterModalAttention module.

        Args:
            num_classes: Number of output emotion classes.
            projection_dim: The projection dimension used in CrossModalAttention.
                           The fusion dimension will be 2 * projection_dim.
        """
        super(GlobalInterModalAttention, self).__init__()
        self.fusion_dim: int = 2 * projection_dim  # e.g., 512
        # Trainable weight vector U for global attention.
        self.U: nn.Parameter = nn.Parameter(torch.randn(self.fusion_dim))
        # Final fully connected layer for classification.
        self.fc: nn.Linear = nn.Linear(self.fusion_dim, num_classes)

    def forward(self, fusion: torch.Tensor) -> torch.Tensor:
        """
        Computes global inter-modal attention over the fused features and outputs logits.

        Args:
            fusion: Tensor of shape (batch, seq_len, fusion_dim).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        # Compute attention scores for each time step by dot-product with U.
        # fusion: (B, seq_len, fusion_dim), U: (fusion_dim,)
        scores: torch.Tensor = torch.matmul(fusion, self.U)  # shape: (B, seq_len)
        # Compute softmax across the sequence dimension to get attention weights.
        attn_weights: torch.Tensor = F.softmax(scores, dim=1)  # (B, seq_len)
        attn_weights = attn_weights.unsqueeze(-1)              # (B, seq_len, 1)

        # Attention-weighted aggregation (global inter-modal context vector).
        aggregated: torch.Tensor = torch.sum(fusion * attn_weights, dim=1)  # (B, fusion_dim)

        # ------------------------------------------------------------------
        # Residual skip-connection (bypasses the attention computation).
        # We follow the paper's MEC block that multiplies (⊗) the attention
        # output with a residual representation of the same dimensionality.
        # Here we use a simple mean-pool over the temporal axis to obtain the
        # residual representation r ∈ R^{fusion_dim}.
        # ------------------------------------------------------------------
        residual: torch.Tensor = torch.mean(fusion, dim=1)  # (B, fusion_dim)

        # Element-wise gating (⊗) between attention vector and residual path.
        gated: torch.Tensor = aggregated * residual          # (B, fusion_dim)

        # Final classification.
        logits: torch.Tensor = self.fc(gated)                # (B, num_classes)
        return logits

# ----------------------------
# MERHAN Model Class
# ----------------------------
class MERHAN(nn.Module):
    """
    MERHAN multimodal emotion recognition model which integrates audio and text modalities
    using hybrid attention networks. It composes the AudioEncoder, TextEncoder,
    CrossModalAttention block, and GlobalInterModalAttention block.
    """
    def __init__(self, config: Dict[str, Any], num_classes: int) -> None:
        """
        Initializes the MERHAN model.

        Args:
            config: Configuration dictionary loaded from config.yaml.
            num_classes: Number of emotion classes for classification.
                (For IEMOCAP: 4, for MELD: 7)
        """
        super(MERHAN, self).__init__()
        # Initialize Audio Encoder.
        self.audio_encoder: AudioEncoder = AudioEncoder(config["model"]["audio_encoder"])
        # Initialize Text Encoder.
        self.text_encoder: TextEncoder = TextEncoder(config["model"]["text_encoder"])
        # The output dimensions from encoders:
        #   AudioEncoder output dimension: 512 (hidden_size*2)
        #   TextEncoder output dimension: 768
        # Initialize Cross Modal Attention block.
        self.cross_modal_attention: CrossModalAttention = CrossModalAttention(
            config=config["model"]["cross_modal_attention"],
            audio_in_dim=512,
            text_in_dim=768
        )
        # Global Inter-Modal Attention (MEC Block)
        projection_dim: int = config["model"]["cross_modal_attention"].get("projection_dim", 256)
        self.global_attention: GlobalInterModalAttention = GlobalInterModalAttention(
            num_classes=num_classes,
            projection_dim=projection_dim
        )

    def forward(self, audio: torch.Tensor, text: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the MERHAN model.

        Args:
            audio: Tensor of shape (batch, seq_len, input_feature_dim) for audio MFCC features.
            text: Dictionary containing text inputs with keys "text_input_ids" and "text_attention_mask",
                  each of shape (batch, seq_len).

        Returns:
            Logits tensor of shape (batch, num_classes) for emotion classification.
        """
        # Process audio modality.
        audio_features: torch.Tensor = self.audio_encoder(audio)  # (B, seq_len, 512)
        # Process text modality.
        text_features: torch.Tensor = self.text_encoder(text)     # (B, seq_len, 768)
        # Cross-modal attention fusion.
        fusion: torch.Tensor = self.cross_modal_attention(audio_features, text_features)  # (B, seq_len, 512)
        # Global inter-modal attention and classification.
        logits: torch.Tensor = self.global_attention(fusion)  # (B, num_classes)
        return logits
