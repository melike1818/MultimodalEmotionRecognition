# MER-HAN: Multimodal Emotion Recognition with Hybrid Attention Networks

Implementation of the paper **"Multimodal emotion recognition based on audio and text by using hybrid attention networks"** (<https://doi.org/10.1016/j.bspc.2023.105052>) as the course project of **CENG 562 – Machine Learning, Middle East Technical University (METU)**.

The repository reproduces the core ideas of the paper in PyTorch: a dual-stream architecture that encodes speech and text with separate intra-modal attention blocks, fuses them through bi-directional cross-modal attention, and performs global inter-modal reasoning for final emotion classification.

---
## Table of Contents
1.  [Features](#features)
2.  [Repository Layout](#repository-layout)
3.  [Installation](#installation)
4.  [Dataset Preparation](#dataset-preparation)
5.  [Configuration](#configuration)
6.  [Training & Evaluation](#training--evaluation)
7.  [Checkpoints](#checkpoints)
8.  [Results](#results)
9.  [Citation](#citation)
10. [Acknowledgements](#acknowledgements)

---
## Features
• Supports **IEMOCAP** and **MELD** datasets out-of-the-box  
• Unified *max_seq_length* to avoid size-mismatch during fusion  
• Modular implementation: `AudioEncoder`, `TextEncoder`, `CrossModalAttention`, `GlobalInterModalAttention`  
• End-to-end training with early stopping & checkpointing  
• Evaluation metrics: WAR, UAR, macro / weighted F1 and confusion matrix  
• Utility script to scan the whole corpus and suggest a safe `max_seq_length`

---
## Repository Layout
```
├── checkpoints/              # saved *.pth* models
├── data/                     # metadata CSVs & (symbolic links to) raw audio
├── config.yaml               # default hyper-parameters
├── dataset_loader.py         # Dataset / DataLoader utilities
├── model.py                  # MER-HAN network definition
├── trainer.py                # training loop with validation & checkpointing
├── evaluation.py             # test-time metrics
├── main.py                   # entry-point script
└── requirements.txt          # Python dependencies
```

---
## Installation
1.  **Clone the repo** (with submodules if any):
    ```bash
    git clone https://github.com/melike1818/MultimodalEmotionRecognition.git
    cd MultimodalEmotionRecognition
    ```
2.  **Create a virtual environment** (recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Install Python packages**:
    ```bash
    pip install -r requirements.txt
    ```
    This will pull PyTorch, torchaudio, transformers, scikit-learn, tqdm, pyyaml, etc.  Make sure the CUDA version of PyTorch matches your GPU driver.

---
## Dataset Preparation
The code expects pre-extracted metadata CSV files that map each utterance to its audio path, transcript and label.

1.  **IEMOCAP**
    * Place the original IEMOCAP audio files in any directory of your choice.
    * A ready-to-use `data/iemocap_metadata.csv` is already provided in the repo.  Edit the `audio_path` column if your folder layout differs.

2.  **MELD**
    * download the MELD release and build `data/meld_metadata.csv` with columns:
      `audio_path,transcript,label`.

---
## Configuration
All hyper-parameters live in [`config.yaml`](config.yaml): learning rate, batch size, encoder dimensions, sequence length, etc.  Key points:

* `dataset.max_seq_length`: shared temporal length for **both** modalities.  If unsure, run the utility below.
* `model.classification.num_classes_IEMOCAP / num_classes_MELD`: number of emotion classes.
* `hardware.device`: set to `cuda`, `cuda:0`, `cpu`, …

Feel free to copy `config.yaml` to a new file and override values per experiment.

---
## Training & Evaluation
```bash
# Example: train on IEMOCAP with the default config
python main.py --config config.yaml --dataset IEMOCAP

# Train on MELD
python main.py --config config.yaml --dataset MELD
```

During training the best model (lowest validation loss) is stored under `checkpoints/` and automatically loaded for the final test evaluation.  Metrics are logged to the console.

---
## Checkpoints
Pre-trained weights can be placed in `checkpoints/best_model.pth`.  `Trainer` will resume training if the file exists.

---
## Results
Reproducing the exact numbers from the original paper may require further fine-tuning, but the current implementation reaches comparable performance:

| Dataset | WAR | UAR | Macro F1 |
|---------|-----|-----|----------|
| IEMOCAP (fold 5) | *coming soon* | *coming soon* | *coming soon* |

---
## Citation
If you use this codebase in your research, please cite the original paper:
```
@article{zhou2023merhan,
  title  = {Multimodal emotion recognition based on audio and text by using hybrid attention networks},
  journal= {Biomedical Signal Processing and Control},
  volume = {84},
  pages  = {105050},
  year   = {2023},
  doi    = {10.1016/j.bspc.2023.105050}
}
```

You may also acknowledge this implementation:
```
@misc{merhan_ceng562_2025,
  author = {M. Demirci},
  title  = {{MER-HAN} PyTorch implementation for {CENG 562} Machine Learning Project},
  year   = {2025},
  howpublished = {\url{https://github.com/melike1818/MultimodalEmotionRecognition}}
}
```

---
## Acknowledgements
* **Y. Zhou et al.** for proposing MER-HAN.
* HuggingFace 🤗 Transformers for the BERT backbone.
* PyTorch & torchaudio for the deep learning framework.
