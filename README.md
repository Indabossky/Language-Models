# Language Modeling Project

## Overview
This project implements and compares three sequential deep learning models—RNN, LSTM, and Transformer—for language modeling using PyTorch. A Byte-Pair Encoding (BPE) tokenizer is trained using SentencePiece and used to preprocess text data.

## Features
- Train a SentencePiece BPE tokenizer from JSONL data (prompt/completion pairs).
- Train and evaluate RNN, LSTM, and Transformer language models.
- Calculate perplexity and BLEU scores for model evaluation.
- Plot training and validation loss curves and save as images.

## Requirements
- Python 3.10 or later
- PyTorch
- sentencepiece
- matplotlib
- nltk
- tqdm

You can install the required packages using:
```bash
pip install torch sentencepiece matplotlib nltk tqdm
```

## Project Structure
```
.
├── main.py                  # Entry point: trains tokenizer, loads models, generates text and evaluates
├── training.py              # Training loop, evaluation metrics, and plotting utilities
├── language_modeld.py       # Model definitions: Base, RNN, LSTM, Transformer
├── data_processing.py       # Tokenizer training/loading and JSONL Dataset class
└── data/
    ├── train.jsonl          # Training data (prompt/completion JSONL)
    └── test.jsonl           # Testing data (prompt/completion JSONL)
```

## Usage

1. **Prepare your data**  
   Place your `train.jsonl` and `test.jsonl` files under the `data/` directory. Each line in these files should be a JSON object with `"prompt"` and `"completion"` fields.

2. **Run the main script**  
   ```bash
   python main.py
   ```
   This will:
   - Train the SentencePiece tokenizer.
   - Load existing model weights (`RNN.pth`, `LSTM.pth`, `Transformer.pth`) if available.
   - Generate text for a sample prompt and compute BLEU scores.

3. **Train models (optional)**  
   To train models from scratch, uncomment the training block in `main.py`. It will save model weights and plot loss curves.

4. **Adjust hyperparameters**  
   Modify settings such as `seq_length`, `batch_size`, `epochs`, and learning rate directly in `main.py`.

## Customization
- Add or modify model architectures in `language_modeld.py`.
- Extend the dataset class in `data_processing.py` for different data formats.
- Tweak training strategies in `training.py` (e.g., optimizer, scheduler).
