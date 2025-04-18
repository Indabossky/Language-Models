import os
import json
import sentencepiece as spm
import torch
from torch.utils.data import Dataset
 
def train_sentencepiece_tokenizer_from_jsonl(jsonl_file, model_prefix="spm_model", vocab_size=10000, separator=" "):
    """
    Train the SentencePiece tokenizer using JSONL data.
    It reads each JSON record (with 'prompt' and 'completion'), concatenates them using the separator,
    writes a temporary corpus file for training, and then removes it.
    """
    temp_file = "temp_corpus.txt"
    with open(jsonl_file, "r", encoding="utf-8") as infile, open(temp_file, "w", encoding="utf-8") as outfile:
        for line in infile:
            if line.strip():
                record = json.loads(line.strip())
                combined_text = record.get("prompt", "") + separator + record.get("completion", "")
                outfile.write(combined_text + "\n")
    spm.SentencePieceTrainer.train(
        f"--input={temp_file} --model_prefix={model_prefix} --vocab_size={vocab_size} --character_coverage=1.0 --model_type=bpe"
    )
    os.remove(temp_file)
    print(f"Trained SentencePiece model. Files generated: {model_prefix}.model and {model_prefix}.vocab")
 
def load_tokenizer(model_file="spm_model.model"):
    """Load a trained SentencePiece model."""
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    return sp
 
class JSONLanguageModelDataset(Dataset):
    """
    PyTorch Dataset for language model training using a JSONL file.
    Each record in the JSONL (with 'prompt' and 'completion') is concatenated (with a separator)
    and tokenized using the provided SentencePiece model. All token IDs across records are merged into
    one long list, and overlapping sequences of length `seq_length` are generated.
    """
    def __init__(self, jsonl_file, tokenizer, seq_length=32, separator=" "):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.tokens = []
        with open(jsonl_file, "r", encoding="utf-8") as infile:
            for line in infile:
                if line.strip():
                    record = json.loads(line.strip())
                    combined_text = record.get("prompt", "") + separator + record.get("completion", "")
                    token_ids = self.tokenizer.encode_as_ids(combined_text)
                    self.tokens.extend(token_ids)
        print(f"Loaded {len(self.tokens)} tokens from {jsonl_file}")
 
    def __len__(self):
        return len(self.tokens) - self.seq_length
 
    def __getitem__(self, idx):
        x = torch.tensor(self.tokens[idx: idx + self.seq_length], dtype=torch.long)
        y = torch.tensor(self.tokens[idx+1: idx + self.seq_length + 1], dtype=torch.long)
        return x, y