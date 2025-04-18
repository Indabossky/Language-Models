import torch
import torch.nn as nn
import torch.nn.functional as F
 
class BaseLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(BaseLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
 
    def sample_next(self, logits, temperature=1.0):
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.argmax(probs, dim=-1)
        return next_token
 
    def prompt(self, prompt_text, sp, max_seq_length=50, temperature=1.0, device="cpu"):
        token_ids = sp.encode_as_ids(prompt_text)
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        self.eval()
        generated = token_ids.copy()
        with torch.no_grad():
            for _ in range(max_seq_length):
                logits = self.forward(input_ids)  # shape: [1, seq_length, vocab_size]
                next_token_logits = logits[:, -1, :]
                next_token = self.sample_next(next_token_logits, temperature)
                next_token_id = next_token.item()
                generated.append(next_token_id)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
        return sp.decode_ids(generated)
 
class RNNLanguageModel(BaseLanguageModel):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=1):
        super(RNNLanguageModel, self).__init__(vocab_size, embedding_dim, hidden_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
 
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        logits = self.fc_out(output)
        return logits
 
class LSTMLanguageModel(BaseLanguageModel):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=1):
        super(LSTMLanguageModel, self).__init__(vocab_size, embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
 
    def forward(self, x, hidden=None):
        x = self.embedding(x)
        output, hidden = self.lstm(x, hidden)
        logits = self.fc_out(output)
        return logits
 
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
 
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
       
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
       
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)
 
    def forward(self, x):
        # x shape: [batch, seq_length, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
 
class TransformerLanguageModel(BaseLanguageModel):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=512, num_layers=2, nhead=2, max_seq_length=512, dropout=0.1):
        # For Transformer, we use embedding_dim for both token and hidden sizes.
        super(TransformerLanguageModel, self).__init__(vocab_size, embedding_dim, embedding_dim)
        # Replace learned positional embedding with sinusoidal PositionalEncoding
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout=dropout, max_len=max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)
        self.max_seq_length = max_seq_length
 
    def forward(self, x):
        # x shape: [batch, seq_length]
        x = self.embedding(x)                  # token embeddings: [batch, seq_length, embedding_dim]
        # Ensure x doesn't exceed max_seq_length
        x = x[:, :self.max_seq_length, :]
        x = self.positional_encoding(x)        # add sinusoidal positional encoding
        output = self.transformer_encoder(x)   # apply transformer encoder layers
        logits = self.fc_out(output)          # project to vocabulary size
        return logits

    def prompt(self, prompt_text, sp, max_seq_length=50, temperature=1.0, device="cpu"):
        token_ids = sp.encode_as_ids(prompt_text)
        # Truncate input if it exceeds max_seq_length
        if len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)
        self.eval()
        generated = token_ids.copy()
        with torch.no_grad():
            for _ in range(max_seq_length):
                if len(generated) >= self.max_seq_length:
                    break
                logits = self.forward(input_ids)
                next_token_logits = logits[:, -1, :]
                next_token = self.sample_next(next_token_logits, temperature)
                next_token_id = next_token.item()
                generated.append(next_token_id)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)
        return sp.decode_ids(generated)