import math
import time
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import matplotlib.pyplot as plt
import torch.nn as nn
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
from tqdm import tqdm
 
def train_model(model, train_loader, valid_loader, device, epochs=30, lr=1e-3, early_stopping_patience=3, model_name=None):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
 
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    patience_counter = 0
 
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch+1}/{epochs}...")
        model.train()
        epoch_train_loss = 0
        for x, y in tqdm(train_loader, desc="Training", total=len(train_loader)):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()


        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        model.eval()
        epoch_valid_loss = 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
                epoch_valid_loss += loss.item()
        avg_valid_loss = epoch_valid_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)
        scheduler.step(avg_valid_loss)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Valid Loss={avg_valid_loss:.4f}")
 
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break
 
    model.load_state_dict(best_model_state)
    # save the model
    if model_name:
        torch.save(model.state_dict(), f"{model_name}.pth")
    return model, train_losses, valid_losses
 
def calculate_perplexity(model, data_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total_loss, total_tokens = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, model.vocab_size), y.view(-1))
            total_loss += loss.item()
            total_tokens += y.numel()
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity
 
def compute_bleu_score(reference, hypothesis):
    smoothie = SmoothingFunction().method4
    score = sentence_bleu([reference.split()], hypothesis.split(), smoothing_function=smoothie)
    return score
 
def plot_loss_curves(train_losses, valid_losses, model_name):
    plt.figure()
    plt.plot(train_losses, label="Training Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.title(f"{model_name} Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{model_name}_loss_curve.png")
    plt.close()
    print(f"Saved loss curve as {model_name}_loss_curve.png")