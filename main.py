import time
from collections import defaultdict
import torch
from torch.utils.data import DataLoader, random_split
 
from data_processing import (
    train_sentencepiece_tokenizer_from_jsonl,
    load_tokenizer,
    JSONLanguageModelDataset
)
from language_models import RNNLanguageModel, LSTMLanguageModel, TransformerLanguageModel
from training import train_model, calculate_perplexity, compute_bleu_score, plot_loss_curves
 
def main():
    # Define file paths for JSONL files
    train_jsonl_file = "data/train.jsonl"   # Update path if needed
    test_jsonl_file = "data/test.jsonl"     # Update path if needed
 
    # Train SentencePiece tokenizer from the training JSONL file.
    train_sentencepiece_tokenizer_from_jsonl(train_jsonl_file, model_prefix="spm_model", vocab_size=10000, separator=" ")
 
    # Load the trained SentencePiece model.
    sp = load_tokenizer("spm_model.model")
 
    # Create the training dataset directly from train.jsonl.
    seq_length = 128
    full_train_dataset = JSONLanguageModelDataset(train_jsonl_file, tokenizer=sp, seq_length=seq_length, separator=" ")
 
    # Optionally split the training dataset into train and validation sets.
    valid_size = int(0.2 * len(full_train_dataset))
    train_size = len(full_train_dataset) - valid_size
    train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size])
   
    # Create the testing dataset from test.jsonl.
    test_dataset = JSONLanguageModelDataset(test_jsonl_file, tokenizer=sp, seq_length=seq_length, separator=" ")
 
    # Create DataLoaders.
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
 
    # Define model hyperparameters.
    vocab_size = sp.get_piece_size()
    embedding_dim, hidden_dim = 512, 512
    
    # Use GPU if available, else use MPS if available, otherwise fallback to CPU.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
 
    # Initialize models dictionary.
    models = {
        "RNN": RNNLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers=2),
        "LSTM": LSTMLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers=2),
        "Transformer": TransformerLanguageModel(vocab_size, embedding_dim, hidden_dim, num_layers=4, nhead=4, max_seq_length=seq_length)
    }
   
    results = defaultdict(dict)

    for model_name, model in models.items():
        model.load_state_dict(torch.load(f"{model_name}.pth", map_location=device))
        # # send model to device
        model.to(device)
        results[model_name]['model'] = model
   
    # for model_name, model in models.items():
    #     print(f"\nTraining {model_name} model...")
    #     trained_model, train_losses, valid_losses = train_model(
    #         model, train_loader, valid_loader, device,
    #         epochs=30, lr=5e-4, early_stopping_patience=3,
    #         model_name=model_name
    #     )
    #     # save the model
    #     torch.save(trained_model.state_dict(), f"{model_name}.pth")
    #     test_perplexity = calculate_perplexity(trained_model, test_loader, device)
    #     print(f"{model_name} Test Perplexity: {test_perplexity:.2f}")
    #     plot_loss_curves(train_losses, valid_losses, model_name)
    #     results[model_name]['model'] = trained_model
    #     results[model_name]['perplexity'] = test_perplexity
 
    # Text Generation and BLEU evaluation.
    prompt_text = "and the youth dressed himself"
    print("\nText Generation:")
    for model_name, info in results.items():
        print(f"\n[{model_name}]")
        model = info['model']
        generated_text = model.prompt(prompt_text, sp, max_seq_length=50, temperature=0.9
                                      , device=device)
        print(f"\n[{model_name}] Generated Text:")
        print(generated_text)
        reference_text = prompt_text + "with a facility his valet de chambre had"
        bleu = compute_bleu_score(reference_text, generated_text)
        results[model_name]['BLEU'] = bleu
        print(f"{model_name} BLEU Score: {bleu:.4f}")
 
    # Print summary of evaluation.
    # print("\nSummary of model evaluation:")
    # for model_name, metrics in results.items():
    #     print(f"{model_name}: Perplexity = {metrics['perplexity']:.2f}, BLEU Score = {metrics['BLEU']:.4f}")
 
if __name__ == '__main__':
    start_time = time.time()
    main()
    print(f"\nTotal Execution Time: {(time.time() - start_time)/60:.2f} minutes.")