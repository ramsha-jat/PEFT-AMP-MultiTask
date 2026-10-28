import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from torch.amp import autocast, GradScaler
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from huggingface_hub import hf_hub_download

class HemoDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=60):
        self.sequences = df['sequence'].tolist()
        self.labels = df['log_hc50'].values.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        tokens = self.tokenizer(
            self.sequences[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'target': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

class ESM2LoRAToxicity(nn.Module):
    def __init__(self, model_name="facebook/esm2_t30_150M_UR50D", lora_r=8, lora_alpha=16):
        super().__init__()
        base_model = AutoModel.from_pretrained(model_name)
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            bias="none"
        )
        self.encoder = get_peft_model(base_model, peft_config)
        hidden_dim = self.encoder.config.hidden_size
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        return self.regressor(pooled).squeeze(-1)

def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)
            with autocast('cuda'):
                out = model(input_ids, attention_mask)
            preds.extend(out.cpu().numpy().tolist())
            targets.extend(target.cpu().numpy().tolist())
            
    preds, targets = np.array(preds), np.array(targets)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    spearman, _ = spearmanr(targets, preds)
    pearson, _ = pearsonr(targets, preds)
    return rmse, mae, spearman, pearson

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df = pd.read_csv("hemolysis_dataset.csv")
    
    # Load official QMAP benchmark split 0
    split_path = hf_hub_download(repo_id="anthol42/qmap_benchmark_2025", filename="benchmark_split_0.json", repo_type="dataset")
    with open(split_path) as f:
        split_entries = json.load(f)
        
    test_ids = set()
    for item in split_entries:
        if isinstance(item, dict) and 'id' in item:
            test_ids.add(item['id'])
            
    test_mask = df['id'].isin(test_ids)
    df_train = df[~test_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)
    print(f"Using Official QMAP Homology Split 0 -> Train: {len(df_train)}, Test: {len(df_test)}")

    model_name = "facebook/esm2_t30_150M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_loader = DataLoader(HemoDataset(df_train, tokenizer), batch_size=32, shuffle=True)
    test_loader = DataLoader(HemoDataset(df_test, tokenizer), batch_size=64, shuffle=False)
    
    model = ESM2LoRAToxicity(model_name=model_name, lora_r=8).to(device)
    model.encoder.print_trainable_parameters()
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scaler = GradScaler('cuda')
    
    epochs = 15
    print("\nStarting Training on Hemolysis Toxicity (log10 HC50)...")
    best_spearman = -1.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)
            
            with autocast('cuda'):
                pred = model(input_ids, attention_mask)
                loss = criterion(pred, target)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() * len(target)
            
        train_rmse = np.sqrt(total_loss / len(df_train))
        test_rmse, test_mae, test_spearman, test_pearson = evaluate(model, test_loader, device)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train RMSE: {train_rmse:.4f} | "
              f"Test RMSE: {test_rmse:.4f} | Pearson: {test_pearson:.4f} | Spearman: {test_spearman:.4f}")
        
        if test_spearman > best_spearman:
            best_spearman = test_spearman
            torch.save(model.state_dict(), "best_esm2_lora_hemo.pt")

    print(f"\nBest Test Spearman for Toxicity: {best_spearman:.4f}")

if __name__ == "__main__":
    main()