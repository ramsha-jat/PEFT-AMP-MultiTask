import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from torch.cuda.amp import autocast, GradScaler
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Dataset Loader
class AMPDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=60, target_col='EC_pMIC'):
        self.df = pd.read_csv(csv_path).dropna(subset=['SEQUENCE', target_col])
        self.sequences = self.df['SEQUENCE'].tolist()
        self.labels = self.df[target_col].values.astype(np.float32)
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

# 2. ESM-2 + LoRA Architecture (Memory Optimized for 10 GB)
class ESM2LoRARegressor(nn.Module):
    def __init__(self, model_name="facebook/esm2_t30_150M_UR50D", lora_r=8, lora_alpha=16):
        super().__init__()
        base_model = AutoModel.from_pretrained(model_name)
        
        # PEFT / LoRA on attention projection layers
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            bias="none"
        )
        self.encoder = get_peft_model(base_model, peft_config)
        hidden_dim = self.encoder.config.hidden_size
        
        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        
        # Mean pooling
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        return self.regressor(pooled).squeeze(-1)

# 3. Evaluation Function
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)
            
            with autocast():
                out = model(input_ids, attention_mask)
                
            preds.extend(out.cpu().numpy().tolist())
            targets.extend(target.cpu().numpy().tolist())
            
    preds, targets = np.array(preds), np.array(targets)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    mae = mean_absolute_error(targets, preds)
    spearman, _ = spearmanr(targets, preds)
    pearson, _ = pearsonr(targets, preds)
    return rmse, mae, spearman, pearson

# 4. Main Training Routine
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_name = "facebook/esm2_t30_150M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_path = "eval_prev_works/AMP_regression_EC_SA/data/train-EC.csv"
    test_path = "eval_prev_works/AMP_regression_EC_SA/data/test-EC.csv"
    
    train_dataset = AMPDataset(train_path, tokenizer)
    test_dataset = AMPDataset(test_path, tokenizer)
    
    # Batch size 16 easily fits within 10 GB VRAM with fp16
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    model = ESM2LoRARegressor(model_name=model_name, lora_r=8).to(device)
    model.encoder.print_trainable_parameters()
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scaler = GradScaler()
    
    epochs = 15
    print("\nStarting Training on EC Potency (pMIC)...")
    best_spearman = -1.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target = batch['target'].to(device)
            
            with autocast():
                pred = model(input_ids, attention_mask)
                loss = criterion(pred, target)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item() * len(target)
            
        train_rmse = np.sqrt(total_loss / len(train_dataset))
        test_rmse, test_mae, test_spearman, test_pearson = evaluate(model, test_loader, device)
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train RMSE: {train_rmse:.4f} | "
              f"Test RMSE: {test_rmse:.4f} | Pearson: {test_pearson:.4f} | Spearman: {test_spearman:.4f}")
        
        if test_spearman > best_spearman:
            best_spearman = test_spearman
            torch.save(model.state_dict(), "best_esm2_lora_ec.pt")

    print(f"\nBest Test Spearman: {best_spearman:.4f} (Weights saved to best_esm2_lora_ec.pt)")

if __name__ == "__main__":
    main()
