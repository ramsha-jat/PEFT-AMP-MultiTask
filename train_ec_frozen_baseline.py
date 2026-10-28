import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.amp import autocast, GradScaler
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

class FrozenESMRegressor(nn.Module):
    def __init__(self, model_name="facebook/esm2_t30_150M_UR50D"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        # Freeze all backbone weights
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        hidden_dim = self.encoder.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
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
    spearman, _ = spearmanr(targets, preds)
    pearson, _ = pearsonr(targets, preds)
    return rmse, spearman, pearson

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "facebook/esm2_t30_150M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_loader = DataLoader(AMPDataset("eval_prev_works/AMP_regression_EC_SA/data/train-EC.csv", tokenizer), batch_size=32, shuffle=True)
    test_loader = DataLoader(AMPDataset("eval_prev_works/AMP_regression_EC_SA/data/test-EC.csv", tokenizer), batch_size=64, shuffle=False)
    
    model = FrozenESMRegressor(model_name).to(device)
    optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    scaler = GradScaler('cuda')
    
    print("\nRunning Frozen ESM-2 Baseline (10 Epochs)...")
    for epoch in range(1, 11):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            with autocast('cuda'):
                pred = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
                loss = criterion(pred, batch['target'].to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        test_rmse, test_spearman, test_pearson = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:02d} | Test RMSE: {test_rmse:.4f} | Pearson: {test_pearson:.4f} | Spearman: {test_spearman:.4f}")

if __name__ == "__main__":
    main()