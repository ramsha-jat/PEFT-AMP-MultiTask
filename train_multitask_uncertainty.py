import os
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

class MultiTaskPeptideDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=60):
        self.sequences = df['sequence'].tolist()
        self.potency = df['potency_pmic'].values.astype(np.float32)
        self.toxicity = df['toxicity_log_hc50'].values.astype(np.float32)
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
            'potency': torch.tensor(self.potency[idx], dtype=torch.float32),
            'toxicity': torch.tensor(self.toxicity[idx], dtype=torch.float32)
        }

class MultiTaskESM2LoRA(nn.Module):
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
        
        self.potency_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        self.toxicity_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        # Learnable log-variance parameters (s = log(sigma^2))
        self.log_var_p = nn.Parameter(torch.zeros(1))
        self.log_var_t = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * mask, dim=1)
        sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        
        pred_potency = self.potency_head(pooled).squeeze(-1)
        pred_toxicity = self.toxicity_head(pooled).squeeze(-1)
        return pred_potency, pred_toxicity

class UncertaintyMultiTaskLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred_p, pred_t, target_p, target_t, log_var_p, log_var_t):
        mask_p = ~torch.isnan(target_p)
        mask_t = ~torch.isnan(target_t)
        
        loss_p = self.mse(pred_p[mask_p], target_p[mask_p]).mean() if mask_p.sum() > 0 else torch.tensor(0.0, device=pred_p.device)
        loss_t = self.mse(pred_t[mask_t], target_t[mask_t]).mean() if mask_t.sum() > 0 else torch.tensor(0.0, device=pred_t.device)
        
        precision_p = torch.exp(-log_var_p)
        precision_t = torch.exp(-log_var_t)
        
        total_loss = 0.5 * precision_p * loss_p + 0.5 * precision_t * loss_t + 0.5 * (log_var_p + log_var_t)
        return total_loss, loss_p.detach(), loss_t.detach()

def evaluate_multitask(model, loader, device):
    model.eval()
    p_preds, p_targets = [], []
    t_preds, t_targets = [], []
    
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            potency = batch['potency'].to(device)
            toxicity = batch['toxicity'].to(device)
            
            with autocast('cuda'):
                pred_p, pred_t = model(input_ids, attention_mask)
                
            p_preds.extend(pred_p.float().cpu().numpy().tolist())
            p_targets.extend(potency.cpu().numpy().tolist())
            t_preds.extend(pred_t.float().cpu().numpy().tolist())
            t_targets.extend(toxicity.cpu().numpy().tolist())
            
    p_preds, p_targets = np.array(p_preds), np.array(p_targets)
    t_preds, t_targets = np.array(t_preds), np.array(t_targets)
    
    valid_p = ~np.isnan(p_targets)
    rmse_p = np.sqrt(mean_squared_error(p_targets[valid_p], p_preds[valid_p]))
    spearman_p, _ = spearmanr(p_targets[valid_p], p_preds[valid_p])
    pearson_p, _ = pearsonr(p_targets[valid_p], p_preds[valid_p])
    
    valid_t = ~np.isnan(t_targets)
    rmse_t = np.sqrt(mean_squared_error(t_targets[valid_t], t_preds[valid_t]))
    spearman_t, _ = spearmanr(t_targets[valid_t], t_preds[valid_t])
    pearson_t, _ = pearsonr(t_targets[valid_t], t_preds[valid_t])
    
    return (rmse_p, spearman_p, pearson_p), (rmse_t, spearman_t, pearson_t)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    df = pd.read_csv("multitask_dataset_final.csv")
    df_train = df[df['split'] == 'train'].reset_index(drop=True)
    df_test = df[df['split'] == 'test'].reset_index(drop=True)
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    train_loader = DataLoader(MultiTaskPeptideDataset(df_train, tokenizer), batch_size=32, shuffle=True)
    test_loader = DataLoader(MultiTaskPeptideDataset(df_test, tokenizer), batch_size=64, shuffle=False)
    
    model = MultiTaskESM2LoRA(lora_r=8, lora_alpha=16).to(device)
    criterion = UncertaintyMultiTaskLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    scaler = GradScaler('cuda')
    
    epochs = 15
    print("\nStarting Uncertainty-Weighted Multi-Task Training...")
    best_avg_spearman = -1.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            target_p = batch['potency'].to(device)
            target_t = batch['toxicity'].to(device)
            
            with autocast('cuda'):
                pred_p, pred_t = model(input_ids, attention_mask)
                loss, _, _ = criterion(pred_p, pred_t, target_p, target_t, model.log_var_p, model.log_var_t)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(target_p)
            
        train_loss = total_loss / len(df_train)
        (rmse_p, sp_p, pr_p), (rmse_t, sp_t, pr_t) = evaluate_multitask(model, test_loader, device)
        avg_sp = (sp_p + sp_t) / 2.0
        
        w_p = torch.exp(-model.log_var_p).item()
        w_t = torch.exp(-model.log_var_t).item()
        
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Weights (P/T): {w_p:.2f}/{w_t:.2f} || "
              f"POTENCY -> Sp: {sp_p:.4f}, Pr: {pr_p:.4f} || "
              f"TOXICITY -> Sp: {sp_t:.4f}, Pr: {pr_t:.4f}")

        if avg_sp > best_avg_spearman:
            best_avg_spearman = avg_sp
            torch.save(model.state_dict(), "best_uncertainty_multitask.pt")

    print(f"\nBest Average Spearman: {best_avg_spearman:.4f} (Saved to best_uncertainty_multitask.pt)")

if __name__ == "__main__":
    main()