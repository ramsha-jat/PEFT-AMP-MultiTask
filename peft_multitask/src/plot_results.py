import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from torch.amp import autocast
from scipy.stats import spearmanr, pearsonr

# 1. Dataset Loader for Test Evaluation
from train_multitask_lora import MultiTaskPeptideDataset, MultiTaskESM2LoRA

def generate_evaluation_plots(model_path="best_multitask_esm2_lora.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv("multitask_dataset_final.csv")
    df_test = df[df['split'] == 'test'].reset_index(drop=True)
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    test_loader = DataLoader(MultiTaskPeptideDataset(df_test, tokenizer), batch_size=64, shuffle=False)
    
    model = MultiTaskESM2LoRA(lora_r=8, lora_alpha=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    p_preds, p_targets = [], []
    t_preds, t_targets = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            with autocast('cuda'):
                pred_p, pred_t = model(input_ids, attention_mask)
            p_preds.extend(pred_p.float().cpu().numpy())
            p_targets.extend(batch['potency'].numpy())
            t_preds.extend(pred_t.float().cpu().numpy())
            t_targets.extend(batch['toxicity'].numpy())
            
    p_preds, p_targets = np.array(p_preds), np.array(p_targets)
    t_preds, t_targets = np.array(t_preds), np.array(t_targets)
    
    # Filter valid pairs
    valid_p = ~np.isnan(p_targets)
    vp_true, vp_pred = p_targets[valid_p], p_preds[valid_p]
    sp_p, _ = spearmanr(vp_true, vp_pred)
    pr_p, _ = pearsonr(vp_true, vp_pred)
    
    valid_t = ~np.isnan(t_targets)
    vt_true, vt_pred = t_targets[valid_t], t_preds[valid_t]
    sp_t, _ = spearmanr(vt_true, vt_pred)
    pr_t, _ = pearsonr(vt_true, vt_pred)
    
    # Plotting
    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Potency
    plt.subplot(1, 2, 1)
    plt.scatter(vp_true, vp_pred, alpha=0.4, color='#1f77b4', edgecolors='none', s=35)
    z = np.polyfit(vp_true, vp_pred, 1)
    p = np.poly1d(z)
    plt.plot(vp_true, p(vp_true), "r--", linewidth=1.5, label=f"Fit (Slope: {z[0]:.2f})")
    plt.xlabel("Ground Truth Potency (pMIC)", fontsize=11, fontweight='bold')
    plt.ylabel("Predicted Potency (pMIC)", fontsize=11, fontweight='bold')
    plt.title(f"E. coli Potency Prediction\nSpearman ρ: {sp_p:.4f} | Pearson r: {pr_p:.4f}", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    # Subplot 2: Toxicity
    plt.subplot(1, 2, 2)
    plt.scatter(vt_true, vt_pred, alpha=0.4, color='#d62728', edgecolors='none', s=35)
    z2 = np.polyfit(vt_true, vt_pred, 1)
    p2 = np.poly1d(z2)
    plt.plot(vt_true, p2(vt_true), "k--", linewidth=1.5, label=f"Fit (Slope: {z2[0]:.2f})")
    plt.xlabel("Ground Truth Toxicity (log10 HC50)", fontsize=11, fontweight='bold')
    plt.ylabel("Predicted Toxicity (log10 HC50)", fontsize=11, fontweight='bold')
    plt.title(f"Hemolysis Toxicity Prediction\nSpearman ρ: {sp_t:.4f} | Pearson r: {pr_t:.4f}", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("multitask_parity_plots.png", dpi=300)
    print("Parity plots successfully saved as multitask_parity_plots.png")

if __name__ == "__main__":
    generate_evaluation_plots()