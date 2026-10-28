import itertools
import random
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.amp import autocast
from transformers import AutoModel, AutoTokenizer

# 1. Model Architecture Definition (Compatible with Uncertainty Weights)
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

# 2. De-Novo Peptide Generators
def calc_charge(seq):
    return seq.count('K') + seq.count('R') - seq.count('D') - seq.count('E')

def calc_hydrophobicity(seq):
    hydrophobic = set('AILMFWV')
    return sum(1 for aa in seq if aa in hydrophobic) / len(seq)

def generate_denovo_pool(n_samples=500):
    candidates = set()
    
    # Strategy A: Alternating Amphipathic Helices [H-H-P-P or H-P-P-H]
    hydro_aa = ['L', 'I', 'F', 'W', 'V', 'A']
    pos_aa = ['K', 'R']
    neutral_polar = ['G', 'S', 'T']
    
    for _ in range(n_samples // 3):
        length = random.choice([12, 14, 16, 18])
        seq = []
        for i in range(length):
            # Helical face pattern approximation
            if (i % 7) in [0, 3, 4]:
                seq.append(random.choice(hydro_aa))
            elif (i % 7) in [1, 5]:
                seq.append(random.choice(pos_aa))
            else:
                seq.append(random.choice(neutral_polar + pos_aa))
        s = "".join(seq)
        if 2 <= calc_charge(s) <= 6 and 0.40 <= calc_hydrophobicity(s) <= 0.60:
            candidates.add(s)

    # Strategy B: Trp-Arg (WR) and Leu-Lys (LK) Rich Motifs
    motifs = ['RW', 'WR', 'RRW', 'WWR', 'KL', 'LK', 'KKL', 'LLK']
    for _ in range(n_samples // 3):
        length = random.choice([8, 10, 12, 14])
        s = ""
        while len(s) < length:
            s += random.choice(motifs)
        s = s[:length]
        if 2 <= calc_charge(s) <= 6:
            candidates.add(s)

    # Strategy C: Template Mutations (Magainin core: GIGKFLHSAKKFGKAFVGEIMNS)
    template = "GIGKFLHSAKKFGKAF"
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    for _ in range(n_samples // 3):
        seq_list = list(template)
        num_mutations = random.choice([1, 2, 3])
        positions = random.sample(range(len(template)), num_mutations)
        for pos in positions:
            seq_list[pos] = random.choice(amino_acids)
        s = "".join(seq_list)
        if 1 <= calc_charge(s) <= 6 and 0.35 <= calc_hydrophobicity(s) <= 0.65:
            candidates.add(s)

    return list(candidates)

# 3. Batch Screening Function
def run_screening(sequences, model_path="best_uncertainty_multitask.pt", batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for screening: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t30_150M_UR50D")
    model = MultiTaskESM2LoRA(lora_r=8, lora_alpha=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model.eval()
    
    all_pmic, all_hc50 = [], []
    
    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]
        tokens = tokenizer(
            batch_seqs,
            truncation=True,
            padding=True,
            max_length=60,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            with autocast('cuda'):
                pred_p, pred_t = model(tokens['input_ids'], tokens['attention_mask'])
                
        all_pmic.extend(pred_p.float().cpu().numpy().tolist())
        all_hc50.extend(pred_t.float().cpu().numpy().tolist())
        
    pmic_arr = np.array(all_pmic)
    hc50_arr = np.array(all_hc50)
    
    # Metric calculations
    mic_est = np.power(10.0, -pmic_arr)
    hc50_est = np.power(10.0, hc50_arr)
    ti_score = hc50_est / (mic_est + 1e-6)
    
    df = pd.DataFrame({
        "Sequence": sequences,
        "Length": [len(s) for s in sequences],
        "Net_Charge": [calc_charge(s) for s in sequences],
        "Hydrophobic_Ratio": [np.round(calc_hydrophobicity(s), 2) for s in sequences],
        "Pred_pMIC": np.round(pmic_arr, 4),
        "Pred_MIC_ug_ml": np.round(mic_est, 2),
        "Pred_log_HC50": np.round(hc50_arr, 4),
        "Pred_HC50_ug_ml": np.round(hc50_est, 2),
        "Therapeutic_Index": np.round(ti_score, 2)
    })
    
    return df.sort_values(by="Therapeutic_Index", ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    print("Generating De-Novo Candidate Pool...")
    pool = generate_denovo_pool(n_samples=1000)
    print(f"Total Unique Valid Candidates Generated: {len(pool)}")
    
    print("\nScreening Candidates through Multi-Task ESM-2 LoRA Pipeline...")
    results_df = run_screening(pool)
    
    results_df.to_csv("screened_denovo_leads.csv", index=False)
    print("\nSaved full results to screened_denovo_leads.csv")
    
    print("\n--- TOP 10 HIT CANDIDATES (HIGHEST THERAPEUTIC INDEX) ---")
    print(results_df.head(10)[['Sequence', 'Length', 'Net_Charge', 'Hydrophobic_Ratio', 'Pred_MIC_ug_ml', 'Pred_HC50_ug_ml', 'Therapeutic_Index']].to_string())
    
    print("\n--- BOTTOM 5 TOXIC/INACTIVE CANDIDATES (DE-PRIORITIZED) ---")
    print(results_df.tail(5)[['Sequence', 'Length', 'Net_Charge', 'Hydrophobic_Ratio', 'Pred_MIC_ug_ml', 'Pred_HC50_ug_ml', 'Therapeutic_Index']].to_string())