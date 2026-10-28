import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from torch.amp import autocast

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

def screen_candidates(sequences, model_path="best_multitask_esm2_lora.pt", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model_name = "facebook/esm2_t30_150M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = MultiTaskESM2LoRA(lora_r=8, lora_alpha=16).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    tokens = tokenizer(
        sequences,
        truncation=True,
        padding=True,
        max_length=60,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        with autocast('cuda'):
            pred_pmic, pred_log_hc50 = model(tokens['input_ids'], tokens['attention_mask'])
            
    # Cast to float32 explicitly before exponentiation to prevent inf overflow
    pmic_vals = pred_pmic.float().cpu().numpy()
    log_hc50_vals = pred_log_hc50.float().cpu().numpy()
    
    # Mathematical transformations to physical units:
    # pMIC = -log10(MIC)  => MIC = 10^(-pMIC)
    # log_hc50 = log10(HC50) => HC50 = 10^(log_hc50)
    mic_est = np.power(10.0, -pmic_vals)
    hc50_est = np.power(10.0, log_hc50_vals)
    ti_score = hc50_est / (mic_est + 1e-6)
    
    results = pd.DataFrame({
        "Sequence": sequences,
        "Length": [len(s) for s in sequences],
        "Pred_pMIC": np.round(pmic_vals, 4),
        "Pred_MIC_ug_ml": np.round(mic_est, 2),
        "Pred_log_HC50": np.round(log_hc50_vals, 4),
        "Pred_HC50_ug_ml": np.round(hc50_est, 2),
        "Therapeutic_Index": np.round(ti_score, 2)
    })
    
    return results.sort_values(by="Therapeutic_Index", ascending=False).reset_index(drop=True)

if __name__ == "__main__":
    test_peptides = [
        "RWRWRWRW",
        "RRWWRRWW",
        "FLPAIVGAAGSLVDEF",
        "KLLKKLLKKLLK",
        "GLFDIVKKVVGALGSL",
    ]
    
    df_ranked = screen_candidates(test_peptides)
    print("\n--- IN SILICO CANDIDATE SCREENING RESULTS ---")
    print(df_ranked.to_string())