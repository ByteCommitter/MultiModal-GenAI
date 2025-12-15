import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.metrics import mean_squared_error
from utils.metrics_utils import compute_regression_metrics, compute_classification_metrics, print_metrics_summary
import json

# ---------- CONFIG ----------
DATA_ROOT = "/home/dipanjan/rugraj/DIAC-WOZ/"
TRAIN_SPLIT = os.path.join(DATA_ROOT, "train_split_Depression_AVEC2017.csv")
DEV_SPLIT   = os.path.join(DATA_ROOT, "dev_split_Depression_AVEC2017.csv")
TEST_SPLIT  = os.path.join(DATA_ROOT, "test_split_Depression_AVEC2017.csv")
SESSION_GLOB = os.path.join(DATA_ROOT, "*_P")
BATCH_SIZE = 8
LR = 2e-5                     # Lowered learning rate for stability
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5   # Stop early if no improvement after 5 epochs
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
GRAD_CLIP_NORM = 1.0
# ----------------------------

# ---------- HELPERS ----------
def read_transcript(session_folder, participant_only=True):
    file_candidate = glob.glob(os.path.join(session_folder, "*_TRANSCRIPT.csv"))
    if not file_candidate:
        return ""
    fp = file_candidate[0]
    try:
        df = pd.read_csv(fp, sep='\t', header=None, quoting=3, dtype=str, engine='python')
    except Exception:
        df = pd.read_csv(fp, sep=None, engine='python', dtype=str)
    try:
        dfh = pd.read_csv(fp, sep='\t', dtype=str, quoting=3)
        if 'speaker' in dfh.columns and 'utterance' in dfh.columns:
            if participant_only:
                part_texts = dfh[dfh['speaker'].str.lower() != 'ellie']['utterance'].fillna('')
            else:
                part_texts = dfh['utterance'].fillna('')
            return " ".join(part_texts.tolist())
    except Exception:
        pass
    if df.shape[1] >= 3:
        speaker_col = df.columns[2]
        utter_col = df.columns[-1]
        try:
            if participant_only:
                mask = ~df[speaker_col].str.lower().str.contains('ellie', na=False)
                texts = df.loc[mask, utter_col].fillna('').astype(str)
            else:
                texts = df[utter_col].fillna('').astype(str)
            return " ".join(texts.tolist())
        except Exception:
            return " ".join(df.iloc[:, -1].fillna('').astype(str).tolist())
    else:
        return " ".join(df.fillna('').astype(str).values.flatten().tolist())

def load_covarep_features(session_folder, pooling="mean", ignore_vuv=True):
    fp = glob.glob(os.path.join(session_folder, "*_COVAREP.csv"))
    if not fp:
        return None
    df = pd.read_csv(fp[0], index_col=False)
    col_lower = [c.lower() for c in df.columns]
    vuv_idx = None
    for i, c in enumerate(col_lower):
        if c == 'vuv' or 'vuv' in c:
            vuv_idx = df.columns[i]
            break
    if ignore_vuv and vuv_idx is not None:
        mask = df[vuv_idx] == 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c != vuv_idx]
        df.loc[mask, numeric_cols] = 0.0
    numeric = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    if numeric.size == 0:
        return None
    vec = np.nanmean(numeric, axis=0) if pooling == "mean" else np.nanmedian(numeric, axis=0)
    vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
    return vec

def load_clnf_aus(session_folder, pooling="mean"):
    fp = glob.glob(os.path.join(session_folder, "*_CLNF_AUs.csv"))
    if not fp:
        return None
    df = pd.read_csv(fp[0], index_col=False)
    cols = [c for c in df.columns if str(c).endswith('_r')]
    if not cols:
        numeric = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
        if numeric.size == 0:
            return None
        vec = np.nanmean(numeric, axis=0)
        return np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
    numeric = df[cols].to_numpy(dtype=np.float32)
    if numeric.size == 0:
        return None
    vec = np.nanmean(numeric, axis=0) if pooling == "mean" else np.nanmedian(numeric, axis=0)
    vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
    return vec

def read_split_csv(split_csv):
    """Modified to handle test set differently"""
    df = pd.read_csv(split_csv)
    
    print(f"\nReading {os.path.basename(split_csv)}:")
    print(f"Total rows: {len(df)}")
    print("Columns found:", df.columns.tolist())
    
    out = pd.DataFrame()
    out['participant'] = df['participant_ID' if 'participant_ID' in df.columns else df.columns[0]].astype(str)
    
    # For test set, we only need participant IDs, no PHQ8 scores
    if 'test' in split_csv.lower():
        print("\nTest set detected - no PHQ8 scores needed for predictions")
        out['phq8'] = None  # Use None instead of np.nan to clearly indicate prediction mode
    else:
        # For train/dev sets, get PHQ8 scores
        phq_cols = [c for c in df.columns if 'phq' in c.lower()]
        if phq_cols:
            phq_col = phq_cols[0]
            out['phq8'] = pd.to_numeric(df[phq_col], errors='coerce')
        else:
            print(f"❌ No PHQ8 column found in {os.path.basename(split_csv)}")
            return None
    
    out['session_folder'] = out['participant'].apply(lambda x: f"{x}_P" if not str(x).endswith('_P') else x)
    return out

def build_dataset_index(split_df):
    rows = []
    valid_phq8_count = 0
    total_count = 0
    
    for _, r in split_df.iterrows():
        total_count += 1
        sfolder = os.path.join(DATA_ROOT, str(r['session_folder']))
        if not os.path.isdir(sfolder):
            candidates = glob.glob(os.path.join(DATA_ROOT, f"{r['participant']}*_P")) + \
                        glob.glob(os.path.join(DATA_ROOT, f"*{r['participant']}*_P"))
            if candidates:
                sfolder = candidates[0]
            else:
                print(f"WARNING: session folder not found for {r['participant']}: expected {r['session_folder']}")
                continue
                
        # Explicitly handle PHQ8 score
        phq8 = r['phq8'] if not pd.isna(r['phq8']) else None
        if phq8 is not None:
            valid_phq8_count += 1
            
        # Load features
        transcript = read_transcript(sfolder, participant_only=True)
        covarep_vec = load_covarep_features(sfolder)
        clnf_vec = load_clnf_aus(sfolder)
        
        if covarep_vec is None and clnf_vec is None:
            print(f"WARNING: both covarep and clnf missing for {sfolder}, skipping.")
            continue
            
        # Initialize with zeros if missing
        if covarep_vec is None:
            covarep_vec = np.zeros(10, dtype=np.float32)
        if clnf_vec is None:
            clnf_vec = np.zeros(10, dtype=np.float32)
            
        # Clean vectors
        covarep_vec = np.nan_to_num(np.array(covarep_vec, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        clnf_vec = np.nan_to_num(np.array(clnf_vec, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        
        rows.append({
            'session_folder': sfolder,
            'transcript': transcript,
            'covarep': covarep_vec.astype(np.float32),
            'clnf_aus': clnf_vec.astype(np.float32),
            'phq8': phq8
        })
    
    print(f"\nDataset Statistics:")
    print(f"Total samples processed: {total_count}")
    print(f"Valid samples created: {len(rows)}")
    print(f"Samples with valid PHQ8 scores: {valid_phq8_count}")
    
    return rows

# ---------- Dataset ----------
class DAICWozPHQ8Dataset(Dataset):
    def __init__(self, rows, tokenizer, max_len=128):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        encoding = self.tokenizer(
            r['transcript'] if r['transcript'] else " ",
            padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['audio'] = torch.tensor(np.nan_to_num(r['covarep'], nan=0.0, posinf=1e6, neginf=-1e6), dtype=torch.float)
        item['video'] = torch.tensor(np.nan_to_num(r['clnf_aus'], nan=0.0, posinf=1e6, neginf=-1e6), dtype=torch.float)
        item['phq8'] = torch.tensor(r['phq8'] if r['phq8'] is not None else np.nan, dtype=torch.float)
        return item

# ---------- LoRA helper ----------
def get_lora_bert_base_for_text():
    bert = BertModel.from_pretrained("bert-base-uncased")
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    bert = get_peft_model(bert, lora_config)
    return bert

# ---------- Improved RMSE Loss ----------
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, preds, targets):
        return torch.sqrt(torch.mean((preds - targets) ** 2) + self.eps)

# ---------- Improved Model ----------
class MultiModalPHQ8LoRA(nn.Module):
    def __init__(self, audio_dim, video_dim, text_feat_dim=768, fused_dim=512):
        super().__init__()
        self.text_encoder = get_lora_bert_base_for_text()
        self.text_proj = nn.Linear(text_feat_dim, fused_dim)
        self.audio_proj = nn.Linear(audio_dim, fused_dim)
        self.video_proj = nn.Linear(video_dim, fused_dim)
        self.norm_text = nn.LayerNorm(fused_dim)
        self.norm_audio = nn.LayerNorm(fused_dim)
        self.norm_video = nn.LayerNorm(fused_dim)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim * 3, fused_dim),
            nn.ReLU(),
            nn.LayerNorm(fused_dim),     # Added normalization
            nn.Dropout(0.2)              # Increased dropout
        )
        self.regressor = nn.Linear(fused_dim, 1)

    def forward(self, input_ids, attention_mask, audio, video):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        cls_vec = outputs.last_hidden_state[:, 0, :] if hasattr(outputs, 'last_hidden_state') else outputs.hidden_states[-1][:, 0, :]
        t = self.norm_text(self.text_proj(cls_vec))
        a = self.norm_audio(self.audio_proj(audio))
        v = self.norm_video(self.video_proj(video))
        fused = torch.cat([t, a, v], dim=1)
        h = self.fusion(fused)
        phq8_pred = self.regressor(h).squeeze(1)
        return phq8_pred

# ---------- Improved Training loop with Early Stopping ----------
def train_loop(model, train_loader, val_loader, optimizer, device, epochs=3):
    model.to(device)
    criterion = RMSELoss()
    history = {'train_loss': [], 'val_rmse': []}
    best_val_rmse = float('inf')
    epochs_no_improve = 0

    torch.autograd.set_detect_anomaly(True)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {ep + 1}")):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio = batch['audio'].to(device).float()
            video = batch['video'].to(device).float()
            audio = torch.nan_to_num(audio, nan=0.0, posinf=1e6, neginf=-1e6)
            video = torch.nan_to_num(video, nan=0.0, posinf=1e6, neginf=-1e6)
            phq8 = batch['phq8'].to(device)
            mask = ~torch.isnan(phq8)
            if mask.sum() == 0:
                continue
            input_ids_m = input_ids[mask]
            attention_mask_m = attention_mask[mask]
            audio_m = audio[mask]
            video_m = video[mask]
            phq8_m = phq8[mask]
            preds = model(input_ids_m, attention_mask_m, audio_m, video_m)
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                print(f"Warning: NaN/Inf in preds at epoch {ep + 1} step {step}, skipping batch")
                continue
            loss = criterion(preds, phq8_m)
            try:
                loss.backward()
            except RuntimeError as e:
                print(f"Backward error at epoch {ep+1} step {step}: {e}. Skipping batch.")
                optimizer.zero_grad()
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            total_loss += loss.item() * mask.sum().item()
            n += mask.sum().item()
        avg_loss = total_loss / n if n > 0 else float('nan')
        history['train_loss'].append(avg_loss)
        # validation
        model.eval()
        ys, ypred = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                audio = batch['audio'].to(device).float()
                video = batch['video'].to(device).float()
                audio = torch.nan_to_num(audio, nan=0.0, posinf=1e6, neginf=-1e6)
                video = torch.nan_to_num(video, nan=0.0, posinf=1e6, neginf=-1e6)
                phq8 = batch['phq8'].to(device)
                mask = ~torch.isnan(phq8)
                if mask.sum() == 0:
                    continue
                preds = model(input_ids[mask], attention_mask[mask], audio[mask], video[mask])
                preds = torch.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
                ys.extend(phq8[mask].cpu().numpy().tolist())
                ypred.extend(preds.cpu().numpy().tolist())
        if len(ys) > 0:
            ys_arr = np.nan_to_num(np.array(ys, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
            ypred_arr = np.nan_to_num(np.array(ypred, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
            rmse = math.sqrt(mean_squared_error(ys_arr, ypred_arr))
        else:
            rmse = float('nan')
        history['val_rmse'].append(rmse)
        print(f"Epoch {ep + 1}/{epochs} train_loss={avg_loss:.4f} val_RMSE={rmse:.4f}")
        # --- early stopping
        if rmse < best_val_rmse:
            best_val_rmse = rmse
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(DATA_ROOT, "trained_models", "best_model.pt"))
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {ep + 1} due to no improvement in validation RMSE.")
                break
    return history

def evaluate_model(model, loader, device, is_validation=False):
    """Enhanced evaluation function that works for both validation and test sets"""
    model.eval()
    predictions = []
    true_values = []
    valid_count = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio = batch['audio'].to(device).float()
            video = batch['video'].to(device).float()
            phq8 = batch['phq8'].to(device)
            
            # Only evaluate samples with valid PHQ8 scores for validation
            if is_validation:
                mask = ~torch.isnan(phq8)
                if mask.sum() == 0:
                    continue
                
                # Get predictions for valid samples
                preds = model(input_ids[mask], attention_mask[mask], 
                            audio[mask], video[mask])
                
                predictions.extend(preds.cpu().numpy())
                true_values.extend(phq8[mask].cpu().numpy())
                valid_count += mask.sum().item()
            else:
                # For test set, just get predictions
                preds = model(input_ids, attention_mask, audio, video)
                predictions.extend(preds.cpu().numpy())
    
    if is_validation:
        # Compute validation metrics
        predictions = np.array(predictions)
        true_values = np.array(true_values)
        
        print("\nValidation Metrics:")
        print(f"Number of valid samples: {valid_count}")
        print(f"Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
        print(f"True values range: [{true_values.min():.2f}, {true_values.max():.2f}]")
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(true_values, predictions)),
            'mae': np.mean(np.abs(true_values - predictions)),
            'r2': compute_regression_metrics(true_values, predictions)['r2'],
            'num_samples': valid_count
        }
        
        # Add classification metrics for depression detection (PHQ8 ≥ 10)
        true_binary = (true_values >= 10).astype(int)
        pred_binary = (predictions >= 10).astype(int)
        
        cls_metrics = compute_classification_metrics(true_binary, pred_binary)
        metrics.update({
            'accuracy': cls_metrics['accuracy'],
            'f1': cls_metrics['f1'],
            'precision': cls_metrics['precision'],
            'recall': cls_metrics['recall']
        })
        
        return metrics
    else:
        # For test set, just return predictions
        return np.array(predictions)

# ---------- Main ----------
def main():
    train_df = read_split_csv(TRAIN_SPLIT)
    dev_df = read_split_csv(DEV_SPLIT)
    print("Building training index (this can take a while)...")
    train_rows = build_dataset_index(train_df)
    print("Building dev index...")
    dev_rows = build_dataset_index(dev_df)
    audio_dim = train_rows[0]['covarep'].shape[0]
    video_dim = train_rows[0]['clnf_aus'].shape[0]
    print(f"Detected audio_dim={audio_dim}, video_dim={video_dim}")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = DAICWozPHQ8Dataset(train_rows, tokenizer, max_len=MAX_LEN)
    dev_dataset = DAICWozPHQ8Dataset(dev_rows, tokenizer, max_len=MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    model = MultiModalPHQ8LoRA(audio_dim=audio_dim, video_dim=video_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    out_dir = os.path.join(DATA_ROOT, "trained_models")
    os.makedirs(out_dir, exist_ok=True)
    history = train_loop(model, train_loader, dev_loader, optimizer, DEVICE, epochs=EPOCHS)
    # Reload best model and save history/final model for reproducibility
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pt")))
    torch.save(model.state_dict(), os.path.join(out_dir, "multimodal_phq8_lora_improved.pt"))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "train_history_improved.csv"), index=False)
    print("Training complete. Model and history saved to:", out_dir)
    
    # Evaluate on validation set with full metrics
    print("\nEvaluating on validation set...")
    val_metrics = evaluate_model(model, dev_loader, DEVICE, is_validation=True)
    
    print("\nValidation Results:")
    print(f"RMSE: {val_metrics['rmse']:.4f}")
    print(f"MAE: {val_metrics['mae']:.4f}")
    print(f"R²: {val_metrics['r2']:.4f}")
    print(f"\nDepression Detection (PHQ8 ≥ 10):")
    print(f"Accuracy: {val_metrics['accuracy']:.4f}")
    print(f"F1-Score: {val_metrics['f1']:.4f}")
    print(f"Precision: {val_metrics['precision']:.4f}")
    print(f"Recall: {val_metrics['recall']:.4f}")
    
    # Save validation metrics
    metrics_path = os.path.join(out_dir, "validation_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(val_metrics, f, indent=2)
    print(f"\nValidation metrics saved to: {metrics_path}")
    
    # Continue with test set evaluation if needed...
    # ...rest of existing code...

if __name__ == "__main__":
    main()