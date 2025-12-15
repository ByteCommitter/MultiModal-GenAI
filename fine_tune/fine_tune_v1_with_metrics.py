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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')

# ---------- CONFIG ----------
DATA_ROOT = "/home/dipanjan/rugraj/DIAC-WOZ/"
TRAIN_SPLIT = os.path.join(DATA_ROOT, "train_split_Depression_AVEC2017.csv")
DEV_SPLIT   = os.path.join(DATA_ROOT, "dev_split_Depression_AVEC2017.csv")

BATCH_SIZE = 4  # Reduced for stability
LR = 2e-5
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
GRAD_CLIP_NORM = 1.0
SAVE_MODEL = False  # Set to False to disable model saving completely
# ----------------------------

# ---------- HELPERS ----------
def read_transcript(session_folder, participant_only=True):
    """Read transcript with robust error handling"""
    file_candidate = glob.glob(os.path.join(session_folder, "*_TRANSCRIPT.csv"))
    if not file_candidate:
        return ""
    fp = file_candidate[0]
    
    try:
        df = pd.read_csv(fp, sep='\t', dtype=str, quoting=3)
        if 'speaker' in df.columns and 'value' in df.columns:
            if participant_only:
                mask = df['speaker'].str.lower() != 'ellie'
                texts = df.loc[mask, 'value'].fillna('')
            else:
                texts = df['value'].fillna('')
            return " ".join(texts.tolist())
    except Exception:
        pass
    
    try:
        df = pd.read_csv(fp, sep='\t', header=None, dtype=str, quoting=3, engine='python')
        if df.shape[1] >= 3:
            speaker_col = df.columns[2]
            utter_col = df.columns[-1]
            if participant_only:
                mask = ~df[speaker_col].str.lower().str.contains('ellie', na=False)
                texts = df.loc[mask, utter_col].fillna('').astype(str)
            else:
                texts = df[utter_col].fillna('').astype(str)
            return " ".join(texts.tolist())
    except Exception:
        pass
    
    return ""


def load_covarep_features(session_folder, pooling="mean", ignore_vuv=True):
    """Load COVAREP audio features"""
    fp = glob.glob(os.path.join(session_folder, "*_COVAREP.csv"))
    if not fp:
        return None
    
    try:
        df = pd.read_csv(fp[0], index_col=False)
        col_lower = [c.lower() for c in df.columns]
        
        # Find and handle VUV column
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
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec
    except Exception as e:
        print(f"Error loading COVAREP: {e}")
        return None


def load_clnf_aus(session_folder, pooling="mean"):
    """Load CLNF facial action units"""
    fp = glob.glob(os.path.join(session_folder, "*_CLNF_AUs.csv"))
    if not fp:
        return None
    
    try:
        df = pd.read_csv(fp[0], index_col=False)
        cols = [c for c in df.columns if str(c).endswith('_r')]
        
        if not cols:
            numeric = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
            if numeric.size == 0:
                return None
            vec = np.nanmean(numeric, axis=0)
        else:
            numeric = df[cols].to_numpy(dtype=np.float32)
            if numeric.size == 0:
                return None
            vec = np.nanmean(numeric, axis=0) if pooling == "mean" else np.nanmedian(numeric, axis=0)
        
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec
    except Exception as e:
        print(f"Error loading CLNF: {e}")
        return None


def read_split_csv(split_csv):
    """Read split CSV and extract participant info"""
    df = pd.read_csv(split_csv)
    
    print(f"\n{'='*60}")
    print(f"Reading: {os.path.basename(split_csv)}")
    print(f"{'='*60}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    out = pd.DataFrame()
    
    # Find participant ID column
    possible_id_cols = [c for c in df.columns if any(x in c.lower() for x in ['participant', 'id'])]
    pid_col = possible_id_cols[0] if possible_id_cols else df.columns[0]
    out['participant'] = df[pid_col].astype(str).str.strip()
    
    # Find PHQ8 score column
    phq_cols = [c for c in df.columns if 'phq' in c.lower()]
    if phq_cols:
        phq_col = phq_cols[0]
        out['phq8'] = pd.to_numeric(df[phq_col], errors='coerce')
        valid_phq8 = out['phq8'].notna().sum()
        print(f"PHQ8 column '{phq_col}' found: {valid_phq8}/{len(out)} valid scores")
    else:
        print("WARNING: No PHQ8 column found!")
        out['phq8'] = np.nan
    
    out['session_folder'] = out['participant'].apply(
        lambda x: f"{x}_P" if not str(x).endswith('_P') else x
    )
    
    return out


def build_dataset_index(split_df):
    """Build dataset index with multimodal features"""
    rows = []
    valid_phq8_count = 0
    total_count = 0
    skipped = {'no_folder': 0, 'no_features': 0, 'no_text': 0}
    
    for _, r in tqdm(split_df.iterrows(), total=len(split_df), desc="Building index"):
        total_count += 1
        pid = str(r['participant']).strip()
        
        # Find session folder
        sfolder = os.path.join(DATA_ROOT, str(r['session_folder']))
        if not os.path.isdir(sfolder):
            candidates = glob.glob(os.path.join(DATA_ROOT, f"{pid}*_P")) + \
                        glob.glob(os.path.join(DATA_ROOT, f"*{pid}*_P"))
            if candidates:
                sfolder = candidates[0]
            else:
                skipped['no_folder'] += 1
                continue
        
        # Load features
        transcript = read_transcript(sfolder, participant_only=True)
        if not transcript or transcript.strip() == "":
            skipped['no_text'] += 1
            continue
        
        covarep_vec = load_covarep_features(sfolder)
        clnf_vec = load_clnf_aus(sfolder)
        
        if covarep_vec is None and clnf_vec is None:
            skipped['no_features'] += 1
            continue
        
        # Use zero vectors if modality missing
        if covarep_vec is None:
            covarep_vec = np.zeros(74, dtype=np.float32)
        if clnf_vec is None:
            clnf_vec = np.zeros(35, dtype=np.float32)
        
        # Clean vectors
        covarep_vec = np.nan_to_num(covarep_vec.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        clnf_vec = np.nan_to_num(clnf_vec.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        
        # Handle PHQ8 score
        phq8 = r['phq8'] if not pd.isna(r['phq8']) else None
        if phq8 is not None:
            valid_phq8_count += 1
        
        rows.append({
            'participant': pid,
            'session_folder': sfolder,
            'transcript': transcript,
            'covarep': covarep_vec,
            'clnf_aus': clnf_vec,
            'phq8': phq8
        })
    
    print(f"\n{'='*60}")
    print("Dataset Building Summary")
    print(f"{'='*60}")
    print(f"Total processed: {total_count}")
    print(f"Valid samples created: {len(rows)}")
    print(f"Samples with PHQ8 scores: {valid_phq8_count}")
    print(f"\nSkipped samples:")
    print(f"  - No folder found: {skipped['no_folder']}")
    print(f"  - No features: {skipped['no_features']}")
    print(f"  - No text: {skipped['no_text']}")
    print(f"{'='*60}\n")
    
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
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['audio'] = torch.tensor(r['covarep'], dtype=torch.float32)
        item['video'] = torch.tensor(r['clnf_aus'], dtype=torch.float32)
        item['phq8'] = torch.tensor(
            r['phq8'] if r['phq8'] is not None else -999.0,
            dtype=torch.float32
        )
        item['participant'] = r['participant']
        
        return item


# ---------- LoRA BERT ----------
def get_lora_bert_base_for_text():
    """Get BERT model with LoRA"""
    bert = BertModel.from_pretrained("bert-base-uncased")
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "value"]
    )
    bert = get_peft_model(bert, lora_config)
    return bert


# ---------- RMSE Loss ----------
class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, preds, targets):
        return torch.sqrt(torch.mean((preds - targets) ** 2) + self.eps)


# ---------- Improved Model (v1 with LayerNorm) ----------
class MultiModalPHQ8LoRA(nn.Module):
    def __init__(self, audio_dim, video_dim, text_feat_dim=768, fused_dim=512):
        super().__init__()
        
        # Text encoder with LoRA
        self.text_encoder = get_lora_bert_base_for_text()
        
        # Projection layers with LayerNorm
        self.text_proj = nn.Linear(text_feat_dim, fused_dim)
        self.audio_proj = nn.Linear(audio_dim, fused_dim)
        self.video_proj = nn.Linear(video_dim, fused_dim)
        
        self.norm_text = nn.LayerNorm(fused_dim)
        self.norm_audio = nn.LayerNorm(fused_dim)
        self.norm_video = nn.LayerNorm(fused_dim)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim * 3, fused_dim),
            nn.ReLU(),
            nn.LayerNorm(fused_dim),
            nn.Dropout(0.3)
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, audio, video):
        # Text features
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        cls_vec = outputs.last_hidden_state[:, 0, :] if hasattr(outputs, 'last_hidden_state') \
                  else outputs.hidden_states[-1][:, 0, :]
        
        # Project and normalize each modality
        t = self.norm_text(self.text_proj(cls_vec))
        a = self.norm_audio(self.audio_proj(audio))
        v = self.norm_video(self.video_proj(video))
        
        # Fuse modalities
        fused = torch.cat([t, a, v], dim=1)
        h = self.fusion(fused)
        
        # Predict PHQ8 score
        phq8_pred = self.regressor(h).squeeze(1)
        
        return phq8_pred


# ---------- Evaluation Function ----------
def evaluate_model(model, loader, device):
    """Comprehensive evaluation with multiple metrics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_participants = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio = batch['audio'].to(device).float()
            video = batch['video'].to(device).float()
            phq8 = batch['phq8']
            
            # Clean inputs
            audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            video = torch.nan_to_num(video, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Filter valid labels
            mask = phq8 != -999.0
            if mask.sum() == 0:
                continue
            
            # Forward pass
            preds = model(
                input_ids[mask],
                attention_mask[mask],
                audio[mask],
                video[mask]
            )
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(phq8[mask].numpy().tolist())
            all_participants.extend([batch['participant'][i] for i in range(len(batch['participant'])) if mask[i]])
    
    if len(all_preds) == 0:
        print("WARNING: No valid predictions!")
        return None
    
    # Convert to numpy
    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    
    # Regression metrics
    mse = mean_squared_error(labels_arr, preds_arr)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels_arr, preds_arr)
    r2 = r2_score(labels_arr, preds_arr)
    
    # Classification metrics (PHQ8 >= 10 = depressed)
    true_binary = (labels_arr >= 10).astype(int)
    pred_binary = (preds_arr >= 10).astype(int)
    
    accuracy = accuracy_score(true_binary, pred_binary)
    f1 = f1_score(true_binary, pred_binary, zero_division=0)
    precision = precision_score(true_binary, pred_binary, zero_division=0)
    recall = recall_score(true_binary, pred_binary, zero_division=0)
    cm = confusion_matrix(true_binary, pred_binary)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'participant': all_participants,
        'true_phq8': labels_arr,
        'pred_phq8': preds_arr,
        'error': np.abs(labels_arr - preds_arr),
        'true_depressed': true_binary,
        'pred_depressed': pred_binary
    })
    
    metrics = {
        'num_samples': len(all_preds),
        'rmse': float(rmse),
        'mae': float(mae),
        'mse': float(mse),
        'r2': float(r2),
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'confusion_matrix': cm.tolist(),
        'results_df': results_df
    }
    
    return metrics


# ---------- Training Loop ----------
def train_loop(model, train_loader, val_loader, optimizer, device, epochs=50, save_model=True):
    """Training loop with early stopping and RMSE loss"""
    model.to(device)
    criterion = RMSELoss()
    
    best_val_rmse = float('inf')
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = None  # Store in memory
    
    history = {
        'train_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'val_r2': [],
        'val_accuracy': [],
        'val_f1': [],
        'lr': []
    }
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    for ep in range(epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        n_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}")
        for step, batch in enumerate(pbar):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio = batch['audio'].to(device).float()
            video = batch['video'].to(device).float()
            phq8 = batch['phq8'].to(device)
            
            # Clean inputs
            audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
            video = torch.nan_to_num(video, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Filter valid labels
            mask = phq8 != -999.0
            if mask.sum() == 0:
                continue
            
            # Forward pass
            preds = model(
                input_ids[mask],
                attention_mask[mask],
                audio[mask],
                video[mask]
            )
            
            # Check for NaN/Inf
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                print(f"Warning: NaN/Inf in predictions at epoch {ep+1} step {step}, skipping")
                continue
            
            # Calculate loss
            loss = criterion(preds, phq8[mask])
            
            # Backward pass
            try:
                loss.backward()
            except RuntimeError as e:
                print(f"Backward error at epoch {ep+1} step {step}: {e}, skipping")
                optimizer.zero_grad()
                continue
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            
            total_loss += loss.item() * mask.sum().item()
            n_samples += mask.sum().item()
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_loss / n_samples if n_samples > 0 else float('nan')
        
        # Validation phase
        val_metrics = evaluate_model(model, val_loader, device)
        
        if val_metrics is None:
            print("WARNING: No valid validation samples")
            continue
        
        # Update scheduler
        scheduler.step(val_metrics['rmse'])
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_r2'].append(val_metrics['r2'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print metrics
        print(f"\n{'='*60}")
        print(f"Epoch {ep+1}/{epochs}")
        print(f"{'='*60}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"\nRegression Metrics:")
        print(f"  Val RMSE: {val_metrics['rmse']:.4f}")
        print(f"  Val MAE:  {val_metrics['mae']:.4f}")
        print(f"  Val R²:   {val_metrics['r2']:.4f}")
        print(f"\nClassification Metrics (PHQ8 >= 10):")
        print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
        print(f"  F1-Score:  {val_metrics['f1']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall:    {val_metrics['recall']:.4f}")
        print(f"\nLearning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_epoch = ep + 1
            epochs_no_improve = 0
            
            # Store best model in memory
            import copy
            best_model_state = copy.deepcopy(model.state_dict())
            
            # Try to save to disk
            if save_model:
                out_dir = os.path.join(DATA_ROOT, "trained_models")
                try:
                    os.makedirs(out_dir, exist_ok=True)
                    
                    # Save with temporary file first (atomic write)
                    temp_path = os.path.join(out_dir, "best_model_v1_temp.pt")
                    final_path = os.path.join(out_dir, "best_model_v1.pt")
                    
                    torch.save({
                        'epoch': ep,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_metrics': {k: v for k, v in val_metrics.items() if k != 'results_df'}
                    }, temp_path)
                    
                    # Move temp file to final location
                    import shutil
                    shutil.move(temp_path, final_path)
                    print(f"  ✓ Model saved to disk")
                    
                except Exception as e:
                    print(f"  ⚠ Warning: Could not save model to disk: {e}")
                    print(f"  ℹ Model state kept in memory only")
            
            # Save predictions (smaller file)
            try:
                val_metrics['results_df'].to_csv(
                    os.path.join(out_dir, "best_val_predictions_v1.csv"),
                    index=False
                )
                print(f"  ✓ Predictions saved")
            except Exception as e:
                print(f"  ⚠ Warning: Could not save predictions: {e}")
            
            print(f"  ✓ New best model! (RMSE: {best_val_rmse:.4f})")
        else:
            epochs_no_improve += 1
            print(f"\nNo improvement ({epochs_no_improve}/{EARLY_STOPPING_PATIENCE})")
        
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n{'='*60}")
            print(f"Early stopping at epoch {ep+1}")
            print(f"Best RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")
            print(f"{'='*60}")
            break
    
    # Load best model from memory if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\n✓ Best model loaded from memory")
    
    return history, best_val_rmse


# ---------- Main ----------
def main():
    print("\n" + "="*60)
    print("LoRA-BERT Fusion Model (v1) for PHQ-8 Prediction")
    print("="*60)
    
    # Check disk space
    try:
        import shutil
        total, used, free = shutil.disk_usage(DATA_ROOT)
        free_gb = free // (2**30)
        print(f"\nDisk space available: {free_gb} GB")
        if free_gb < 1:
            print("⚠ WARNING: Low disk space! Model saving may fail.")
            save_model_flag = False
        else:
            save_model_flag = SAVE_MODEL
    except:
        save_model_flag = SAVE_MODEL
    
    # Load splits
    print("\n1. Loading dataset splits...")
    train_df = read_split_csv(TRAIN_SPLIT)
    dev_df = read_split_csv(DEV_SPLIT)
    
    # Build indices
    print("\n2. Building dataset indices...")
    train_rows = build_dataset_index(train_df)
    dev_rows = build_dataset_index(dev_df)
    
    if len(train_rows) == 0 or len(dev_rows) == 0:
        print("ERROR: No valid samples found!")
        return
    
    # Get dimensions
    audio_dim = train_rows[0]['covarep'].shape[0]
    video_dim = train_rows[0]['clnf_aus'].shape[0]
    
    print(f"\n3. Feature dimensions:")
    print(f"   Audio (COVAREP): {audio_dim}")
    print(f"   Video (CLNF AUs): {video_dim}")
    
    # Create datasets
    print("\n4. Creating datasets...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    train_dataset = DAICWozPHQ8Dataset(train_rows, tokenizer, max_len=MAX_LEN)
    dev_dataset = DAICWozPHQ8Dataset(dev_rows, tokenizer, max_len=MAX_LEN)
    
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    # Create model
    print("\n5. Initializing model...")
    model = MultiModalPHQ8LoRA(audio_dim=audio_dim, video_dim=video_dim)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Device: {DEVICE}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=0.01
    )
    
    # Train
    print("\n6. Starting training...")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LR}")
    print(f"   Early stopping patience: {EARLY_STOPPING_PATIENCE}")
    print(f"   Model saving: {'Enabled' if save_model_flag else 'Disabled (low disk space)'}")
    print()
    
    history, best_rmse = train_loop(
        model, train_loader, dev_loader, optimizer, DEVICE, 
        epochs=EPOCHS, save_model=save_model_flag
    )
    
    # Save results
    print("\n7. Saving final results...")
    out_dir = os.path.join(DATA_ROOT, "trained_models")
    os.makedirs(out_dir, exist_ok=True)
    
    # Save training history
    try:
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(out_dir, "training_history_v1.csv"), index=False)
        print("   ✓ Training history saved")
    except Exception as e:
        print(f"   ⚠ Could not save training history: {e}")
    
    # Final evaluation
    print("\n8. Final validation evaluation...")
    final_metrics = evaluate_model(model, dev_loader, DEVICE)
    
    if final_metrics is None:
        print("ERROR: Final evaluation failed!")
        return
    
    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nBest Validation Metrics:")
    print(f"  RMSE: {final_metrics['rmse']:.4f}")
    print(f"  MAE:  {final_metrics['mae']:.4f}")
    print(f"  R²:   {final_metrics['r2']:.4f}")
    print(f"\nDepression Classification (PHQ8 >= 10):")
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  F1-Score:  {final_metrics['f1']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {final_metrics['confusion_matrix']}")
    
    # Save metrics
    try:
        metrics_to_save = {k: v for k, v in final_metrics.items() if k != 'results_df'}
        with open(os.path.join(out_dir, "final_validation_metrics_v1.json"), 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        print("\n   ✓ Final metrics saved")
    except Exception as e:
        print(f"\n   ⚠ Could not save final metrics: {e}")
    
    print(f"\n{'='*60}")
    print("Files saved:")
    if save_model_flag:
        print(f"  - best_model_v1.pt (if disk write succeeded)")
    print(f"  - best_val_predictions_v1.csv")
    print(f"  - training_history_v1.csv")
    print(f"  - final_validation_metrics_v1.json")
    print(f"\nNote: Best model is loaded in memory and ready for use")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()