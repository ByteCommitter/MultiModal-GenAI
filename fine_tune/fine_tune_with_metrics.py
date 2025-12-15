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
import warnings
warnings.filterwarnings('ignore')

# ---------- CONFIG ----------
DATA_ROOT = "/home/dipanjan/rugraj/DIAC-WOZ/"
TRAIN_SPLIT = os.path.join(DATA_ROOT, "train_split_Depression_AVEC2017.csv")
DEV_SPLIT   = os.path.join(DATA_ROOT, "dev_split_Depression_AVEC2017.csv")
TEST_SPLIT  = os.path.join(DATA_ROOT, "test_split_Depression_AVEC2017.csv")

BATCH_SIZE = 4  # Reduced for stability
LR = 2e-5  # Lower learning rate
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
GRAD_CLIP_NORM = 1.0
WARMUP_EPOCHS = 3
PATIENCE = 10  # Early stopping patience
# ----------------------------

def read_transcript(session_folder, participant_only=True):
    """Read transcript with robust error handling"""
    file_candidate = glob.glob(os.path.join(session_folder, "*_TRANSCRIPT.csv"))
    if not file_candidate:
        return ""
    fp = file_candidate[0]
    
    # Try multiple reading strategies
    try:
        df = pd.read_csv(fp, sep='\t', dtype=str, quoting=3)
        if 'speaker' in df.columns and 'value' in df.columns:
            if participant_only:
                mask = df['speaker'].str.lower() != 'ellie'
                texts = df.loc[mask, 'value'].fillna('')
            else:
                texts = df['value'].fillna('')
            return " ".join(texts.tolist())
    except Exception as e:
        pass
    
    # Fallback: just concatenate all text
    try:
        df = pd.read_csv(fp, sep='\t', header=None, dtype=str, quoting=3)
        if df.shape[1] >= 3:
            return " ".join(df.iloc[:, -1].fillna('').astype(str).tolist())
    except Exception:
        pass
    
    return ""


def load_covarep_features(session_folder, pooling="mean"):
    """Load COVAREP audio features with robust handling"""
    fp = glob.glob(os.path.join(session_folder, "*_COVAREP.csv"))
    if not fp:
        return None
    
    try:
        df = pd.read_csv(fp[0])
        # Remove non-numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return None
        
        # Handle VUV (voiced/unvoiced) - set unvoiced frames to 0
        if 'VUV' in numeric_df.columns:
            mask = numeric_df['VUV'] == 0
            numeric_df = numeric_df.copy()
            numeric_df.loc[mask] = 0.0
        
        # Pool over time
        if pooling == "mean":
            vec = numeric_df.mean(axis=0).values
        else:
            vec = numeric_df.median(axis=0).values
        
        # Clean the vector
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec.astype(np.float32)
    
    except Exception as e:
        print(f"Error loading COVAREP from {fp[0]}: {e}")
        return None


def load_clnf_aus(session_folder, pooling="mean"):
    """Load CLNF facial action units with robust handling"""
    fp = glob.glob(os.path.join(session_folder, "*_CLNF_AUs.csv"))
    if not fp:
        return None
    
    try:
        df = pd.read_csv(fp[0])
        # Use only AU intensity columns (end with _r)
        au_cols = [c for c in df.columns if c.endswith('_r')]
        
        if not au_cols:
            # Fallback: use all numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return None
            au_data = numeric_df.values
        else:
            au_data = df[au_cols].values
        
        # Pool over time
        if pooling == "mean":
            vec = np.nanmean(au_data, axis=0)
        else:
            vec = np.nanmedian(au_data, axis=0)
        
        # Clean the vector
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec.astype(np.float32)
    
    except Exception as e:
        print(f"Error loading CLNF from {fp[0]}: {e}")
        return None


def read_split_csv(split_csv):
    """Read split CSV and extract participant IDs and PHQ-8 scores"""
    df = pd.read_csv(split_csv)
    
    # Find participant ID column
    possible_id_cols = [c for c in df.columns if any(x in c.lower() for x in ['participant', 'id'])]
    pid_col = possible_id_cols[0] if possible_id_cols else df.columns[0]
    
    # Find PHQ-8 score column
    phq_cols = [c for c in df.columns if 'phq' in c.lower()]
    phq_col = phq_cols[0] if phq_cols else None
    
    out = pd.DataFrame()
    out['participant'] = df[pid_col].astype(str).str.strip()
    
    if phq_col is not None:
        out['phq8'] = pd.to_numeric(df[phq_col], errors='coerce')
    else:
        out['phq8'] = np.nan
    
    return out


def build_dataset_index(split_df, data_root):
    """Build dataset index with multimodal features"""
    rows = []
    skipped = 0
    
    for _, r in tqdm(split_df.iterrows(), total=len(split_df), desc="Building index"):
        pid = str(r['participant']).strip()
        
        # Find session folder
        possible_folders = [
            os.path.join(data_root, f"{pid}_P"),
            os.path.join(data_root, pid),
        ]
        
        # Search for folder
        session_folder = None
        for pf in possible_folders:
            if os.path.isdir(pf):
                session_folder = pf
                break
        
        if session_folder is None:
            # Try glob search
            candidates = glob.glob(os.path.join(data_root, f"*{pid}*_P"))
            if candidates:
                session_folder = candidates[0]
        
        if session_folder is None or not os.path.isdir(session_folder):
            print(f"WARNING: Session folder not found for participant {pid}")
            skipped += 1
            continue
        
        # Load modalities
        transcript = read_transcript(session_folder, participant_only=True)
        covarep_vec = load_covarep_features(session_folder)
        clnf_vec = load_clnf_aus(session_folder)
        
        # Skip if missing critical data
        if not transcript or transcript.strip() == "":
            print(f"WARNING: Empty transcript for {pid}")
            skipped += 1
            continue
        
        # Use zero vectors if modality missing
        if covarep_vec is None:
            covarep_vec = np.zeros(74, dtype=np.float32)  # COVAREP standard size
            print(f"INFO: Using zero COVAREP for {pid}")
        
        if clnf_vec is None:
            clnf_vec = np.zeros(35, dtype=np.float32)  # CLNF AU standard size
            print(f"INFO: Using zero CLNF for {pid}")
        
        phq8 = r['phq8'] if not pd.isna(r['phq8']) else None
        
        rows.append({
            'participant': pid,
            'session_folder': session_folder,
            'transcript': transcript,
            'covarep': covarep_vec,
            'clnf_aus': clnf_vec,
            'phq8': phq8
        })
    
    print(f"\nDataset built: {len(rows)} samples, {skipped} skipped")
    return rows


class DAICWozPHQ8Dataset(Dataset):
    """Dataset for DAIC-WOZ multimodal data"""
    
    def __init__(self, rows, tokenizer, max_len=128):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Store feature dimensions
        self.audio_dim = rows[0]['covarep'].shape[0]
        self.video_dim = rows[0]['clnf_aus'].shape[0]
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        r = self.rows[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            r['transcript'],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        
        item = {k: v.squeeze(0) for k, v in encoding.items()}
        item['audio'] = torch.tensor(r['covarep'], dtype=torch.float32)
        item['video'] = torch.tensor(r['clnf_aus'], dtype=torch.float32)
        item['phq8'] = torch.tensor(r['phq8'] if r['phq8'] is not None else -999.0, dtype=torch.float32)
        item['participant'] = r['participant']
        
        return item


def get_lora_bert():
    """Get BERT model with LoRA for efficient fine-tuning"""
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
    bert.print_trainable_parameters()
    
    return bert


class MultiModalPHQ8LoRA(nn.Module):
    """LoRA-BERT Baseline Model"""
    
    def __init__(self, audio_dim, video_dim, text_feat_dim=768, fused_dim=256):
        super().__init__()
        
        # Text encoder with LoRA
        self.text_encoder = get_lora_bert()
        
        # Projection layers
        self.text_proj = nn.Sequential(
            nn.Linear(text_feat_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.video_proj = nn.Sequential(
            nn.Linear(video_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim * 3, fused_dim * 2),
            nn.LayerNorm(fused_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fused_dim * 2, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
    
    def forward(self, input_ids, attention_mask, audio, video):
        # Text features
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        cls_vec = outputs.last_hidden_state[:, 0, :]
        
        # Project each modality
        t = self.text_proj(cls_vec)
        a = self.audio_proj(audio)
        v = self.video_proj(video)
        
        # Fuse modalities
        fused = torch.cat([t, a, v], dim=1)
        h = self.fusion(fused)
        
        # Predict PHQ-8 score
        phq8_pred = self.regressor(h).squeeze(1)
        
        return phq8_pred


def evaluate(model, data_loader, device):
    """Comprehensive evaluation with multiple metrics"""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_participants = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            phq8 = batch['phq8']
            
            # Filter out missing labels
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
        return None
    
    # Convert to numpy arrays
    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    
    # Calculate metrics
    mse = mean_squared_error(labels_arr, preds_arr)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels_arr, preds_arr)
    r2 = r2_score(labels_arr, preds_arr)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'participant': all_participants,
        'true_phq8': labels_arr,
        'pred_phq8': preds_arr,
        'error': np.abs(labels_arr - preds_arr)
    })
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'mse': mse,
        'r2': r2,
        'results_df': results_df
    }
    
    return metrics


def train_model(model, train_loader, val_loader, device, epochs=30, save_model=True):
    """Training loop with early stopping and learning rate scheduling"""
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    criterion = nn.MSELoss()
    
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_model_state = None  # Store in memory instead
    
    history = {
        'train_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'val_r2': [],
        'lr': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        n_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio = batch['audio'].to(device)
            video = batch['video'].to(device)
            phq8 = batch['phq8'].to(device)
            
            # Filter missing labels
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
            
            # Calculate loss
            loss = criterion(preds, phq8[mask])
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            
            optimizer.step()
            
            train_loss += loss.item() * mask.sum().item()
            n_samples += mask.sum().item()
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / n_samples if n_samples > 0 else float('nan')
        
        # Validation phase
        val_metrics = evaluate(model, val_loader, device)
        
        if val_metrics is None:
            print("WARNING: No valid samples in validation set")
            continue
        
        # Update learning rate
        scheduler.step(val_metrics['rmse'])
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_r2'].append(val_metrics['r2'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val RMSE: {val_metrics['rmse']:.4f}")
        print(f"  Val MAE: {val_metrics['mae']:.4f}")
        print(f"  Val R²: {val_metrics['r2']:.4f}")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Store model state in memory
            import copy
            best_model_state = copy.deepcopy(model.state_dict())
            
            # Try to save model to disk (with error handling)
            if save_model:
                try:
                    # Ensure directory exists and has write permissions
                    save_dir = DATA_ROOT
                    os.makedirs(save_dir, exist_ok=True)
                    
                    # Save with temporary file first (atomic write)
                    temp_path = os.path.join(save_dir, "best_model_temp.pt")
                    final_path = os.path.join(save_dir, "best_model.pt")
                    
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_rmse': best_val_rmse,
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
                    os.path.join(DATA_ROOT, "best_val_predictions.csv"),
                    index=False
                )
                print(f"  ✓ Predictions saved")
            except Exception as e:
                print(f"  ⚠ Warning: Could not save predictions: {e}")
            
            print(f"  ✓ New best model! (RMSE: {best_val_rmse:.4f})")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
        
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")
            break
    
    # Load best model from memory if available
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("\n✓ Best model loaded from memory")
    
    return history, best_val_rmse


def main():
    print("="*60)
    print("LoRA-BERT Baseline Model for PHQ-8 Prediction")
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
            save_model_flag = True
    except:
        save_model_flag = True
    
    # Load split files
    print("\n1. Loading dataset splits...")
    train_df = read_split_csv(TRAIN_SPLIT)
    dev_df = read_split_csv(DEV_SPLIT)
    
    print(f"   Train samples: {len(train_df)}")
    print(f"   Dev samples: {len(dev_df)}")
    
    # Build dataset indices
    print("\n2. Building dataset indices...")
    train_rows = build_dataset_index(train_df, DATA_ROOT)
    dev_rows = build_dataset_index(dev_df, DATA_ROOT)
    
    if len(train_rows) == 0 or len(dev_rows) == 0:
        print("ERROR: No valid samples found!")
        return
    
    # Get feature dimensions
    audio_dim = train_rows[0]['covarep'].shape[0]
    video_dim = train_rows[0]['clnf_aus'].shape[0]
    
    print(f"\n3. Feature dimensions:")
    print(f"   Audio (COVAREP): {audio_dim}")
    print(f"   Video (CLNF AUs): {video_dim}")
    print(f"   Text: BERT embeddings (768)")
    
    # Create datasets
    print("\n4. Creating datasets...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    train_dataset = DAICWozPHQ8Dataset(train_rows, tokenizer, max_len=MAX_LEN)
    dev_dataset = DAICWozPHQ8Dataset(dev_rows, tokenizer, max_len=MAX_LEN)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Create model
    print("\n5. Initializing model...")
    model = MultiModalPHQ8LoRA(audio_dim=audio_dim, video_dim=video_dim)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Device: {DEVICE}")
    
    # Train model
    print("\n6. Starting training...")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LR}")
    print(f"   Early stopping patience: {PATIENCE}")
    print(f"   Model saving: {'Enabled' if save_model_flag else 'Disabled (low disk space)'}")
    print()
    
    history, best_rmse = train_model(
        model,
        train_loader,
        dev_loader,
        DEVICE,
        epochs=EPOCHS,
        save_model=save_model_flag
    )
    
    # Save training history
    print("\n7. Saving results...")
    try:
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(DATA_ROOT, "training_history.csv"), index=False)
        print("   ✓ Training history saved")
    except Exception as e:
        print(f"   ⚠ Could not save training history: {e}")
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"{'='*60}")
    print(f"Best Validation RMSE: {best_rmse:.4f}")
    print(f"\nNote: Best model is loaded in memory and ready for use")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()