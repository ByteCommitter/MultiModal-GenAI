"""
Improved Multimodal Generative Transformer Architecture
- Comprehensive validation metrics
- Removed test set evaluation
- Enhanced error handling and logging
- Focus on validation performance
"""

import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                            accuracy_score, f1_score, precision_score, recall_score,
                            confusion_matrix, classification_report)
import glob
import json
import warnings
warnings.filterwarnings('ignore')

# ============ CONFIGURATION ============
DATA_ROOT = "/home/dipanjan/rugraj/DIAC-WOZ/"
TRAIN_SPLIT = os.path.join(DATA_ROOT, "train_split_Depression_AVEC2017.csv")
DEV_SPLIT = os.path.join(DATA_ROOT, "dev_split_Depression_AVEC2017.csv")

MODEL_CONFIG = {
    "text_model": "distilbert-base-uncased",
    "hidden_dim": 512,
    "num_layers": 4,
    "num_heads": 8,
    "dropout": 0.15,
    "fusion_type": "crossmodal",
    "max_audio_frames": 100,
    "max_video_frames": 100,
    "decoder_layers": 3,
}

TRAINING_CONFIG = {
    "batch_size": 6,
    "learning_rate": 3e-5,
    "epochs": 40,
    "max_text_length": 256,
    "weight_decay": 0.01,
    "grad_clip_norm": 1.0,
    "num_workers": 4,
    "pin_memory": True,
    "early_stopping_patience": 7,
}

MULTITASK_CONFIG = {
    "main_task_weight": 1.0,
    "aux_task_weight": 0.3,
    "recon_weight": 0.2,
    "use_auxiliary_tasks": True,
    "use_reconstruction": True,
}

OUTPUT_CONFIG = {
    "output_dir": os.path.join(DATA_ROOT, "trained_models"),
    "model_name": "generative_multimodal",
    "save_best_model": True,
    "save_final_model": True,
    "verbose_logging": True,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")

# ============ PATH CHECKING ============
def check_paths():
    """Verify all required paths exist"""
    paths = [DATA_ROOT, TRAIN_SPLIT, DEV_SPLIT]
    for path in paths:
        if not os.path.exists(path):
            print(f"❌ Error: Path does not exist: {path}")
            return False
    print("✅ All required paths verified")
    return True

# ============ DATA LOADING ============
def read_transcript_with_turns(session_folder, participant_only=True):
    """Read transcript with robust error handling"""
    file_candidate = glob.glob(os.path.join(session_folder, "*_TRANSCRIPT.csv"))
    if not file_candidate:
        return "", []
    fp = file_candidate[0]
    
    try:
        dfh = pd.read_csv(fp, sep='\t', dtype=str, quoting=3)
        if 'speaker' in dfh.columns and 'value' in dfh.columns:
            if participant_only:
                mask = dfh['speaker'].str.lower() != 'ellie'
                turns = dfh.loc[mask, 'value'].fillna('').tolist()
            else:
                turns = dfh['value'].fillna('').tolist()
            full_text = " ".join(turns)
            return full_text, turns[:10]
    except Exception:
        pass
    
    try:
        df = pd.read_csv(fp, sep='\t', header=None, quoting=3, dtype=str, engine='python')
        if df.shape[1] >= 3:
            text_col = df.iloc[:, -1].fillna('').astype(str)
            full_text = " ".join(text_col.tolist())
            return full_text, text_col.tolist()[:10]
    except Exception:
        pass
    
    return "", []


def load_temporal_covarep_features(session_folder, max_frames=100):
    """Load temporal COVAREP features"""
    fp = glob.glob(os.path.join(session_folder, "*_COVAREP.csv"))
    if not fp:
        return None
    
    try:
        df = pd.read_csv(fp[0], index_col=False)
        col_lower = [c.lower() for c in df.columns]
        
        # Handle VUV
        vuv_idx = None
        for i, c in enumerate(col_lower):
            if c == 'vuv' or 'vuv' in c:
                vuv_idx = df.columns[i]
                break
        
        if vuv_idx is not None:
            mask = df[vuv_idx] == 0
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != vuv_idx]
            df.loc[mask, numeric_cols] = 0.0
        
        numeric = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
        if numeric.size == 0:
            return None
        
        # Sample/pad to max_frames
        if len(numeric) > max_frames:
            indices = np.linspace(0, len(numeric)-1, max_frames, dtype=int)
            numeric = numeric[indices]
        elif len(numeric) < max_frames:
            padding = np.zeros((max_frames - len(numeric), numeric.shape[1]), dtype=np.float32)
            numeric = np.vstack([numeric, padding])
        
        numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
        return numeric
    except Exception as e:
        print(f"Error loading COVAREP: {e}")
        return None


def load_temporal_clnf_aus(session_folder, max_frames=100):
    """Load temporal CLNF features"""
    fp = glob.glob(os.path.join(session_folder, "*_CLNF_AUs.csv"))
    if not fp:
        return None
    
    try:
        df = pd.read_csv(fp[0], index_col=False)
        cols = [c for c in df.columns if str(c).endswith('_r')]
        
        if not cols:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not cols:
            return None
        
        numeric = df[cols].to_numpy(dtype=np.float32)
        if numeric.size == 0:
            return None
        
        # Sample/pad to max_frames
        if len(numeric) > max_frames:
            indices = np.linspace(0, len(numeric)-1, max_frames, dtype=int)
            numeric = numeric[indices]
        elif len(numeric) < max_frames:
            padding = np.zeros((max_frames - len(numeric), numeric.shape[1]), dtype=np.float32)
            numeric = np.vstack([numeric, padding])
        
        numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)
        return numeric
    except Exception as e:
        print(f"Error loading CLNF: {e}")
        return None


def read_split_csv(split_csv):
    """Read split CSV"""
    df = pd.read_csv(split_csv)
    
    print(f"\n{'='*60}")
    print(f"Reading: {os.path.basename(split_csv)}")
    print(f"{'='*60}")
    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
    possible_id_cols = [c for c in df.columns if any(x in c.lower() for x in ['participant','id','file'])]
    pid_col = possible_id_cols[0] if possible_id_cols else df.columns[0]
    
    phq_cols = [c for c in df.columns if 'phq' in c.lower()]
    phq_col = phq_cols[0] if phq_cols else None
    
    out = pd.DataFrame()
    out['participant'] = df[pid_col].astype(str).str.strip()
    
    if phq_col is not None:
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
    """Build dataset index with temporal features"""
    max_audio_frames = MODEL_CONFIG["max_audio_frames"]
    max_video_frames = MODEL_CONFIG["max_video_frames"]
    
    rows = []
    emotion_counts = {0: 0, 1: 0}
    skipped = {'no_folder': 0, 'no_features': 0, 'no_phq8': 0, 'no_text': 0}
    
    for _, r in tqdm(split_df.iterrows(), total=len(split_df), desc="Building index"):
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
        transcript, turns = read_transcript_with_turns(sfolder, participant_only=True)
        if not transcript or transcript.strip() == "":
            skipped['no_text'] += 1
            continue
        
        audio_temporal = load_temporal_covarep_features(sfolder, max_audio_frames)
        video_temporal = load_temporal_clnf_aus(sfolder, max_video_frames)
        
        if audio_temporal is None and video_temporal is None:
            skipped['no_features'] += 1
            continue
        
        # Use zero arrays if modality missing
        if audio_temporal is None:
            audio_temporal = np.zeros((max_audio_frames, 74), dtype=np.float32)
        if video_temporal is None:
            video_temporal = np.zeros((max_video_frames, 35), dtype=np.float32)
        
        # Handle PHQ8 score
        phq8 = r['phq8'] if not pd.isna(r['phq8']) else None
        if phq8 is None:
            skipped['no_phq8'] += 1
            continue
        
        # Create emotion category (threshold at 10 for depression)
        emotion_category = 1 if phq8 >= 10 else 0
        emotion_counts[emotion_category] += 1
        
        rows.append({
            'participant': pid,
            'session_folder': sfolder,
            'transcript': transcript,
            'turns': turns,
            'audio_temporal': audio_temporal.astype(np.float32),
            'video_temporal': video_temporal.astype(np.float32),
            'phq8': phq8,
            'emotion_category': emotion_category,
        })
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Dataset Building Summary")
    print(f"{'='*60}")
    print(f"Valid samples created: {len(rows)}")
    print(f"\nEmotion category distribution:")
    total = sum(emotion_counts.values())
    if total > 0:
        print(f"  Non-depressed (PHQ8<10): {emotion_counts[0]} ({100*emotion_counts[0]/total:.1f}%)")
        print(f"  Depressed (PHQ8≥10):     {emotion_counts[1]} ({100*emotion_counts[1]/total:.1f}%)")
    
    print(f"\nSkipped samples:")
    print(f"  - No folder: {skipped['no_folder']}")
    print(f"  - No features: {skipped['no_features']}")
    print(f"  - No PHQ8: {skipped['no_phq8']}")
    print(f"  - No text: {skipped['no_text']}")
    print(f"{'='*60}\n")
    
    return rows


# ============ DATASET CLASS ============
class GenerativeDAICWozDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = TRAINING_CONFIG["max_text_length"]
    
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
        item['audio_temporal'] = torch.tensor(r['audio_temporal'], dtype=torch.float32)
        item['video_temporal'] = torch.tensor(r['video_temporal'], dtype=torch.float32)
        item['phq8'] = torch.tensor(r['phq8'], dtype=torch.float32)
        item['emotion_category'] = torch.tensor(r['emotion_category'], dtype=torch.long)
        item['participant'] = r['participant']
        
        return item


# ============ MODEL COMPONENTS ============
class ModalityTransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=4, nhead=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x, src_key_padding_mask=None):
        x = self.input_proj(x)
        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return out


class CrossModalFusion(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, _ = self.cross_attn(query, key, value, key_padding_mask=key_padding_mask)
        return self.norm(query + attn_output)


class GenerativeMultimodalModel(nn.Module):
    def __init__(self, text_hidden_dim, audio_feature_dim, video_feature_dim, hidden_dim):
        super().__init__()
        
        # Text Encoder (pretrained)
        self.text_encoder = AutoModel.from_pretrained(MODEL_CONFIG["text_model"])
        self.text_proj = nn.Linear(text_hidden_dim, hidden_dim)
        
        # Audio/Video Transformer Encoders
        self.audio_encoder = ModalityTransformerEncoder(
            audio_feature_dim, hidden_dim,
            num_layers=MODEL_CONFIG["num_layers"],
            nhead=MODEL_CONFIG["num_heads"],
            dropout=MODEL_CONFIG["dropout"]
        )
        
        self.video_encoder = ModalityTransformerEncoder(
            video_feature_dim, hidden_dim,
            num_layers=MODEL_CONFIG["num_layers"],
            nhead=MODEL_CONFIG["num_heads"],
            dropout=MODEL_CONFIG["dropout"]
        )
        
        # Cross-modal Fusion
        self.fusion_text_audio = CrossModalFusion(hidden_dim, num_heads=MODEL_CONFIG["num_heads"])
        self.fusion_fused_video = CrossModalFusion(hidden_dim, num_heads=MODEL_CONFIG["num_heads"])
        
        # Generative Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=MODEL_CONFIG["num_heads"],
            dropout=MODEL_CONFIG["dropout"],
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=MODEL_CONFIG["decoder_layers"])
        
        # Regression Head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG["dropout"]),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG["dropout"]),
            nn.Linear(hidden_dim // 2, 2)
        )
        
        # Reconstruction projection
        self.recon_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, input_ids, attention_mask, audio_feats, video_feats):
        # Text Encoding
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_embed = self.text_proj(text_outputs.last_hidden_state)
        
        # Audio/Video Encoding
        audio_embed = self.audio_encoder(audio_feats)
        video_embed = self.video_encoder(video_feats)
        
        # Cross-Modal Fusion
        fused_text_audio = self.fusion_text_audio(text_embed, audio_embed, audio_embed)
        fused_all = self.fusion_fused_video(fused_text_audio, video_embed, video_embed)
        
        # CLS token
        cls_token = fused_all[:, 0, :]
        
        # Generative decoding
        memory = fused_all
        tgt = torch.zeros_like(memory)
        decoded = self.decoder(tgt, memory)
        
        # Predictions
        depression_pred = self.regressor(cls_token).squeeze(-1)
        emotion_pred = self.classifier(cls_token)
        
        # Reconstruction
        recon_out = self.recon_proj(decoded)
        
        return depression_pred, emotion_pred, recon_out, memory


# ============ LOSS FUNCTIONS ============
def reconstruction_loss(recon_out, memory):
    """L2 loss for reconstruction"""
    return F.mse_loss(recon_out, memory)


# ============ EVALUATION FUNCTION ============
def evaluate_model(model, loader, device):
    """Comprehensive evaluation with all metrics"""
    model.eval()
    
    # Regression metrics
    phq8_preds = []
    phq8_true = []
    
    # Classification metrics
    emotion_preds = []
    emotion_true = []
    
    # Per-sample results
    participants = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_temporal = batch['audio_temporal'].to(device)
            video_temporal = batch['video_temporal'].to(device)
            phq8 = batch['phq8']
            emotion_category = batch['emotion_category']
            
            # Forward pass
            depression_pred, emotion_pred, _, _ = model(
                input_ids, attention_mask, audio_temporal, video_temporal
            )
            
            # Collect predictions
            phq8_preds.extend(depression_pred.cpu().numpy())
            phq8_true.extend(phq8.numpy())
            
            emotion_preds.extend(emotion_pred.argmax(dim=1).cpu().numpy())
            emotion_true.extend(emotion_category.numpy())
            
            participants.extend(batch['participant'])
    
    # Convert to numpy
    phq8_preds = np.array(phq8_preds)
    phq8_true = np.array(phq8_true)
    emotion_preds = np.array(emotion_preds)
    emotion_true = np.array(emotion_true)
    
    # Regression metrics
    mse = mean_squared_error(phq8_true, phq8_preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(phq8_true, phq8_preds)
    r2 = r2_score(phq8_true, phq8_preds)
    
    # Classification metrics
    accuracy = accuracy_score(emotion_true, emotion_preds)
    f1 = f1_score(emotion_true, emotion_preds, average='weighted', zero_division=0)
    precision = precision_score(emotion_true, emotion_preds, average='weighted', zero_division=0)
    recall = recall_score(emotion_true, emotion_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(emotion_true, emotion_preds)
    
    # Per-sample results
    results_df = pd.DataFrame({
        'participant': participants,
        'true_phq8': phq8_true,
        'pred_phq8': phq8_preds,
        'error': np.abs(phq8_true - phq8_preds),
        'true_emotion': emotion_true,
        'pred_emotion': emotion_preds
    })
    
    metrics = {
        'num_samples': len(phq8_preds),
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


# ============ TRAINING FUNCTION ============
def train_generative_model(model, train_loader, val_loader, device):
    """Training loop with comprehensive metrics"""
    model.to(device)
    
    epochs = TRAINING_CONFIG["epochs"]
    lr = TRAINING_CONFIG["learning_rate"]
    weight_decay = TRAINING_CONFIG["weight_decay"]
    grad_clip = TRAINING_CONFIG["grad_clip_norm"]
    patience = TRAINING_CONFIG["early_stopping_patience"]
    
    # Loss functions
    mse_loss = nn.MSELoss()
    
    # Calculate class weights
    all_labels = []
    for batch in train_loader:
        all_labels.extend(batch['emotion_category'].numpy())
    
    label_counts = np.bincount(all_labels, minlength=2)
    label_counts = np.maximum(label_counts, 1)
    weights = 1.0 / label_counts
    weights = weights / weights.sum()
    class_weights = torch.FloatTensor(weights).to(device)
    
    print(f"\nClass distribution in training:")
    print(f"  Class 0 (PHQ8<10): {label_counts[0]}")
    print(f"  Class 1 (PHQ8≥10): {label_counts[1]}")
    print(f"  Class weights: {class_weights}")
    
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_regression_loss': [],
        'train_classification_loss': [],
        'train_reconstruction_loss': [],
        'val_rmse': [],
        'val_mae': [],
        'val_r2': [],
        'val_accuracy': [],
        'val_f1': [],
        'lr': []
    }
    
    best_val_rmse = float('inf')
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(epochs):
        # ========== TRAINING PHASE ==========
        model.train()
        total_loss = 0.0
        total_reg_loss = 0.0
        total_cls_loss = 0.0
        total_recon_loss = 0.0
        n_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_pbar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_temporal = batch['audio_temporal'].to(device)
            video_temporal = batch['video_temporal'].to(device)
            phq8 = batch['phq8'].to(device)
            emotion_category = batch['emotion_category'].to(device)
            
            # Forward pass
            depression_pred, emotion_pred, recon_out, memory = model(
                input_ids, attention_mask, audio_temporal, video_temporal
            )
            
            # Regression loss
            reg_loss = mse_loss(depression_pred, phq8)
            batch_loss = reg_loss * MULTITASK_CONFIG["main_task_weight"]
            
            # Classification loss
            if MULTITASK_CONFIG["use_auxiliary_tasks"]:
                cls_loss = ce_loss(emotion_pred, emotion_category)
                batch_loss += cls_loss * MULTITASK_CONFIG["aux_task_weight"]
                total_cls_loss += cls_loss.item() * len(phq8)
            
            # Reconstruction loss
            if MULTITASK_CONFIG["use_reconstruction"]:
                recon_loss_val = reconstruction_loss(recon_out, memory)
                batch_loss += recon_loss_val * MULTITASK_CONFIG["recon_weight"]
                total_recon_loss += recon_loss_val.item() * len(phq8)
            
            # Backward pass
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            # Track losses
            total_loss += batch_loss.item() * len(phq8)
            total_reg_loss += reg_loss.item() * len(phq8)
            n_samples += len(phq8)
            
            train_pbar.set_postfix({'Loss': f"{batch_loss.item():.4f}"})
        
        # Average training losses
        avg_train_loss = total_loss / n_samples
        avg_train_reg_loss = total_reg_loss / n_samples
        avg_train_cls_loss = total_cls_loss / n_samples if n_samples > 0 else 0
        avg_train_recon_loss = total_recon_loss / n_samples if n_samples > 0 else 0
        
        # ========== VALIDATION PHASE ==========
        val_metrics = evaluate_model(model, val_loader, device)
        
        # Update scheduler
        scheduler.step(val_metrics['rmse'])
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_regression_loss'].append(avg_train_reg_loss)
        history['train_classification_loss'].append(avg_train_cls_loss)
        history['train_reconstruction_loss'].append(avg_train_recon_loss)
        history['val_rmse'].append(val_metrics['rmse'])
        history['val_mae'].append(val_metrics['mae'])
        history['val_r2'].append(val_metrics['r2'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print metrics
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*60}")
        print(f"Training Losses:")
        print(f"  Total:          {avg_train_loss:.4f}")
        print(f"  Regression:     {avg_train_reg_loss:.4f}")
        print(f"  Classification: {avg_train_cls_loss:.4f}")
        print(f"  Reconstruction: {avg_train_recon_loss:.4f}")
        print(f"\nValidation Regression Metrics:")
        print(f"  RMSE: {val_metrics['rmse']:.4f}")
        print(f"  MAE:  {val_metrics['mae']:.4f}")
        print(f"  R²:   {val_metrics['r2']:.4f}")
        print(f"\nValidation Classification Metrics:")
        print(f"  Accuracy:  {val_metrics['accuracy']:.4f}")
        print(f"  F1-Score:  {val_metrics['f1']:.4f}")
        print(f"  Precision: {val_metrics['precision']:.4f}")
        print(f"  Recall:    {val_metrics['recall']:.4f}")
        print(f"\nConfusion Matrix:")
        print(val_metrics['confusion_matrix'])
        print(f"\nLearning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # ========== EARLY STOPPING & MODEL SAVING ==========
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best model
            if OUTPUT_CONFIG["save_best_model"]:
                out_dir = OUTPUT_CONFIG["output_dir"]
                os.makedirs(out_dir, exist_ok=True)
                
                model_path = os.path.join(out_dir, f"{OUTPUT_CONFIG['model_name']}_best.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_metrics': {k: v for k, v in val_metrics.items() if k != 'results_df'}
                }, model_path)
                
                # Save predictions
                val_metrics['results_df'].to_csv(
                    os.path.join(out_dir, f"{OUTPUT_CONFIG['model_name']}_best_predictions.csv"),
                    index=False
                )
                
                print(f"\n✓ New best model saved! (RMSE: {best_val_rmse:.4f})")
        else:
            patience_counter += 1
            print(f"\nNo improvement ({patience_counter}/{patience})")
        
        if patience_counter >= patience:
            print(f"\n{'='*60}")
            print(f"Early stopping at epoch {epoch+1}")
            print(f"Best RMSE: {best_val_rmse:.4f} at epoch {best_epoch}")
            print(f"{'='*60}")
            break
    
    return history, best_val_rmse


# ============ MAIN EXECUTION ============
def main():
    print("\n" + "="*60)
    print("🚀 Multimodal Generative Transformer (Improved)")
    print("="*60)
    
    # Check paths
    if not check_paths():
        print("❌ Exiting due to missing paths")
        sys.exit(1)
    
    # Load data
    print("\n1. Loading dataset splits...")
    train_df = read_split_csv(TRAIN_SPLIT)
    dev_df = read_split_csv(DEV_SPLIT)
    
    print(f"\n2. Building dataset indices...")
    train_rows = build_dataset_index(train_df)
    dev_rows = build_dataset_index(dev_df)
    
    if not train_rows or not dev_rows:
        print("❌ No valid data found!")
        return
    
    print(f"\nValid samples - Train: {len(train_rows)}, Dev: {len(dev_rows)}")
    
    # Get feature dimensions
    audio_dim = train_rows[0]['audio_temporal'].shape[1]
    video_dim = train_rows[0]['video_temporal'].shape[1]
    
    print(f"\n3. Feature dimensions:")
    print(f"   Audio (temporal): {audio_dim} features × {MODEL_CONFIG['max_audio_frames']} frames")
    print(f"   Video (temporal): {video_dim} features × {MODEL_CONFIG['max_video_frames']} frames")
    
    # Create datasets
    print("\n4. Creating datasets...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["text_model"])
    
    train_dataset = GenerativeDAICWozDataset(train_rows, tokenizer)
    dev_dataset = GenerativeDAICWozDataset(dev_rows, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=True,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=TRAINING_CONFIG["pin_memory"],
        drop_last=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=TRAINING_CONFIG["pin_memory"]
    )
    
    # Initialize model
    print("\n5. Initializing model...")
    temp_model = AutoModel.from_pretrained(MODEL_CONFIG["text_model"])
    text_hidden_dim = temp_model.config.hidden_size
    del temp_model
    
    model = GenerativeMultimodalModel(
        text_hidden_dim=text_hidden_dim,
        audio_feature_dim=audio_dim,
        video_feature_dim=video_dim,
        hidden_dim=MODEL_CONFIG["hidden_dim"]
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Device: {DEVICE}")
    
    # Print configuration
    print(f"\n6. Training configuration:")
    print(f"   Epochs: {TRAINING_CONFIG['epochs']}")
    print(f"   Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"   Learning rate: {TRAINING_CONFIG['learning_rate']}")
    print(f"   Early stopping patience: {TRAINING_CONFIG['early_stopping_patience']}")
    print(f"   Main task weight: {MULTITASK_CONFIG['main_task_weight']}")
    print(f"   Auxiliary task weight: {MULTITASK_CONFIG['aux_task_weight']}")
    print(f"   Reconstruction weight: {MULTITASK_CONFIG['recon_weight']}")
    
    # Train model
    print("\n7. Starting training...")
    print("="*60)
    
    history, best_rmse = train_generative_model(model, train_loader, dev_loader, DEVICE)
    
    # Save results
    print("\n8. Saving final results...")
    out_dir = OUTPUT_CONFIG["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(out_dir, f"{OUTPUT_CONFIG['model_name']}_best.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_path = os.path.join(out_dir, f"{OUTPUT_CONFIG['model_name']}_history.csv")
    history_df.to_csv(history_path, index=False)
    print(f"   Training history saved: {history_path}")
    
    # Final validation evaluation
    print("\n9. Final validation evaluation...")
    final_metrics = evaluate_model(model, dev_loader, DEVICE)
    
    # Save final metrics
    metrics_to_save = {k: v for k, v in final_metrics.items() if k != 'results_df'}
    metrics_path = os.path.join(out_dir, f"{OUTPUT_CONFIG['model_name']}_final_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"   Final metrics saved: {metrics_path}")
    
    # Save final model if configured
    if OUTPUT_CONFIG["save_final_model"]:
        final_model_path = os.path.join(out_dir, f"{OUTPUT_CONFIG['model_name']}_final.pt")
        torch.save(model.state_dict(), final_model_path)
        print(f"   Final model saved: {final_model_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("🎉 TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\nBest Validation Metrics (Epoch {checkpoint['epoch']+1}):")
    print(f"\nRegression:")
    print(f"  RMSE: {final_metrics['rmse']:.4f}")
    print(f"  MAE:  {final_metrics['mae']:.4f}")
    print(f"  R²:   {final_metrics['r2']:.4f}")
    print(f"\nClassification (Depression Detection):")
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  F1-Score:  {final_metrics['f1']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"\nConfusion Matrix:")
    cm = np.array(final_metrics['confusion_matrix'])
    print(f"  [[TN={cm[0,0]:3d}, FP={cm[0,1]:3d}]")
    print(f"   [FN={cm[1,0]:3d}, TP={cm[1,1]:3d}]]")
    print(f"\nSample Distribution:")
    print(f"  Total samples: {final_metrics['num_samples']}")
    print(f"  True negatives: {cm[0,0]} ({100*cm[0,0]/final_metrics['num_samples']:.1f}%)")
    print(f"  True positives: {cm[1,1]} ({100*cm[1,1]/final_metrics['num_samples']:.1f}%)")
    
    print(f"\n{'='*60}")
    print("Files saved:")
    print(f"  - {OUTPUT_CONFIG['model_name']}_best.pt")
    print(f"  - {OUTPUT_CONFIG['model_name']}_best_predictions.csv")
    print(f"  - {OUTPUT_CONFIG['model_name']}_history.csv")
    print(f"  - {OUTPUT_CONFIG['model_name']}_final_metrics.json")
    if OUTPUT_CONFIG["save_final_model"]:
        print(f"  - {OUTPUT_CONFIG['model_name']}_final.pt")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()