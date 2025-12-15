"""
Advanced Multi-Modal Emotion Detection - Simplified Version
Self-contained with built-in configuration
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
from sklearn.metrics import mean_squared_error, accuracy_score
import glob
import random
import json
from typing import Dict, List, Tuple, Optional
from utils.metrics_utils import compute_regression_metrics, compute_classification_metrics, print_metrics_summary

# ============ BUILT-IN CONFIGURATION ============
DATA_ROOT = "/home/dipanjan/rugraj/DIAC-WOZ/"
TRAIN_SPLIT = os.path.join(DATA_ROOT, "train_split_Depression_AVEC2017.csv")
DEV_SPLIT = os.path.join(DATA_ROOT, "dev_split_Depression_AVEC2017.csv")
TEST_SPLIT = os.path.join(DATA_ROOT, "test_split_Depression_AVEC2017.csv")

MODEL_CONFIG = {
    "text_model": "distilbert-base-uncased",
    "hidden_dim": 512,
    "num_heads": 8,
    "num_layers": 6,
    "dropout": 0.15,
    "fusion_type": "dynamic",
    "max_audio_frames": 100,
    "max_video_frames": 100,
    "tcn_layers": 3,
}

TRAINING_CONFIG = {
    "batch_size": 6,
    "learning_rate": 3e-5,
    "epochs": 60,
    "max_text_length": 256,
    "weight_decay": 0.01,
    "grad_clip_norm": 1.0,
    "warmup_ratio": 0.1,
    "lr_scheduler": "onecycle",
    "num_workers": 4,
    "pin_memory": True,
}

MULTITASK_CONFIG = {
    "main_task_weight": 1.0,
    "aux_task_weight": 0.2,
    "contrastive_weight": 0.3,
    "use_contrastive": True,
    "use_auxiliary_tasks": True,
    "contrastive_temperature": 0.07,
    "projection_dim": 128,
}

AUGMENTATION_CONFIG = {
    "enable_training_augmentation": True,
    "audio_noise_prob": 0.3,
    "audio_noise_std": 0.01,
    "video_dropout_prob": 0.3,
    "video_dropout_rate": 0.1,
    "text_augmentation": False,
}

OUTPUT_CONFIG = {
    "output_dir": os.path.join(DATA_ROOT, "trained_models"),
    "model_name": "advanced_multimodal",
    "save_predictions": True,
    "save_embeddings": False,
    "verbose_logging": True,
    "save_best_model": True,
    "save_final_model": True,
}

def validate_config():
    """Simple validation"""
    return []  # No errors for built-in config

def get_config_summary():
    """Get configuration summary"""
    return "Built-in configuration loaded successfully!"

# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {DEVICE}")

# ============ DATA LOADING (Same as advanced_multimodal.py) ============
def read_transcript_with_turns(session_folder, participant_only=True):
    """Enhanced transcript reading that preserves turn structure"""
    file_candidate = glob.glob(os.path.join(session_folder, "*_TRANSCRIPT.csv"))
    if not file_candidate:
        return "", []

    fp = file_candidate[0]
    try:
        dfh = pd.read_csv(fp, sep='\t', dtype=str, quoting=3)
        if 'speaker' in dfh.columns and 'utterance' in dfh.columns:
            if participant_only:
                part_texts = dfh[dfh['speaker'].str.lower() != 'ellie']['utterance'].fillna('')
                turns = part_texts.tolist()
            else:
                turns = dfh['utterance'].fillna('').tolist()
            full_text = " ".join(turns)
            return full_text, turns[:10]  # Limit to first 10 turns for memory
    except Exception:
        pass

    # Fallback to simple reading
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
    """Load temporal audio features instead of just mean pooling"""
    fp = glob.glob(os.path.join(session_folder, "*_COVAREP.csv"))
    if not fp:
        return None

    df = pd.read_csv(fp[0], index_col=False)

    # Handle VUV (voiced/unvoiced) column
    col_lower = [c.lower() for c in df.columns]
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

    # Sample frames uniformly if too many
    if len(numeric) > max_frames:
        indices = np.linspace(0, len(numeric)-1, max_frames, dtype=int)
        numeric = numeric[indices]

    # Pad if too few frames
    if len(numeric) < max_frames:
        padding = np.zeros((max_frames - len(numeric), numeric.shape[1]), dtype=np.float32)
        numeric = np.vstack([numeric, padding])

    # Clean data
    numeric = np.nan_to_num(numeric, nan=0.0, posinf=1e6, neginf=-1e6)
    return numeric


def load_temporal_clnf_aus(session_folder, max_frames=100):
    """Load temporal facial action units"""
    fp = glob.glob(os.path.join(session_folder, "*_CLNF_AUs.csv"))
    if not fp:
        return None

    df = pd.read_csv(fp[0], index_col=False)

    # Prefer intensity (_r) columns over presence (_c) columns
    cols = [c for c in df.columns if str(c).endswith('_r')]
    if not cols:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not cols:
        return None

    numeric = df[cols].to_numpy(dtype=np.float32)
    if numeric.size == 0:
        return None

    # Sample frames uniformly if too many
    if len(numeric) > max_frames:
        indices = np.linspace(0, len(numeric)-1, max_frames, dtype=int)
        numeric = numeric[indices]

    # Pad if too few frames
    if len(numeric) < max_frames:
        padding = np.zeros((max_frames - len(numeric), numeric.shape[1]), dtype=np.float32)
        numeric = np.vstack([numeric, padding])

    # Clean data
    numeric = np.nan_to_num(numeric, nan=0.0, posinf=1e6, neginf=-1e6)
    return numeric


def read_split_csv(split_csv):
    """Same as original but with enhanced error handling"""
    df = pd.read_csv(split_csv)
    possible_id_cols = [c for c in df.columns if any(x in c.lower() for x in ['participant','id','file'])]
    pid_col = possible_id_cols[0] if possible_id_cols else df.columns[0]
    phq_cols = [c for c in df.columns if 'phq' in c.lower() and any(x in c.lower() for x in ['score','8'] )]
    phq_col = phq_cols[0] if phq_cols else None

    out = pd.DataFrame()
    out['participant'] = df[pid_col].astype(str)
    if phq_col is not None:
        out['phq8'] = pd.to_numeric(df[phq_col], errors='coerce')
    else:
        out['phq8'] = np.nan
    out['session_folder'] = out['participant'].apply(lambda x: f"{x}_P" if not str(x).endswith('_P') else x)
    return out


def build_enhanced_dataset_index(split_df):
    max_audio_frames = MODEL_CONFIG["max_audio_frames"]
    max_video_frames = MODEL_CONFIG["max_video_frames"]
    
    rows = []
    stats = {
        'total': 0,
        'valid_phq8': 0,
        'valid_audio': 0,
        'valid_video': 0,
        'emotion_dist': {0: 0, 1: 0}
    }
    
    for _, r in split_df.iterrows():
        stats['total'] += 1
        sfolder = os.path.join(DATA_ROOT, str(r['session_folder']))
        
        # Handle folder paths
        if not os.path.isdir(sfolder):
            candidates = glob.glob(os.path.join(DATA_ROOT, f"{r['participant']}*_P")) + \
                        glob.glob(os.path.join(DATA_ROOT, f"*{r['participant']}*_P"))
            if candidates:
                sfolder = candidates[0]
            else:
                continue
        
        # Get PHQ8 score
        phq8 = r['phq8'] if not pd.isna(r['phq8']) else None
        if phq8 is not None:
            stats['valid_phq8'] += 1
        
        # Load features
        transcript, turns = read_transcript_with_turns(sfolder, participant_only=True)
        audio_temporal = load_temporal_covarep_features(sfolder, max_audio_frames)
        video_temporal = load_temporal_clnf_aus(sfolder, max_video_frames)
        
        if audio_temporal is not None:
            stats['valid_audio'] += 1
        if video_temporal is not None:
            stats['valid_video'] += 1
        
        # Skip if both modalities are missing
        if audio_temporal is None and video_temporal is None:
            continue
            
        # Initialize with zeros if missing
        if audio_temporal is None:
            audio_temporal = np.zeros((max_audio_frames, 10), dtype=np.float32)
        if video_temporal is None:
            video_temporal = np.zeros((max_video_frames, 10), dtype=np.float32)
        
        # Create emotion category
        emotion_category = 0 if phq8 is not None and phq8 < 8 else 1
        stats['emotion_dist'][emotion_category] += 1
        
        rows.append({
            'session_folder': sfolder,
            'transcript': transcript,
            'turns': turns,
            'audio_temporal': audio_temporal.astype(np.float32),
            'video_temporal': video_temporal.astype(np.float32),
            'phq8': phq8,
            'emotion_category': emotion_category,
            'participant': r['participant']
        })
    
    # Print detailed statistics
    print(f"\nDataset Statistics:")
    print(f"Total samples processed: {stats['total']}")
    print(f"Valid samples created: {len(rows)}")
    print(f"Samples with valid PHQ8: {stats['valid_phq8']}")
    print(f"Samples with audio: {stats['valid_audio']}")
    print(f"Samples with video: {stats['valid_video']}")
    print("\nEmotion Distribution:")
    total = sum(stats['emotion_dist'].values())
    if total > 0:
        for label, count in stats['emotion_dist'].items():
            print(f"Class {label}: {count} ({100*count/total:.1f}%)")
    
    return rows


# ============ DATASET CLASS ============
class ConfigurableDAICWozDataset(Dataset):
    def __init__(self, rows, tokenizer, is_training=False):
        self.rows = rows
        self.tokenizer = tokenizer
        self.is_training = is_training
        self.max_len = TRAINING_CONFIG["max_text_length"]
        self.max_audio_frames = MODEL_CONFIG["max_audio_frames"]
        self.max_video_frames = MODEL_CONFIG["max_video_frames"]

    def __len__(self):
        return len(self.rows)

    def _augment_audio(self, audio_features):
        """Simple audio augmentation through noise injection"""
        if (self.is_training and
            AUGMENTATION_CONFIG["enable_training_augmentation"] and
            random.random() < AUGMENTATION_CONFIG["audio_noise_prob"]):
            noise = np.random.normal(0, AUGMENTATION_CONFIG["audio_noise_std"],
                                   audio_features.shape).astype(np.float32)
            audio_features = audio_features + noise
        return audio_features

    def _augment_video(self, video_features):
        """Simple video augmentation through feature dropout"""
        if (self.is_training and
            AUGMENTATION_CONFIG["enable_training_augmentation"] and
            random.random() < AUGMENTATION_CONFIG["video_dropout_prob"]):
            mask = np.random.random(video_features.shape) > AUGMENTATION_CONFIG["video_dropout_rate"]
            video_features = video_features * mask
        return video_features

    def __getitem__(self, idx):
        r = self.rows[idx]

        # Process text
        encoding = self.tokenizer(
            r['transcript'] if r['transcript'] else " ",
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        item = {k: v.squeeze(0) for k, v in encoding.items()}

        # Process temporal features
        audio_temp = self._augment_audio(r['audio_temporal'].copy())
        video_temp = self._augment_video(r['video_temporal'].copy())

        item['audio_temporal'] = torch.tensor(audio_temp, dtype=torch.float)
        item['video_temporal'] = torch.tensor(video_temp, dtype=torch.float)
        item['phq8'] = torch.tensor(r['phq8'] if r['phq8'] is not None else np.nan, dtype=torch.float)
        item['emotion_category'] = torch.tensor(r['emotion_category'], dtype=torch.long)

        return item


# ============ MODEL COMPONENTS ============
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TemporalConvNet(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        num_layers = MODEL_CONFIG["tcn_layers"]
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            dilation = 2 ** i
            conv = nn.Conv1d(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                kernel_size=3,
                dilation=dilation,
                padding=dilation
            )
            norm = nn.BatchNorm1d(hidden_dim)
            self.convs.append(conv)
            self.norms.append(norm)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(MODEL_CONFIG["dropout"])

    def forward(self, x):
        # x: [batch, seq_len, features] -> [batch, features, seq_len]
        x = x.transpose(1, 2)

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x)
            x = norm(x)
            x = self.activation(x)
            x = self.dropout(x)

            # Residual connection if dimensions match
            if residual.size(1) == x.size(1):
                x = x + residual

        # [batch, features, seq_len] -> [batch, seq_len, features]
        return x.transpose(1, 2)


class CrossModalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        num_heads = MODEL_CONFIG["num_heads"]
        dropout = MODEL_CONFIG["dropout"]

        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        # Self-attention
        attn_out, _ = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask)
        query = self.norm1(query + self.dropout(attn_out))

        # Feed forward
        ffn_out = self.ffn(query)
        query = self.norm2(query + self.dropout(ffn_out))

        return query


class DynamicFusion(nn.Module):
    def __init__(self, input_dims, output_dim):
        super().__init__()
        self.modality_weights = nn.Parameter(torch.ones(len(input_dims)))
        self.fusion_layers = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        self.final_projection = nn.Linear(output_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(MODEL_CONFIG["dropout"])

    def forward(self, modality_features):
        # Normalize weights
        weights = F.softmax(self.modality_weights, dim=0)

        # Project each modality
        projected = []
        for i, (features, layer) in enumerate(zip(modality_features, self.fusion_layers)):
            proj = layer(features) * weights[i]
            projected.append(proj)

        # Combine
        fused = torch.stack(projected, dim=0).sum(dim=0)
        fused = self.final_projection(self.dropout(self.activation(fused)))

        return fused


# ============ MAIN MODEL ============
class ConfigurableMultiModalModel(nn.Module):
    def __init__(self, audio_dim, video_dim):
        super().__init__()
        hidden_dim = MODEL_CONFIG["hidden_dim"]
        num_layers = MODEL_CONFIG["num_layers"]
        text_model = MODEL_CONFIG["text_model"]

        # Text encoder
        self.text_encoder = AutoModel.from_pretrained(text_model)
        text_dim = self.text_encoder.config.hidden_size
        self.text_projection = nn.Linear(text_dim, hidden_dim)

        # Temporal encoders
        self.audio_tcn = TemporalConvNet(audio_dim, hidden_dim)
        self.video_tcn = TemporalConvNet(video_dim, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)

        # Cross-modal attention layers
        self.cross_modal_layers = nn.ModuleList([
            CrossModalAttention(hidden_dim) for _ in range(num_layers)
        ])

        # Fusion mechanism
        fusion_type = MODEL_CONFIG["fusion_type"]
        if fusion_type == "dynamic":
            self.fusion = DynamicFusion([hidden_dim, hidden_dim, hidden_dim], hidden_dim)
        else:
            self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)

        # Task-specific heads
        self.depression_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(MODEL_CONFIG["dropout"]),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Auxiliary task head (if enabled)
        if MULTITASK_CONFIG["use_auxiliary_tasks"]:
            self.emotion_classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(MODEL_CONFIG["dropout"]),
                nn.Linear(hidden_dim // 2, 2)  # Binary classification
            )

        # Contrastive learning projection (if enabled)
        if MULTITASK_CONFIG["use_contrastive"]:
            proj_dim = MULTITASK_CONFIG["projection_dim"]
            self.contrastive_projection = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, proj_dim)
            )

    def forward(self, input_ids, attention_mask, audio_temporal, video_temporal):
        hidden_dim = MODEL_CONFIG["hidden_dim"]

        # Text encoding
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state  # [batch, seq_len, hidden]
        text_pooled = text_features[:, 0, :]  # CLS token
        text_encoded = self.text_projection(text_pooled)  # [batch, hidden_dim]

        # Audio temporal encoding
        audio_encoded = self.audio_tcn(audio_temporal)  # [batch, audio_frames, hidden_dim]
        audio_pooled = audio_encoded.mean(dim=1)  # Global average pooling

        # Video temporal encoding
        video_encoded = self.video_tcn(video_temporal)  # [batch, video_frames, hidden_dim]
        video_pooled = video_encoded.mean(dim=1)  # Global average pooling

        # Cross-modal fusion (simplified for this version)
        text_seq = text_encoded.unsqueeze(1)  # [batch, 1, hidden_dim]
        audio_seq = audio_pooled.unsqueeze(1)  # [batch, 1, hidden_dim]
        video_seq = video_pooled.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Apply a few cross-modal attention layers
        for layer in self.cross_modal_layers[:3]:  # Use only first 3 layers for efficiency
            text_seq = layer(text_seq, torch.cat([audio_seq, video_seq], dim=1),
                           torch.cat([audio_seq, video_seq], dim=1))

        text_final = text_seq.squeeze(1)

        # Final fusion
        fusion_type = MODEL_CONFIG["fusion_type"]
        if fusion_type == "dynamic":
            fused_features = self.fusion([text_final, audio_pooled, video_pooled])
        else:
            fused_features = self.fusion(torch.cat([text_final, audio_pooled, video_pooled], dim=1))

        # Task predictions
        depression_pred = self.depression_regressor(fused_features).squeeze(1)

        outputs = {'depression_pred': depression_pred}

        # Auxiliary tasks (if enabled)
        if MULTITASK_CONFIG["use_auxiliary_tasks"] and hasattr(self, 'emotion_classifier'):
            outputs['emotion_class_pred'] = self.emotion_classifier(fused_features)

        if MULTITASK_CONFIG["use_contrastive"] and hasattr(self, 'contrastive_projection'):
            contrastive_emb = self.contrastive_projection(fused_features)
            outputs['contrastive_emb'] = F.normalize(contrastive_emb, p=2, dim=1)

        return outputs


# ============ LOSS FUNCTIONS ============
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = MULTITASK_CONFIG["contrastive_temperature"]

    def forward(self, embeddings, labels):
        batch_size = embeddings.size(0)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Create labels for contrastive learning
        labels = labels.unsqueeze(1)
        mask = torch.eq(labels, labels.T).float()

        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=mask.device)

        # Compute contrastive loss
        exp_sim = torch.exp(similarity_matrix)
        exp_sim = exp_sim * (1 - torch.eye(batch_size, device=exp_sim.device))

        positive_pairs = exp_sim * mask
        negative_pairs = exp_sim.sum(dim=1, keepdim=True)

        loss = -torch.log(positive_pairs.sum(dim=1) / negative_pairs.squeeze(1))
        loss = loss[~torch.isnan(loss)]  # Remove NaN values

        return loss.mean() if len(loss) > 0 else torch.tensor(0.0, device=embeddings.device)


# ============ TRAINING FUNCTION ============
def train_configurable_model(model, train_loader, val_loader, device):
    model.to(device)

    # Get training configuration
    epochs = TRAINING_CONFIG["epochs"]
    lr = TRAINING_CONFIG["learning_rate"]
    weight_decay = TRAINING_CONFIG["weight_decay"]
    grad_clip = TRAINING_CONFIG["grad_clip_norm"]

    # Loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    contrastive_loss = ContrastiveLoss() if MULTITASK_CONFIG["use_contrastive"] else None

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=TRAINING_CONFIG["warmup_ratio"]
    )

    history = {'train_loss': [], 'val_rmse': [], 'val_accuracy': []}
    best_val_rmse = float('inf')

    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        n_samples = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in train_pbar:
            optimizer.zero_grad()

            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_temporal = batch['audio_temporal'].to(device)
            video_temporal = batch['video_temporal'].to(device)
            phq8 = batch['phq8'].to(device)
            emotion_category = batch['emotion_category'].to(device)

            # Clean inputs
            audio_temporal = torch.nan_to_num(audio_temporal, nan=0.0, posinf=1e6, neginf=-1e6)
            video_temporal = torch.nan_to_num(video_temporal, nan=0.0, posinf=1e6, neginf=-1e6)

            # Forward pass
            outputs = model(input_ids, attention_mask, audio_temporal, video_temporal)

            # Handle missing labels
            valid_mask = ~torch.isnan(phq8)
            if valid_mask.sum() == 0:
                continue

            # Main task loss
            main_loss = mse_loss(outputs['depression_pred'][valid_mask], phq8[valid_mask])
            total_batch_loss = main_loss * MULTITASK_CONFIG["main_task_weight"]

            # Auxiliary tasks
            if MULTITASK_CONFIG["use_auxiliary_tasks"] and 'emotion_class_pred' in outputs:
                aux_loss = ce_loss(outputs['emotion_class_pred'], emotion_category)
                total_batch_loss += aux_loss * MULTITASK_CONFIG["aux_task_weight"]

            if MULTITASK_CONFIG["use_contrastive"] and 'contrastive_emb' in outputs:
                cont_loss = contrastive_loss(outputs['contrastive_emb'], emotion_category)
                total_batch_loss += cont_loss * MULTITASK_CONFIG["contrastive_weight"]

            # Backward pass
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            # Accumulate loss
            batch_size = valid_mask.sum().item()
            total_loss += total_batch_loss.item() * batch_size
            n_samples += batch_size

            train_pbar.set_postfix({'Loss': f"{total_batch_loss.item():.4f}"})

        avg_train_loss = total_loss / n_samples if n_samples > 0 else 0

        # Validation phase
        model.eval()
        val_preds = []
        val_targets = []
        val_emotion_preds = []
        val_emotion_targets = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                audio_temporal = batch['audio_temporal'].to(device)
                video_temporal = batch['video_temporal'].to(device)
                phq8 = batch['phq8'].to(device)
                emotion_category = batch['emotion_category'].to(device)

                # Clean inputs
                audio_temporal = torch.nan_to_num(audio_temporal, nan=0.0, posinf=1e6, neginf=-1e6)
                video_temporal = torch.nan_to_num(video_temporal, nan=0.0, posinf=1e6, neginf=-1e6)

                outputs = model(input_ids, attention_mask, audio_temporal, video_temporal)

                # Collect predictions
                valid_mask = ~torch.isnan(phq8)
                if valid_mask.sum() > 0:
                    val_preds.extend(outputs['depression_pred'][valid_mask].cpu().numpy())
                    val_targets.extend(phq8[valid_mask].cpu().numpy())

                if 'emotion_class_pred' in outputs:
                    val_emotion_preds.extend(outputs['emotion_class_pred'].argmax(dim=1).cpu().numpy())
                    val_emotion_targets.extend(emotion_category.cpu().numpy())

        # Calculate metrics
        val_rmse = math.sqrt(mean_squared_error(val_targets, val_preds)) if val_preds else float('inf')
        val_accuracy = accuracy_score(val_emotion_targets, val_emotion_preds) if val_emotion_preds else 0.0

        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_rmse'].append(val_rmse)
        history['val_accuracy'].append(val_accuracy)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, "
              f"Val RMSE={val_rmse:.4f}, Val Acc={val_accuracy:.4f}")

        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            if OUTPUT_CONFIG["save_best_model"]:
                out_dir = OUTPUT_CONFIG["output_dir"]
                os.makedirs(out_dir, exist_ok=True)
                model_path = os.path.join(out_dir, f"{OUTPUT_CONFIG['model_name']}_best.pt")
                torch.save(model.state_dict(), model_path)
                print(f"  → Best model saved: {model_path}")

    return history, best_val_rmse


# ============ EVALUATION FUNCTION ============
def evaluate_model(model, test_loader, device):
    """Evaluate model on test set with comprehensive metrics"""
    model.eval()
    phq8_true, phq8_pred = [], []
    emotion_true, emotion_pred = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_temporal = batch['audio_temporal'].to(device)
            video_temporal = batch['video_temporal'].to(device)
            phq8 = batch['phq8'].to(device)
            emotion_category = batch['emotion_category'].to(device)
            
            # Clean inputs
            audio_temporal = torch.nan_to_num(audio_temporal, nan=0.0, posinf=1e6, neginf=-1e6)
            video_temporal = torch.nan_to_num(video_temporal, nan=0.0, posinf=1e6, neginf=-1e6)
            
            outputs = model(input_ids, attention_mask, audio_temporal, video_temporal)
            
            # Collect PHQ8 predictions
            valid_mask = ~torch.isnan(phq8)
            if valid_mask.sum() > 0:
                phq8_true.extend(phq8[valid_mask].cpu().numpy())
                phq8_pred.extend(outputs['depression_pred'][valid_mask].cpu().numpy())
            
            # Collect emotion predictions
            if 'emotion_class_pred' in outputs:
                emotion_pred.extend(outputs['emotion_class_pred'].argmax(dim=1).cpu().numpy())
                emotion_true.extend(emotion_category.cpu().numpy())
    
    # Compute metrics
    reg_metrics = compute_regression_metrics(np.array(phq8_true), np.array(phq8_pred))
    cls_metrics = compute_classification_metrics(np.array(emotion_true), np.array(emotion_pred)) if emotion_pred else None
    
    return reg_metrics, cls_metrics


# ============ MAIN EXECUTION ============
def main():
    print("🚀 Advanced Multi-Modal Emotion Detection (Configurable)")
    print("=" * 60)

    # Validate configuration
    print("⚙️  Validating configuration...")
    errors = validate_config()
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        return
    print("✅ Configuration valid!")

    # Print configuration summary
    if OUTPUT_CONFIG["verbose_logging"]:
        print(get_config_summary())

    # Load data
    print("\n📊 Loading data...")
    train_df = read_split_csv(TRAIN_SPLIT)
    dev_df = read_split_csv(DEV_SPLIT)
    print(f"  Train samples: {len(train_df)}, Dev samples: {len(dev_df)}")

    # Build dataset
    print("🔧 Building dataset indices...")
    train_rows = build_enhanced_dataset_index(train_df)
    dev_rows = build_enhanced_dataset_index(dev_df)

    if not train_rows or not dev_rows:
        print("❌ No valid data found!")
        return

    print(f"  Valid samples - Train: {len(train_rows)}, Dev: {len(dev_rows)}")

    # Get feature dimensions
    audio_dim = train_rows[0]['audio_temporal'].shape[1]
    video_dim = train_rows[0]['video_temporal'].shape[1]
    print(f"  Feature dimensions - Audio: {audio_dim}, Video: {video_dim}")

    # Initialize tokenizer and datasets
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["text_model"])
    train_dataset = ConfigurableDAICWozDataset(train_rows, tokenizer, is_training=True)
    dev_dataset = ConfigurableDAICWozDataset(dev_rows, tokenizer, is_training=False)

    # Create data loaders
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
    print(f"\n🏗️  Initializing model...")
    model = ConfigurableMultiModalModel(audio_dim=audio_dim, video_dim=video_dim)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")

    print(f"  Trainable parameters: {trainable_params:,}")
    # Training
    print(f"\n🎯 Starting training...")
    history, best_rmse = train_configurable_model(model, train_loader, dev_loader, DEVICE)

    # Save final results
    print("\n💾 Saving results...")
    out_dir = OUTPUT_CONFIG["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    if OUTPUT_CONFIG["save_final_model"]:
        final_model_path = os.path.join(out_dir, f"{OUTPUT_CONFIG['model_name']}_final.pt")
        torch.save(model.state_dict(), final_model_path)
        print(f"  Final model saved: {final_model_path}")

    # Save training history
    history_path = os.path.join(out_dir, f"{OUTPUT_CONFIG['model_name']}_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"  Training history saved: {history_path}")

    # Add test set evaluation
    print("\n📊 Loading test data...")
    test_df = read_split_csv(TEST_SPLIT)
    print(f"Test samples: {len(test_df)}")
    
    print("🔧 Building test dataset...")
    test_rows = build_enhanced_dataset_index(test_df)
    test_dataset = ConfigurableDAICWozDataset(test_rows, tokenizer, is_training=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAINING_CONFIG["batch_size"],
        shuffle=False,
        num_workers=TRAINING_CONFIG["num_workers"],
        pin_memory=TRAINING_CONFIG["pin_memory"]
    )
    
    # Load best model for evaluation
    best_model_path = os.path.join(out_dir, f"{OUTPUT_CONFIG['model_name']}_best.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded best model for evaluation")
    
    print("\n📈 Evaluating model on test set...")
    reg_metrics, cls_metrics = evaluate_model(model, test_loader, DEVICE)
    print_metrics_summary(reg_metrics, cls_metrics)
    
    # Save metrics
    metrics = {
        'regression': reg_metrics,
        'classification': {k:v.tolist() if isinstance(v, np.ndarray) else v 
                         for k,v in cls_metrics.items()} if cls_metrics else None
    }
    metrics_path = os.path.join(out_dir, f"{OUTPUT_CONFIG['model_name']}_test_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Test metrics saved: {metrics_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 Training Complete!")
    print(f"  Best validation RMSE: {best_rmse:.4f}")
    print(f"  Final validation RMSE: {history['val_rmse'][-1]:.4f}")
    if history['val_accuracy']:
        print(f"  Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()