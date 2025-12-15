import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_squared_error, classification_report
import wandb
from typing import Dict, List, Tuple, Optional
import random
from einops import rearrange, reduce
from tqdm.auto import tqdm

# ---------- CONFIG ----------
DATA_ROOT = "/home/dipanjan/rugraj/DIAC-WOZ/"  # change if needed
TRAIN_SPLIT = os.path.join(DATA_ROOT, "train_split_Depression_AVEC2017.csv")
DEV_SPLIT   = os.path.join(DATA_ROOT, "dev_split_Depression_AVEC2017.csv")
TEST_SPLIT  = os.path.join(DATA_ROOT, "test_split_Depression_AVEC2017.csv")
SESSION_GLOB = os.path.join(DATA_ROOT, "*_P")

# Hyperparameters
BATCH_SIZE = 6
LR = 3e-5
EPOCHS = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 256
GRAD_CLIP_NORM = 1.0
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01

# Model Architecture Hyperparameters
HIDDEN_DIM = 512
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.15
CONTRASTIVE_TEMP = 0.07
FUSION_TYPE = "dynamic"  # "concat", "attention", "dynamic"

# Training Configuration
USE_CONTRASTIVE = True
USE_AUXILIARY_TASKS = True
CONTRASTIVE_WEIGHT = 0.3
AUX_TASK_WEIGHT = 0.2
MAIN_TASK_WEIGHT = 1.0

# Diffusion Hyperparameters
NUM_TIMESTEPS = 1000
BETA_START = 0.0001
BETA_END = 0.02
GUIDANCE_SCALE = 3.0

# ----------------------------

# ---------- ENHANCED DATA LOADING WITH TEMPORAL FEATURES ----------
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


def build_enhanced_dataset_index(split_df, max_audio_frames=100, max_video_frames=100):
    """Enhanced dataset building with temporal features"""
    rows = []
    for _, r in split_df.iterrows():
        sfolder = os.path.join(DATA_ROOT, str(r['session_folder']))
        if not os.path.isdir(sfolder):
            candidates = glob.glob(os.path.join(DATA_ROOT, f"{r['participant']}*_P")) + \
                        glob.glob(os.path.join(DATA_ROOT, f"*{r['participant']}*_P"))
            if candidates:
                sfolder = candidates[0]
            else:
                print(f"WARNING: session folder not found for {r['participant']}: expected {r['session_folder']}")
                continue
        
        # Load enhanced features
        transcript, turns = read_transcript_with_turns(sfolder, participant_only=True)
        audio_temporal = load_temporal_covarep_features(sfolder, max_audio_frames)
        video_temporal = load_temporal_clnf_aus(sfolder, max_video_frames)
        
        if audio_temporal is None and video_temporal is None:
            print(f"WARNING: both audio and video missing for {sfolder}, skipping.")
            continue
        
        # Handle missing modalities with zeros
        if audio_temporal is None:
            audio_temporal = np.zeros((max_audio_frames, 10), dtype=np.float32)
        if video_temporal is None:
            video_temporal = np.zeros((max_video_frames, 10), dtype=np.float32)
        
        phq8 = r['phq8'] if not np.isnan(r['phq8']) else None
        
        # Create emotion category for auxiliary task (binary: low vs high depression)
        emotion_category = 0 if phq8 is not None and phq8 <= 10 else 1
        
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
    
    return rows


# ---------- ADVANCED DATASET CLASS ----------
class AdvancedDAICWozDataset(Dataset):
    def __init__(self, rows, tokenizer, max_len=256, max_audio_frames=100, max_video_frames=100, augment=False):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_audio_frames = max_audio_frames
        self.max_video_frames = max_video_frames
        self.augment = augment
        
    def __len__(self):
        return len(self.rows)
    
    def _augment_audio(self, audio_features):
        """Simple audio augmentation through noise injection"""
        if self.augment and random.random() < 0.3:
            noise = np.random.normal(0, 0.01, audio_features.shape).astype(np.float32)
            audio_features = audio_features + noise
        return audio_features
    
    def _augment_video(self, video_features):
        """Simple video augmentation through feature dropout"""
        if self.augment and random.random() < 0.3:
            mask = np.random.random(video_features.shape) > 0.1  # 10% dropout
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
        
        # Audio and video sequence lengths for attention masking
        item['audio_length'] = torch.tensor(self.max_audio_frames, dtype=torch.long)
        item['video_length'] = torch.tensor(self.max_video_frames, dtype=torch.long)
        
        return item


# ---------- ADVANCED MODEL ARCHITECTURE ----------
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
    """Temporal Convolutional Network for sequential feature processing"""
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()
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
        self.dropout = nn.Dropout(DROPOUT)
    
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
    """Cross-modal attention mechanism"""
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=DROPOUT, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, query, key, value, key_padding_mask=None):
        # Self-attention
        attn_out, _ = self.multihead_attn(query, key, value, key_padding_mask=key_padding_mask)
        query = self.norm1(query + self.dropout(attn_out))
        
        # Feed forward
        ffn_out = self.ffn(query)
        query = self.norm2(query + self.dropout(ffn_out))
        
        return query


class DynamicFusion(nn.Module):
    """Dynamic fusion with learnable modality weights"""
    def __init__(self, input_dims, output_dim):
        super().__init__()
        self.modality_weights = nn.Parameter(torch.ones(len(input_dims)))
        self.fusion_layers = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in input_dims
        ])
        self.final_projection = nn.Linear(output_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
    
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


class HierarchicalCrossModalTransformer(nn.Module):
    """Main model: Hierarchical Cross-Modal Transformer with Contrastive Learning"""
    
    def __init__(self, audio_dim, video_dim, text_dim=768, hidden_dim=512, num_heads=8, num_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        
        # Temporal encoders
        self.audio_tcn = TemporalConvNet(audio_dim, hidden_dim)
        self.video_tcn = TemporalConvNet(video_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim)
        
        # Cross-modal attention layers
        self.cross_modal_layers = nn.ModuleList([
            CrossModalAttention(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Fusion mechanism
        if FUSION_TYPE == "dynamic":
            self.fusion = DynamicFusion([hidden_dim, hidden_dim, hidden_dim], hidden_dim)
        else:
            self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        
        # Task-specific heads
        self.depression_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Auxiliary task head
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim // 2, 2)  # Binary classification
        )
        
        # Contrastive learning projection
        self.contrastive_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 128)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode_modalities(self, input_ids, attention_mask, audio_temporal, video_temporal):
        # Text encoding
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state  # [batch, seq_len, 768]
        text_pooled = text_features[:, 0, :]  # CLS token
        text_encoded = self.text_projection(text_pooled)  # [batch, hidden_dim]
        
        # Audio temporal encoding
        audio_encoded = self.audio_tcn(audio_temporal)  # [batch, audio_frames, hidden_dim]
        audio_pooled = audio_encoded.mean(dim=1)  # Global average pooling
        
        # Video temporal encoding  
        video_encoded = self.video_tcn(video_temporal)  # [batch, video_frames, hidden_dim]
        video_pooled = video_encoded.mean(dim=1)  # Global average pooling
        
        return text_encoded, audio_pooled, video_pooled, audio_encoded, video_encoded
    
    def cross_modal_fusion(self, text_features, audio_features, video_features):
        # Prepare for cross-modal attention
        # Stack features for attention computation
        batch_size = text_features.size(0)
        
        # Expand to sequence format for attention
        text_seq = text_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        audio_seq = audio_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        video_seq = video_features.unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Apply positional encoding
        text_seq = self.pos_encoding(text_seq)
        audio_seq = self.pos_encoding(audio_seq)
        video_seq = self.pos_encoding(video_seq)
        
        # Cross-modal attention
        for layer in self.cross_modal_layers:
            # Text attends to audio and video
            text_seq = layer(text_seq, torch.cat([audio_seq, video_seq], dim=1), 
                           torch.cat([audio_seq, video_seq], dim=1))
            
            # Audio attends to text and video  
            audio_seq = layer(audio_seq, torch.cat([text_seq, video_seq], dim=1),
                            torch.cat([text_seq, video_seq], dim=1))
            
            # Video attends to text and audio
            video_seq = layer(video_seq, torch.cat([text_seq, audio_seq], dim=1),
                            torch.cat([text_seq, audio_seq], dim=1))
        
        # Extract features
        text_final = text_seq.squeeze(1)
        audio_final = audio_seq.squeeze(1)
        video_final = video_seq.squeeze(1)
        
        return text_final, audio_final, video_final
    
    def forward(self, input_ids, attention_mask, audio_temporal, video_temporal, return_embeddings=False):
        # Encode each modality
        text_enc, audio_enc, video_enc, audio_seq, video_seq = self.encode_modalities(
            input_ids, attention_mask, audio_temporal, video_temporal
        )
        
        # Cross-modal fusion
        text_fused, audio_fused, video_fused = self.cross_modal_fusion(
            text_enc, audio_enc, video_enc
        )
        
        # Final fusion
        if FUSION_TYPE == "dynamic":
            fused_features = self.fusion([text_fused, audio_fused, video_fused])
        else:
            fused_features = self.fusion(torch.cat([text_fused, audio_fused, video_fused], dim=1))
        
        # Task predictions
        depression_pred = self.depression_regressor(fused_features).squeeze(1)
        emotion_class_pred = self.emotion_classifier(fused_features)
        
        # Contrastive embeddings
        contrastive_emb = self.contrastive_projection(fused_features)
        contrastive_emb = F.normalize(contrastive_emb, p=2, dim=1)
        
        outputs = {
            'depression_pred': depression_pred,
            'emotion_class_pred': emotion_class_pred,
            'contrastive_emb': contrastive_emb,
            'fused_features': fused_features
        }
        
        if return_embeddings:
            outputs.update({
                'text_features': text_fused,
                'audio_features': audio_fused,
                'video_features': video_fused
            })
        
        return outputs


# ---------- DIFFUSION SCHEDULER ----------
class DiffusionScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        
    def add_noise(self, x, t):
        sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod[t])
        sqrt_one_minus_alpha_cumprod = torch.sqrt(1 - self.alpha_cumprod[t])
        ε = torch.randn_like(x)
        return sqrt_alpha_cumprod * x + sqrt_one_minus_alpha_cumprod * ε, ε
    
    def remove_noise(self, x, t, pred_noise):
        alpha_cumprod = self.alpha_cumprod[t]
        alpha_cumprod_prev = self.alpha_cumprod[t-1] if t > 0 else torch.ones_like(alpha_cumprod)
        beta = self.betas[t]
        
        pred_x0 = (x - torch.sqrt(1 - alpha_cumprod) * pred_noise) / torch.sqrt(alpha_cumprod)
        direction = torch.sqrt(1 - alpha_cumprod_prev) * pred_noise
        noise_scale = torch.sqrt(beta) * pred_noise
        
        return pred_x0 + direction + noise_scale


# ---------- DIFFUSION-BASED MULTIMODAL TRANSFORMER ----------
class MultimodalDiffusionTransformer(nn.Module):
    def __init__(self, audio_dim, video_dim, text_dim=768, hidden_dim=512):
        super().__init__()
        # Text encoder
        self.text_encoder = AutoModel.from_pretrained("distilbert-base-uncased")
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        
        # Temporal encoders
        self.audio_encoder = TemporalConvNet(audio_dim, hidden_dim)
        self.video_encoder = TemporalConvNet(video_dim, hidden_dim)
        
        # Diffusion components
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.noise_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Enhanced classification heads
        self.depression_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, 1)
        )
        
        self.emotion_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, 2)
        )
        
        self.scheduler = DiffusionScheduler(
            num_timesteps=NUM_TIMESTEPS,
            beta_start=BETA_START,
            beta_end=BETA_END
        )
    
    def forward(self, input_ids, attention_mask, audio_temporal, video_temporal, timesteps=None):
        # Get multimodal features
        text_features = self.text_encoder(input_ids, attention_mask).last_hidden_state
        audio_features = self.audio_encoder(audio_temporal)
        video_features = self.video_encoder(video_temporal)
        
        # Fusion
        text_features = text_features.mean(dim=1)  # Average pooling over sequence
        fused = torch.cat([text_features, audio_features, video_features], dim=-1)
        fused = nn.Linear(fused.shape[-1], self.hidden_dim).to(fused.device)(fused)
        
        if timesteps is not None:
            # Training - predict noise
            time_emb = self.time_embedding(timesteps.unsqueeze(-1).float())
            noised_features, target_noise = self.scheduler.add_noise(fused, timesteps)
            
            # Predict noise
            combined_features = torch.cat([noised_features, time_emb], dim=-1)
            pred_noise = self.noise_predictor(combined_features)
            
            # Get predictions
            depression_pred = self.depression_classifier(fused)
            emotion_pred = self.emotion_classifier(fused)
            
            return {
                'pred_noise': pred_noise,
                'target_noise': target_noise,
                'depression_pred': depression_pred,
                'emotion_pred': emotion_pred,
                'fused_features': fused
            }
        
        else:
            # Inference - denoise
            x = torch.randn_like(fused)
            for t in reversed(range(NUM_TIMESTEPS)):
                timesteps = torch.full((x.shape[0],), t, device=x.device)
                time_emb = self.time_embedding(timesteps.unsqueeze(-1).float())
                combined_features = torch.cat([x, time_emb], dim=-1)
                pred_noise = self.noise_predictor(combined_features)
                x = self.scheduler.remove_noise(x, t, pred_noise)
            
            # Get predictions from denoised features
            depression_pred = self.depression_classifier(x)
            emotion_pred = self.emotion_classifier(x)
            
            return {
                'depression_pred': depression_pred,
                'emotion_pred': emotion_pred,
                'denoised_features': x
            }


def compute_metrics(outputs, targets):
    """Compute comprehensive metrics"""
    depression_preds = outputs['depression_pred'].sigmoid().round()
    emotion_preds = outputs['emotion_pred'].argmax(dim=1)
    
    metrics = {}
    
    # Depression metrics
    metrics['depression_accuracy'] = (depression_preds == targets['depression']).float().mean()
    metrics['depression_f1'] = f1_score(targets['depression'].cpu(), depression_preds.cpu())
    metrics['depression_recall'] = recall_score(targets['depression'].cpu(), depression_preds.cpu())
    metrics['depression_precision'] = precision_score(targets['depression'].cpu(), depression_preds.cpu())
    
    # Emotion metrics
    metrics['emotion_accuracy'] = (emotion_preds == targets['emotion']).float().mean()
    metrics['emotion_f1'] = f1_score(targets['emotion'].cpu(), emotion_preds.cpu(), average='weighted')
    metrics['emotion_recall'] = recall_score(targets['emotion'].cpu(), emotion_preds.cpu(), average='weighted')
    metrics['emotion_precision'] = precision_score(targets['emotion'].cpu(), emotion_preds.cpu(), average='weighted')
    
    return metrics

# ---------- ENHANCED TRAINING LOOP ----------
def train_advanced_model(model, train_loader, val_loader, device, epochs=60):
    model.to(device)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    contrastive_loss = ContrastiveLoss(CONTRASTIVE_TEMP)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LR, 
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    history = {
        'train_loss': [], 'train_depression_loss': [], 'train_emotion_loss': [], 
        'train_contrastive_loss': [], 'val_rmse': [], 'val_accuracy': []
    }
    
    best_val_rmse = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        total_depression_loss = 0.0
        total_emotion_loss = 0.0
        total_contrastive_loss = 0.0
        n_samples = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch_idx, batch in enumerate(train_pbar):
            optimizer.zero_grad()
            
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            audio_temporal = batch['audio_temporal'].to(device)
            video_temporal = batch['video_temporal'].to(device)
            phq8 = batch['phq8'].to(device)
            emotion_category = batch['emotion_category'].to(device)
            
            # Get batch size from input tensor
            batch_size = input_ids.size(0)
            
            # Clean inputs
            audio_temporal = torch.nan_to_num(audio_temporal, nan=0.0, posinf=1e6, neginf=-1e6)
            video_temporal = torch.nan_to_num(video_temporal, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Add timesteps
            timesteps = torch.randint(0, NUM_TIMESTEPS, (batch_size,), device=device)
            
            # Forward pass with timesteps
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_temporal=audio_temporal,
                video_temporal=video_temporal,
                timesteps=timesteps
            )
            
            # Get valid mask for depression predictions
            valid_mask = ~torch.isnan(phq8)
            
            # Calculate losses
            noise_loss = F.mse_loss(outputs['pred_noise'], outputs['target_noise'])
            depression_loss = mse_loss(
                outputs['depression_pred'][valid_mask], 
                phq8[valid_mask]
            ) * MAIN_TASK_WEIGHT if valid_mask.any() else torch.tensor(0.0, device=device)

            # Auxiliary task loss (emotion classification)
            emotion_loss = ce_loss(
                outputs['emotion_class_pred'], 
                emotion_category
            ) * AUX_TASK_WEIGHT
            
            # Contrastive loss
            if USE_CONTRASTIVE:
                contrastive_loss_val = contrastive_loss(
                    outputs['contrastive_emb'], 
                    emotion_category
                ) * CONTRASTIVE_WEIGHT
            else:
                contrastive_loss_val = torch.tensor(0.0, device=device)
            
            # Total loss
            total_batch_loss = noise_loss + depression_loss + emotion_loss + contrastive_loss_val
            
            # Backward pass
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()
            
            # Accumulate losses
            batch_size = valid_mask.sum().item()
            total_loss += total_batch_loss.item() * batch_size
            total_depression_loss += depression_loss.item() * batch_size
            total_emotion_loss += emotion_loss.item() * batch_size
            total_contrastive_loss += contrastive_loss_val.item() * batch_size
            n_samples += batch_size
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f"{total_batch_loss.item():.4f}",
                'LR': f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
        # Average training losses
        avg_train_loss = total_loss / n_samples if n_samples > 0 else 0
        avg_depression_loss = total_depression_loss / n_samples if n_samples > 0 else 0
        avg_emotion_loss = total_emotion_loss / n_samples if n_samples > 0 else 0
        avg_contrastive_loss = total_contrastive_loss / n_samples if n_samples > 0 else 0
        
        # Validation phase
        model.eval()
        val_depression_preds = []
        val_depression_targets = []
        val_emotion_preds = []
        val_emotion_targets = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
            for batch in val_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                audio_temporal = batch['audio_temporal'].to(device)
                video_temporal = batch['video_temporal'].to(device)
                phq8 = batch['phq8'].to(device)
                emotion_category = batch['emotion_category'].to(device)
                
                # Clean inputs
                audio_temporal = torch.nan_to_num(audio_temporal, nan=0.0, posinf=1e6, neginf=-1e6)
                video_temporal = torch.nan_to_num(video_temporal, nan=0.0, posinf=1e6, neginf=-1e6)
                
                # Don't pass timesteps during inference
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    audio_temporal=audio_temporal,
                    video_temporal=video_temporal
                )
                
                # Collect predictions for depression regression
                valid_mask = ~torch.isnan(phq8)
                if valid_mask.sum() > 0:
                    val_depression_preds.extend(
                        torch.nan_to_num(outputs['depression_pred'][valid_mask]).cpu().numpy()
                    )
                    val_depression_targets.extend(phq8[valid_mask].cpu().numpy())
                
                # Collect predictions for emotion classification
                val_emotion_preds.extend(
                    outputs['emotion_class_pred'].argmax(dim=1).cpu().numpy()
                )
                val_emotion_targets.extend(emotion_category.cpu().numpy())
        
        # Calculate validation metrics
        if len(val_depression_preds) > 0:
            val_rmse = math.sqrt(mean_squared_error(val_depression_targets, val_depression_preds))
        else:
            val_rmse = float('inf')
        
        val_accuracy = np.mean(np.array(val_emotion_preds) == np.array(val_emotion_targets))
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['train_depression_loss'].append(avg_depression_loss)
        history['train_emotion_loss'].append(avg_emotion_loss)
        history['train_contrastive_loss'].append(avg_contrastive_loss)
        history['val_rmse'].append(val_rmse)
        history['val_accuracy'].append(val_accuracy)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f} (Dep: {avg_depression_loss:.4f}, "
              f"Emo: {avg_emotion_loss:.4f}, Cont: {avg_contrastive_loss:.4f})")
        print(f"  Val RMSE: {val_rmse:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            out_dir = os.path.join(DATA_ROOT, "trained_models")
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(out_dir, "advanced_multimodal_best.pt"))
            print(f"  → New best model saved! (RMSE: {val_rmse:.4f})")
    
    return history, best_val_rmse


# Add this class before train_advanced_model function
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Get similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create labels matrix
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float()
        
        # Remove diagonal elements (self-similarity)
        mask = mask - torch.eye(mask.shape[0], device=mask.device)
        
        # Calculate positive and negative pairs
        exp_sim = torch.exp(similarity_matrix)
        pos_pairs = (similarity_matrix * mask).sum(dim=1)
        neg_pairs = torch.log(exp_sim.sum(dim=1) - exp_sim.diag())
        
        loss = -(pos_pairs - neg_pairs).mean()
        return loss

# ---------- MAIN EXECUTION ----------
def main():
    print("🚀 Starting Advanced Multi-Modal Emotion Detection Training")
    print("=" * 60)
    
    # Load data splits
    print("📊 Loading data splits...")
    train_df = read_split_csv(TRAIN_SPLIT)
    dev_df = read_split_csv(DEV_SPLIT)
    print(f"   Train samples: {len(train_df)}")
    print(f"   Dev samples: {len(dev_df)}")
    
    # Build enhanced dataset indices
    print("\n🔧 Building enhanced dataset indices...")
    train_rows = build_enhanced_dataset_index(train_df, max_audio_frames=100, max_video_frames=100)
    dev_rows = build_enhanced_dataset_index(dev_df, max_audio_frames=100, max_video_frames=100)
    
    if not train_rows or not dev_rows:
        print("❌ Error: No valid data found!")
        return
    
    print(f"   Valid train samples: {len(train_rows)}")
    print(f"   Valid dev samples: {len(dev_rows)}")
    
    # Detect feature dimensions
    audio_dim = train_rows[0]['audio_temporal'].shape[1]
    video_dim = train_rows[0]['video_temporal'].shape[1]
    print(f"   Audio feature dim: {audio_dim}")
    print(f"   Video feature dim: {video_dim}")
    
    # Initialize tokenizer and datasets
    print("\n🤖 Initializing tokenizer and datasets...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    train_dataset = AdvancedDAICWozDataset(
        train_rows, tokenizer, max_len=MAX_LEN, 
        max_audio_frames=100, max_video_frames=100, augment=True
    )
    dev_dataset = AdvancedDAICWozDataset(
        dev_rows, tokenizer, max_len=MAX_LEN,
        max_audio_frames=100, max_video_frames=100, augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=2, pin_memory=True, drop_last=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    # Initialize model
    print(f"\n🏗️  Initializing Multimodal Diffusion Transformer...")
    print(f"   Architecture: {NUM_LAYERS} layers, {NUM_HEADS} heads, {HIDDEN_DIM} hidden dim")
    print(f"   Diffusion steps: {NUM_TIMESTEPS}")
    print(f"   Guidance scale: {GUIDANCE_SCALE}")
    
    model = MultimodalDiffusionTransformer(
        audio_dim=audio_dim,
        video_dim=video_dim,
        text_dim=768,  # DistilBERT
        hidden_dim=HIDDEN_DIM
    )
    
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Training
    print(f"\n🎯 Starting training for {EPOCHS} epochs...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LR}")
    print(f"   Device: {DEVICE}")
    
    history, best_rmse = train_advanced_model(
        model, train_loader, dev_loader, DEVICE, epochs=EPOCHS
    )
    
    # Save results
    print("\n💾 Saving training results...")
    out_dir = os.path.join(DATA_ROOT, "trained_models")
    os.makedirs(out_dir, exist_ok=True)
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(out_dir, "advanced_multimodal_final.pt"))
    
    # Save training history
    pd.DataFrame(history).to_csv(
        os.path.join(out_dir, "advanced_training_history.csv"), index=False
    )
    
    # Print final results
    print("=" * 60)
    print("🎉 Training Complete!")
    print(f"   Best validation RMSE: {best_rmse:.4f}")
    print(f"   Final validation RMSE: {history['val_rmse'][-1]:.4f}")
    print(f"   Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    print(f"   Models saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
