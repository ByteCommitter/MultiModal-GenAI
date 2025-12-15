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

# ---------- CONFIG ----------
DATA_ROOT = "/home/dipanjan/rugraj/DIAC-WOZ/"  # change if needed
TRAIN_SPLIT = os.path.join(DATA_ROOT, "train_split_Depression_AVEC2017.csv")
DEV_SPLIT   = os.path.join(DATA_ROOT, "dev_split_Depression_AVEC2017.csv")
TEST_SPLIT  = os.path.join(DATA_ROOT, "test_split_Depression_AVEC2017.csv")
SESSION_GLOB = os.path.join(DATA_ROOT, "*_P")
BATCH_SIZE = 8
LR = 5e-5               # lowered LR for stability (was 1e-4)
EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
GRAD_CLIP_NORM = 1.0
# ----------------------------

# ---------- HELPERS (same as yours, with nan-safe returns) ----------
def read_transcript(session_folder, participant_only=True):
    file_candidate = glob.glob(os.path.join(session_folder, "*_TRANSCRIPT.csv"))
    if not file_candidate:
        return ""
    fp = file_candidate[0]
    try:
        df = pd.read_csv(fp, sep='\t', header=None, quoting=3, dtype=str, engine='python')
    except Exception:
        df = pd.read_csv(fp, sep=None, engine='python', dtype=str)
    # try header-aware read
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
    # heuristic fallback
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
    if pooling == "mean":
        vec = np.nanmean(numeric, axis=0)
    elif pooling == "median":
        vec = np.nanmedian(numeric, axis=0)
    else:
        vec = np.nanmean(numeric, axis=0)
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
    if pooling == "mean":
        vec = np.nanmean(numeric, axis=0)
    else:
        vec = np.nanmedian(numeric, axis=0)
    vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
    return vec


def read_split_csv(split_csv):
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


def build_dataset_index(split_df):
    rows = []
    for _, r in split_df.iterrows():
        sfolder = os.path.join(DATA_ROOT, str(r['session_folder']))
        if not os.path.isdir(sfolder):
            candidates = glob.glob(os.path.join(DATA_ROOT, f"{r['participant']}*_P")) + glob.glob(os.path.join(DATA_ROOT, f"*{r['participant']}*_P"))
            if candidates:
                sfolder = candidates[0]
            else:
                print(f"WARNING: session folder not found for {r['participant']}: expected {r['session_folder']}")
                continue
        transcript = read_transcript(sfolder, participant_only=True)
        covarep_vec = load_covarep_features(sfolder)
        clnf_vec = load_clnf_aus(sfolder)
        if covarep_vec is None and clnf_vec is None:
            print(f"WARNING: both covarep and clnf missing for {sfolder}, skipping.")
            continue
        if covarep_vec is None:
            covarep_vec = np.zeros(10, dtype=np.float32)
        if clnf_vec is None:
            clnf_vec = np.zeros(10, dtype=np.float32)
        # final safety: ensure finite
        covarep_vec = np.nan_to_num(np.array(covarep_vec, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        clnf_vec = np.nan_to_num(np.array(clnf_vec, dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        phq8 = r['phq8'] if not np.isnan(r['phq8']) else None
        rows.append({
            'session_folder': sfolder,
            'transcript': transcript,
            'covarep': covarep_vec.astype(np.float32),
            'clnf_aus': clnf_vec.astype(np.float32),
            'phq8': phq8
        })
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
        # ensure shapes: audio/video must be 1D arrays of correct length
        item['audio'] = torch.tensor(np.nan_to_num(r['covarep'], nan=0.0, posinf=1e6, neginf=-1e6), dtype=torch.float)
        item['video'] = torch.tensor(np.nan_to_num(r['clnf_aus'], nan=0.0, posinf=1e6, neginf=-1e6), dtype=torch.float)
        item['phq8'] = torch.tensor(r['phq8'] if r['phq8'] is not None else np.nan, dtype=torch.float)
        return item


# ---------- LoRA helper (unchanged) ----------
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


# ---------- Model ----------
class MultiModalPHQ8LoRA(nn.Module):
    def __init__(self, audio_dim, video_dim, text_feat_dim=768, fused_dim=512):
        super().__init__()
        self.text_encoder = get_lora_bert_base_for_text()
        self.text_proj = nn.Linear(text_feat_dim, fused_dim)
        self.audio_proj = nn.Linear(audio_dim, fused_dim)
        self.video_proj = nn.Linear(video_dim, fused_dim)
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim * 3, fused_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.regressor = nn.Linear(fused_dim, 1)

    def forward(self, input_ids, attention_mask, audio, video):
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if hasattr(outputs, 'last_hidden_state'):
            cls_vec = outputs.last_hidden_state[:, 0, :]
        else:
            cls_vec = outputs.hidden_states[-1][:, 0, :]
        t = self.text_proj(cls_vec)
        # audio/video may come with shape [batch, audio_dim] already
        a = self.audio_proj(audio)
        v = self.video_proj(video)
        fused = torch.cat([t, a, v], dim=1)
        h = self.fusion(fused)
        phq8_pred = self.regressor(h).squeeze(1)
        return phq8_pred


# ---------- Training loop (hardened) ----------
def train_loop(model, train_loader, val_loader, optimizer, device, epochs=3):
    model.to(device)
    criterion = nn.MSELoss()
    history = {'train_loss': [], 'val_rmse': []}

    # Enable anomaly detection (helps trace bad ops). Comment out for speed if noisy.
    torch.autograd.set_detect_anomaly(True)

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        n = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {ep+1}")):
            optimizer.zero_grad()

            # move tensors
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            # cast/clean audio & video
            audio = batch['audio'].to(device).float()
            video = batch['video'].to(device).float()
            # replace any NaN/Inf in inputs
            audio = torch.nan_to_num(audio, nan=0.0, posinf=1e6, neginf=-1e6)
            video = torch.nan_to_num(video, nan=0.0, posinf=1e6, neginf=-1e6)

            phq8 = batch['phq8'].to(device)
            # skip samples with missing label
            mask = ~torch.isnan(phq8)
            if mask.sum() == 0:
                continue
            input_ids_m = input_ids[mask]
            attention_mask_m = attention_mask[mask]
            audio_m = audio[mask]
            video_m = video[mask]
            phq8_m = phq8[mask]

            preds = model(input_ids_m, attention_mask_m, audio_m, video_m)

            # safety: check preds for NaN/Inf before loss
            if torch.isnan(preds).any() or torch.isinf(preds).any():
                print(f"Warning: NaN/Inf in preds at epoch {ep+1} step {step}, skipping batch")
                continue

            loss = criterion(preds, phq8_m)
            try:
                loss.backward()
            except RuntimeError as e:
                print(f"Backward error at epoch {ep+1} step {step}: {e}. Skipping batch.")
                optimizer.zero_grad()
                continue

            # gradient clipping
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
                # nan-safe preds
                preds = torch.nan_to_num(preds, nan=0.0, posinf=1e6, neginf=-1e6)
                ys.extend(phq8[mask].cpu().numpy().tolist())
                ypred.extend(preds.cpu().numpy().tolist())

        if len(ys) > 0:
            # final safety: ensure no NaNs in arrays
            ys_arr = np.nan_to_num(np.array(ys, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
            ypred_arr = np.nan_to_num(np.array(ypred, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
            rmse = math.sqrt(mean_squared_error(ys_arr, ypred_arr))
        else:
            rmse = float('nan')
        history['val_rmse'].append(rmse)
        print(f"Epoch {ep+1}/{epochs} train_loss={avg_loss:.4f} val_RMSE={rmse:.4f}")
    return history


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

    history = train_loop(model, train_loader, dev_loader, optimizer, DEVICE, epochs=EPOCHS)

    out_dir = os.path.join(DATA_ROOT, "trained_models")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "multimodal_phq8_lora_fixed.pt"))
    pd.DataFrame(history).to_csv(os.path.join(out_dir, "train_history_fixed.csv"), index=False)
    print("Training complete. Model and history saved to:", out_dir)


if __name__ == "__main__":
    main()