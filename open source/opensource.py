import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import wandb
import glob

# Configuration
CONFIG = {
    "model_name": "j-hartmann/emotion-english-distilroberta-base",  # Pre-trained emotion detection model
    "data_root": "/home/dipanjan/rugraj/DIAC-WOZ/",
    "batch_size": 8,
    "max_length": 512,
    "epochs": 10,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "output_dir": "trained_models",
    "use_wandb": True,
    "num_workers": 4
}

class EmotionDataset(Dataset):
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        row = self.rows[idx]
        encoding = self.tokenizer(
            row['text'],
            truncation=True,
            max_length=CONFIG['max_length'],
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(row['emotion_label'])
        }

def read_transcript(session_folder):
    file_candidate = glob.glob(os.path.join(session_folder, "*_TRANSCRIPT.csv"))
    if not file_candidate:
        return ""
    
    try:
        df = pd.read_csv(file_candidate[0], sep='\t', dtype=str)
        if 'speaker' in df.columns and 'utterance' in df.columns:
            participant_texts = df[df['speaker'].str.lower() != 'ellie']['utterance'].fillna('')
            return " ".join(participant_texts)
    except:
        pass
    
    try:
        df = pd.read_csv(file_candidate[0], sep='\t', header=None, dtype=str)
        text_col = df.iloc[:, -1].fillna('')
        return " ".join(text_col)
    except:
        return ""

def prepare_dataset():
    # Read split files
    train_df = pd.read_csv(os.path.join(CONFIG['data_root'], "train_split_Depression_AVEC2017.csv"))
    dev_df = pd.read_csv(os.path.join(CONFIG['data_root'], "dev_split_Depression_AVEC2017.csv"))
    
    def process_split(df):
        rows = []
        for _, row in df.iterrows():
            session_folder = os.path.join(CONFIG['data_root'], f"{row['Participant_ID']}_P")
            text = read_transcript(session_folder)
            if text:
                # Map PHQ scores to emotion labels (customize based on your needs)
                phq_score = float(row['PHQ8_Score']) if 'PHQ8_Score' in row else 0
                # Simple mapping: 0-4: 0 (low), 5-9: 1 (mild), 10-14: 2 (moderate), 15+: 3 (severe)
                emotion_label = min(3, int(phq_score / 5))
                
                rows.append({
                    'text': text,
                    'emotion_label': emotion_label,
                    'phq_score': phq_score
                })
        return rows
    
    return process_split(train_df), process_split(dev_df)

def train_model(model, train_loader, val_loader, device):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    num_training_steps = len(train_loader) * CONFIG['epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=CONFIG['learning_rate'],
        total_steps=num_training_steps,
        pct_start=CONFIG['warmup_ratio']
    )
    
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        # Training
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{CONFIG["epochs"]} [Train]')
        
        for batch in train_pbar:
            optimizer.zero_grad()
            
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            
            outputs = model(**inputs)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                inputs = {
                    'input_ids': batch['input_ids'].to(device),
                    'attention_mask': batch['attention_mask'].to(device),
                    'labels': batch['labels'].to(device)
                }
                
                outputs = model(**inputs)
                val_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Log metrics
        if CONFIG['use_wandb']:
            wandb.log({
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'epoch': epoch + 1
            })
        
        print(f'\nEpoch {epoch+1}:')
        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Average validation loss: {avg_val_loss:.4f}')
        print('\nClassification Report:')
        print(classification_report(true_labels, predictions))
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(CONFIG['output_dir'], exist_ok=True)
            torch.save(model.state_dict(), 
                      os.path.join(CONFIG['output_dir'], 'best_emotion_model.pt'))

def main():
    # Initialize wandb
    if CONFIG['use_wandb']:
        wandb.init(project="diac-woz-emotion", config=CONFIG)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG['model_name'],
        num_labels=4  # Adjust based on your emotion categories
    ).to(device)
    
    # Prepare datasets
    print("Preparing datasets...")
    train_rows, val_rows = prepare_dataset()
    
    train_dataset = EmotionDataset(train_rows, tokenizer)
    val_dataset = EmotionDataset(val_rows, tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers']
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Train model
    print("\nStarting training...")
    train_model(model, train_loader, val_loader, device)
    
    if CONFIG['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    main()
