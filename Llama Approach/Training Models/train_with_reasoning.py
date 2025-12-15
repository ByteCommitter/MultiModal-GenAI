#!/usr/bin/env python3
"""
Final Experiment: Combining GenAI Scores + Text Embeddings of Reasoning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# CONFIG
BASE_DIR = Path("/home/dipanjan/rugraj/DIAC-WOZ")
OUTPUT_DIR = BASE_DIR / "final_results_embedding"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

def load_data():
    print("Loading data...")
    meta = pd.read_csv(BASE_DIR / "daic_metadata.csv")
    test_lbls = pd.read_csv(BASE_DIR / "full_test_split.csv")
    genai = pd.read_csv(BASE_DIR / "genai_features.csv")

    # Merge Labels
    test_lbls = test_lbls.rename(columns={'Participant_ID': 'participant_id', 'PHQ_Binary': 'phq8_binary'})
    meta['participant_id'] = meta['participant_id'].astype(str)
    test_lbls['participant_id'] = test_lbls['participant_id'].astype(str)
    genai['participant_id'] = genai['participant_id'].astype(str)

    meta = meta.merge(test_lbls[['participant_id', 'phq8_binary']], on='participant_id', how='left', suffixes=('', '_new'))
    meta['phq8_binary'] = meta['phq8_binary'].fillna(meta['phq8_binary_new'])
    
    # Merge Features
    df = meta.merge(genai, on='participant_id', how='inner')
    return df

def get_embeddings(text_list):
    """Convert text descriptions into vectors"""
    print("Encoding reasoning text (this takes a moment)...")
    # Small, fast model for embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_list, show_progress_bar=True)
    return embeddings

def main():
    df = load_data()
    
    # 1. numeric features
    score_cols = ['cognitive_negativity', 'emotional_flatness', 'overall_risk', 
                  'low_engagement', 'psychomotor_slowing']
    X_scores = df[score_cols].values
    
    # 2. text embeddings (The Reasoning Column)
    # Fill NaNs with empty string
    df['llm_reasoning'] = df['llm_reasoning'].fillna("No reasoning provided.")
    X_text = get_embeddings(df['llm_reasoning'].tolist())
    
    # 3. Combine them
    # X = Scores (5 dims) + Embeddings (384 dims)
    X_combined = np.hstack([X_scores, X_text])
    y = df['phq8_binary'].values
    
    # Split based on 'split' column
    train_mask = (df['split'] == 'train')
    test_mask = (df['split'] == 'test')
    
    X_train = X_combined[train_mask]
    y_train = y[train_mask]
    
    X_test = X_combined[test_mask]
    y_test = y[test_mask]
    
    print(f"\nTraining on {len(X_train)} samples, Testing on {len(X_test)}")
    
    # Scale (Important for mixing embeddings and scores)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # --- MODEL: Logistic Regression ---
    # (Linear models work surprisingly well with high-dim embeddings)
    print("\nTraining Classifier (scores + reasoning)...")
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, C=0.1)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    # Results
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*50)
    print(f"FINAL RESULT (Scores + Reasoning Text)")
    print("="*50)
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples')
    plt.title(f"Confusion Matrix (w/ Text Embeddings)\nF1: {f1:.3f}")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(OUTPUT_DIR / "embedding_confusion_matrix.png")
    print(f"Saved plot to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

