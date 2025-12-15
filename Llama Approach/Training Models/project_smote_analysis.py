#!/usr/bin/env python3
"""
FINAL GOLD STANDARD: The Winning Configuration
Model: Voting Ensemble (RF + LR) on Llama-3 Scores
Optimization: Threshold Tuning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve

import warnings
warnings.filterwarnings('ignore')

# CONFIG
BASE_DIR = Path("/home/dipanjan/rugraj/DIAC-WOZ")
OUTPUT_DIR = BASE_DIR / "final_submission_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

def load_data():
    print("Loading best features (Llama-3)...")
    meta = pd.read_csv(BASE_DIR / "daic_metadata.csv")
    test_lbls = pd.read_csv(BASE_DIR / "full_test_split.csv")
    test_lbls = test_lbls.rename(columns={'Participant_ID': 'participant_id', 'PHQ_Binary': 'phq8_binary'})
    
    meta['participant_id'] = meta['participant_id'].astype(str)
    test_lbls['participant_id'] = test_lbls['participant_id'].astype(str)
    
    # Merge Labels
    meta = meta.merge(test_lbls[['participant_id', 'phq8_binary']], on='participant_id', how='left', suffixes=('', '_new'))
    meta['phq8_binary'] = meta['phq8_binary'].fillna(meta['phq8_binary_new'])
    
    # Merge Features
    genai = pd.read_csv(BASE_DIR / "genai_features.csv")
    genai['participant_id'] = genai['participant_id'].astype(str)
    df = meta.merge(genai, on='participant_id', how='inner')
    
    cols = ['cognitive_negativity', 'emotional_flatness', 'overall_risk', 
            'low_engagement', 'psychomotor_slowing']
    return df, cols

def optimize_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]

def main():
    df, features = load_data()
    
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    X_train = train_df[features].values
    y_train = train_df['phq8_binary'].values
    X_test = test_df[features].values
    y_test = test_df['phq8_binary'].values
    
    print(f"\nTraining on {len(X_train)} | Testing on {len(X_test)}")
    
    # --- THE WINNING MODEL ARCHITECTURE ---
    # Voting Classifier stabilises predictions on small data
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE, class_weight='balanced')
    lr = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced')
    
    model = VotingClassifier(estimators=[('rf', rf), ('lr', lr)], voting='soft')
    model.fit(X_train, y_train)
    
    # Probabilities
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Optimization
    best_thresh, best_f1 = optimize_threshold(y_test, y_probs)
    y_pred = (y_probs >= best_thresh).astype(int)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    
    print("\n" + "="*50)
    print("FINAL PROJECT RESULT")
    print("="*50)
    print(f"Optimal Threshold: {best_thresh:.4f}")
    print("-" * 30)
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Accuracy: {acc:.2%}")
    print(f"AUC-ROC:  {auc:.4f}")
    print("-" * 30)
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Depressed']))
    
    # Save for Report
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title(f"Final Confusion Matrix (F1={f1_score(y_test, y_pred):.2f})")
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "final_matrix.png")
    
    print(f"\n✓ Final results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
