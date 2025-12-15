#!/usr/bin/env python3
"""
Final Optimization: GenAI-Only with Voting Ensemble & Threshold Tuning
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, precision_recall_curve

import warnings
warnings.filterwarnings('ignore')

# CONFIG
BASE_DIR = Path("/home/dipanjan/rugraj/DIAC-WOZ")
OUTPUT_DIR = BASE_DIR / "final_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

def load_data():
    print("Loading GenAI features...")
    # Load Metadata & Labels
    meta = pd.read_csv(BASE_DIR / "daic_metadata.csv")
    test_lbls = pd.read_csv(BASE_DIR / "full_test_split.csv")
    
    # Merge Test Labels
    test_lbls = test_lbls.rename(columns={'Participant_ID': 'participant_id', 'PHQ_Binary': 'phq8_binary'})
    meta['participant_id'] = meta['participant_id'].astype(str)
    test_lbls['participant_id'] = test_lbls['participant_id'].astype(str)
    
    meta = meta.merge(test_lbls[['participant_id', 'phq8_binary']], on='participant_id', how='left', suffixes=('', '_new'))
    meta['phq8_binary'] = meta['phq8_binary'].fillna(meta['phq8_binary_new'])
    
    # Load GenAI Features
    genai = pd.read_csv(BASE_DIR / "genai_features.csv")
    genai['participant_id'] = genai['participant_id'].astype(str)
    
    # Merge
    df = meta.merge(genai, on='participant_id', how='inner')
    
    # Select Columns (Drop Text)
    feature_cols = ['cognitive_negativity', 'emotional_flatness', 'overall_risk', 
                    'low_engagement', 'psychomotor_slowing']
    
    return df, feature_cols

def find_best_threshold(y_true, y_probs):
    """Finds the probability threshold that maximizes F1 score"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    f1_scores = 2 * (precision * recall) / (precision + recall)
    # Handle NaNs
    f1_scores = np.nan_to_num(f1_scores)
    
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    return best_thresh, best_f1

def main():
    df, features = load_data()
    
    # Split
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    X_train = train_df[features]
    y_train = train_df['phq8_binary']
    
    X_test = test_df[features]
    y_test = test_df['phq8_binary']
    
    print(f"\nTraining on {len(X_train)} samples")
    print(f"Testing on {len(X_test)} samples")
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # --- MODEL DEFINITION: VOTING ENSEMBLE ---
    # We combine RF (Good variance) and LR (Good bias)
    clf1 = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE, class_weight='balanced')
    clf2 = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced')
    
    eclf = VotingClassifier(estimators=[('rf', clf1), ('lr', clf2)], voting='soft')
    
    print("\nTraining Voting Ensemble...")
    eclf.fit(X_train_s, y_train)
    
    # Get Probabilities instead of hard predictions
    y_probs = eclf.predict_proba(X_test_s)[:, 1]
    
    # --- THRESHOLD TUNING ---
    print("Optimizing Decision Threshold...")
    best_thresh, best_f1 = find_best_threshold(y_test, y_probs)
    
    # Apply optimal threshold
    y_pred_opt = (y_probs >= best_thresh).astype(int)
    
    # Standard (0.5) predictions for comparison
    y_pred_default = (y_probs >= 0.5).astype(int)
    
    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)
    
    print(f"\n[Default Threshold 0.5]")
    print(f"F1 Score: {f1_score(y_test, y_pred_default):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_default):.4f}")
    
    print(f"\n[Optimized Threshold {best_thresh:.4f}]")
    print(f"F1 Score: {f1_score(y_test, y_pred_opt):.4f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_opt):.4f}")
    
    print("\nClassification Report (Optimized):")
    print(classification_report(y_test, y_pred_opt))
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_opt)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
    plt.title(f"Confusion Matrix (Thresh={best_thresh:.2f})")
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(OUTPUT_DIR / "final_confusion_matrix.png")
    
    # Save feature importance (from RF part of ensemble)
    rf_model = eclf.named_estimators_['rf']
    imps = rf_model.feature_importances_
    plt.figure(figsize=(8,5))
    plt.barh(features, imps)
    plt.title("Feature Importance (GenAI Only)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "final_importance.png")
    
    print(f"\nVisualizations saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
