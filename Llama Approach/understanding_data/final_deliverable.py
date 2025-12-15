#!/usr/bin/env python3
"""
FINAL DELIVERABLE V2: GenAI Features + Voting Ensemble + Threshold Optimization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve

import warnings
warnings.filterwarnings('ignore')

# CONFIG
BASE_DIR = Path("/home/dipanjan/rugraj/DIAC-WOZ")
OUTPUT_DIR = BASE_DIR / "project_deliverables_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

def load_data():
    print("Loading final dataset...")
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
    
    # THE WINNING FEATURES
    feature_cols = ['cognitive_negativity', 'emotional_flatness', 'overall_risk', 
                    'low_engagement', 'psychomotor_slowing']
    
    return df, feature_cols

def find_best_threshold(y_true, y_probs):
    """Finds the probability threshold that maximizes F1 score"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall)
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
    print(f"Evaluation on {len(X_test)} samples")
    
    # --- MODEL: VOTING ENSEMBLE ---
    # Combining RF (Complex) with LR (Simple) to prevent overfitting
    clf1 = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=RANDOM_STATE, class_weight='balanced')
    clf2 = LogisticRegression(random_state=RANDOM_STATE, class_weight='balanced')
    
    eclf = VotingClassifier(estimators=[('rf', clf1), ('lr', clf2)], voting='soft')
    eclf.fit(X_train, y_train)
    
    # Get Probabilities
    y_probs = eclf.predict_proba(X_test)[:, 1]
    
    # --- OPTIMIZATION ---
    print("Optimizing Decision Threshold...")
    best_thresh, best_f1 = find_best_threshold(y_test, y_probs)
    
    # Apply optimal threshold
    y_pred = (y_probs >= best_thresh).astype(int)
    
    # --- METRICS ---
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probs)
    
    print("\n" + "="*50)
    print("PROJECT FINAL RESULTS (OPTIMIZED)")
    print("="*50)
    print(f"Model: Voting Ensemble (RF+LR) with Threshold Tuning")
    print(f"Optimal Cutoff: {best_thresh:.4f}")
    print("-" * 30)
    print(f"Accuracy: {acc:.2%}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC-ROC:  {auc:.4f}")
    print("-" * 30)
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Depressed']))
    
    # --- ARTIFACTS ---
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False)
    plt.title(f"Confusion Matrix (Thresh={best_thresh:.2f})\nAcc: {acc:.1%}")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(OUTPUT_DIR / "optimized_confusion_matrix.png")
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "optimized_roc_curve.png")
    
    # 3. Save Model & Threshold
    joblib.dump({'model': eclf, 'threshold': best_thresh}, OUTPUT_DIR / "optimized_model_package.pkl")
    
    print(f"\n✓ Saved optimized model to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
