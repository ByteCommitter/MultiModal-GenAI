#!/usr/bin/env python3
"""
Depression Classification V3: Feature Selection & Optimization
Goal: Combine GenAI features ONLY with the best Traditional features.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
from statsmodels.stats.contingency_tables import mcnemar
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path("/home/dipanjan/rugraj/DIAC-WOZ")
OUTPUT_DIR = BASE_DIR / "results_v3_optimized"
IMG_DIR = OUTPUT_DIR / "confusion_matrices"
IMP_DIR = OUTPUT_DIR / "feature_importance"
MODEL_DIR = OUTPUT_DIR / "best_models"

for d in [OUTPUT_DIR, IMG_DIR, IMP_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# ==========================================
# 1. DATA LOADING
# ==========================================
def load_and_merge_data():
    print("Loading datasets...")
    meta = pd.read_csv(BASE_DIR / "daic_metadata.csv")
    text = pd.read_csv(BASE_DIR / "text_features.csv")
    acoustic = pd.read_csv(BASE_DIR / "acoustic_features.csv")
    visual = pd.read_csv(BASE_DIR / "visual_features.csv")
    genai = pd.read_csv(BASE_DIR / "genai_features.csv")

    # Integrate Test Labels
    test_labels_path = BASE_DIR / "full_test_split.csv"
    if test_labels_path.exists():
        print(f"✓ Found external test labels.")
        test_lbls = pd.read_csv(test_labels_path)
        test_lbls = test_lbls.rename(columns={'Participant_ID': 'participant_id', 'PHQ_Binary': 'phq8_binary', 'PHQ8_Score': 'phq8_score'})
        test_lbls['participant_id'] = test_lbls['participant_id'].astype(str)
        meta['participant_id'] = meta['participant_id'].astype(str)
        meta = meta.merge(test_lbls[['participant_id', 'phq8_binary', 'phq8_score']], 
                         on='participant_id', how='left', suffixes=('', '_new'))
        meta['phq8_binary'] = meta['phq8_binary'].fillna(meta['phq8_binary_new'])
        meta.drop(columns=['phq8_binary_new', 'phq8_score_new'], inplace=True, errors='ignore')

    for df in [text, acoustic, visual, genai]:
        df['participant_id'] = df['participant_id'].astype(str)

    df = meta.merge(text, on='participant_id', how='left')
    df = df.merge(acoustic, on='participant_id', how='left')
    df = df.merge(visual, on='participant_id', how='left')
    df = df.merge(genai, on='participant_id', how='left')

    return df, text.columns, acoustic.columns, visual.columns, genai.columns

# ==========================================
# 2. FEATURE SELECTION
# ==========================================
def select_best_features(X_train, y_train, feature_names, top_n=15):
    """Uses Random Forest to pick the best traditional features"""
    print(f"   > Selecting top {top_n} traditional features from {len(feature_names)}...")
    
    selector = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    selector.fit(X_train, y_train)
    
    importances = selector.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    
    selected_feats = [feature_names[i] for i in indices]
    print(f"   > Top 5 Selected: {selected_feats[:5]}")
    return selected_feats

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
def main():
    df, t_cols, a_cols, v_cols, g_cols = load_and_merge_data()

    # Define Column Sets
    exclude = ['participant_id', 'split', 'phq8_binary', 'phq8_score', 'raw_response', 'llm_reasoning']
    def clean(cols): return [c for c in cols if c not in exclude]

    trad_feats = clean(t_cols) + clean(a_cols) + clean(v_cols)
    genai_feats = clean(g_cols)

    # Split Data
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    y_train = train_df['phq8_binary'].dropna()
    y_test = test_df['phq8_binary'].dropna()
    
    # Preprocessing for Selection
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_trad = pd.DataFrame(scaler.fit_transform(imputer.fit_transform(df.loc[y_train.index, trad_feats])), columns=trad_feats)
    
    # --- STEP 1: SMART SELECTION ---
    # Find the best Traditional features to help GenAI
    best_trad_feats = select_best_features(X_train_trad, y_train, trad_feats, top_n=15)
    
    configs = {
        "Baseline (All Trad)": trad_feats,
        "GenAI-Only": genai_feats,
        "Hybrid-Optimized": genai_feats + best_trad_feats  # <--- THE NEW WINNING CONFIG
    }

    results_list = []
    
    print("\n" + "="*60)
    print(f"Training on {len(y_train)} | Testing on {len(y_test)}")
    print("="*60)

    for config_name, feats in configs.items():
        print(f"\nRunning: {config_name} ({len(feats)} features)")
        
        X_train = df.loc[y_train.index, feats]
        X_test = df.loc[y_test.index, feats]
        
        # Pipeline
        X_train = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test = scaler.transform(imputer.transform(X_test))

        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=RANDOM_STATE),
            "XGBoost": xgb.XGBClassifier(n_estimators=100, max_depth=5, eval_metric='logloss', random_state=RANDOM_STATE, use_label_encoder=False)
        }

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            f1 = f1_score(y_test, y_pred, zero_division=0)
            acc = accuracy_score(y_test, y_pred)
            
            results_list.append({
                "config": config_name,
                "model": model_name,
                "f1": f1,
                "accuracy": acc,
                "recall": recall_score(y_test, y_pred)
            })

            # Feature Importance Plot (for Hybrid only)
            if config_name == "Hybrid-Optimized" and model_name == "RandomForest":
                plt.figure(figsize=(10, 6))
                imps = model.feature_importances_
                idxs = np.argsort(imps)[::-1]
                plt.title(f"Feature Importance: Hybrid Model")
                plt.barh(range(len(imps)), imps[idxs], align="center")
                plt.yticks(range(len(imps)), [feats[i] for i in idxs])
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(IMP_DIR / "hybrid_importance.png")
                plt.close()

    # Summary
    res_df = pd.DataFrame(results_list).sort_values(by='f1', ascending=False)
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(res_df.to_string())
    res_df.to_csv(OUTPUT_DIR / "final_results.csv", index=False)

if __name__ == "__main__":
    main()
