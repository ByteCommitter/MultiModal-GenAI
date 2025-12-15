import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

# CONFIG
BASE_DIR = Path("/home/dipanjan/rugraj/DIAC-WOZ")
OUTPUT_DIR = BASE_DIR / "diagnostics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("Loading data...")
    # Load Data
    meta = pd.read_csv(BASE_DIR / "daic_metadata.csv")
    test_lbls = pd.read_csv(BASE_DIR / "full_test_split.csv")
    genai = pd.read_csv(BASE_DIR / "genai_features.csv")
    
    # Merge
    test_lbls = test_lbls.rename(columns={'Participant_ID': 'participant_id', 'PHQ_Binary': 'phq8_binary'})
    meta['participant_id'] = meta['participant_id'].astype(str)
    test_lbls['participant_id'] = test_lbls['participant_id'].astype(str)
    genai['participant_id'] = genai['participant_id'].astype(str)
    
    meta = meta.merge(test_lbls[['participant_id', 'phq8_binary']], on='participant_id', how='left', suffixes=('', '_new'))
    meta['phq8_binary'] = meta['phq8_binary'].fillna(meta['phq8_binary_new'])
    df = meta.merge(genai, on='participant_id', how='inner')
    
    # Features
    features = ['cognitive_negativity', 'emotional_flatness', 'overall_risk', 'low_engagement', 'psychomotor_slowing']
    
    # Split
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    # CONVERT TO PURE NUMPY IMMEDIATELY
    # This prevents the Pandas/Matplotlib version conflict
    X_train = train_df[features].values
    y_train = train_df['phq8_binary'].values
    
    X_test = test_df[features].values
    y_test = test_df['phq8_binary'].values
    
    # Train Standard RF
    print("Training diagnostic model...")
    clf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Get Probabilities
    probs_test = clf.predict_proba(X_test)[:, 1]
    
    # --- VISUALIZATION (Pure Matplotlib) ---
    plt.figure(figsize=(10, 6))
    
    # Separate by class
    healthy_probs = probs_test[y_test == 0]
    depressed_probs = probs_test[y_test == 1]

    # Plot histograms
    plt.hist(healthy_probs, color='green', alpha=0.5, label='Healthy', bins=10)
    plt.hist(depressed_probs, color='red', alpha=0.5, label='Depressed', bins=10)
    
    plt.axvline(0.5, color='black', linestyle='--', label='Default (0.5)')
    
    plt.title("Probability Distribution: Healthy vs Depressed")
    plt.xlabel("Model Probability of Depression")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = OUTPUT_DIR / "probability_distribution.png"
    plt.savefig(save_path)
    print(f"Saved plot to {save_path}")
    
    # --- PRINT RAW NUMBERS ---
    print("\n" + "="*50)
    print("RAW PROBABILITY DUMP (Use these to pick threshold)")
    print("="*50)
    
    print("\n[DEPRESSED PATIENTS (1)] Scores:")
    # sort and print formatted
    d_sorted = np.sort(depressed_probs)
    print([f"{p:.3f}" for p in d_sorted])
    
    print("\n[HEALTHY PATIENTS (0)] Scores:")
    h_sorted = np.sort(healthy_probs)
    print([f"{p:.3f}" for p in h_sorted])
    
    # Calculate simple separation metrics
    med_dep = np.median(depressed_probs)
    med_healthy = np.median(healthy_probs)
    print(f"\nMedian Depressed Score: {med_dep:.3f}")
    print(f"Median Healthy Score:   {med_healthy:.3f}")
    
    suggested = (med_dep + med_healthy) / 2
    print(f"Suggested Threshold:    {suggested:.3f}")

if __name__ == "__main__":
    main()
