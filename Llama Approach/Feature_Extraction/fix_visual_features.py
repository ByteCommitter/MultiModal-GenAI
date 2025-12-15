#!/usr/bin/env python3
"""
FIXED: Visual Features Extraction for DAIC-WOZ
Handles column names with leading spaces
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path("/home/dipanjan/rugraj/DIAC-WOZ")
METADATA_PATH = BASE_DIR / "daic_metadata.csv"
OUTPUT_PATH = BASE_DIR / "visual_features.csv"

def extract_visual_features(au_path, participant_id):
    """Extract visual features from Action Units (with space handling)"""
    try:
        # Read CLNF_AUs file
        df = pd.read_csv(au_path)
        
        # Strip whitespace from column names (THIS WAS THE BUG!)
        df.columns = df.columns.str.strip()
        
        # Get all AU_r columns (Action Unit intensities)
        au_cols = [col for col in df.columns if col.endswith('_r') and col.startswith('AU')]
        
        if len(au_cols) == 0:
            print(f"  ⚠️  No AU_r columns found for {participant_id}")
            return None
        
        features = {'participant_id': participant_id}
        
        # Calculate mean and std for each AU, skip if all zeros
        for au_col in au_cols:
            au_data = df[au_col]
            
            # Skip if all values are 0 (inactive AU)
            if au_data.sum() == 0:
                continue
            
            # Extract AU number (e.g., 'AU01_r' -> 'au01')
            au_name = au_col.lower().replace('_r', '')
            
            features[f'{au_name}_mean'] = au_data.mean()
            features[f'{au_name}_std'] = au_data.std()
        
        return features
    
    except Exception as e:
        print(f"  ⚠️  Error processing AUs {participant_id}: {str(e)}")
        return None


def main():
    """Re-extract visual features with fix"""
    print("\n" + "="*70)
    print("FIXED: Visual Features Re-extraction")
    print("="*70 + "\n")
    
    # Load metadata
    print("Loading metadata...")
    metadata_df = pd.read_csv(METADATA_PATH)
    
    # Filter to only participants with complete files
    complete_df = metadata_df[metadata_df['files_exist'] == True].copy()
    print(f"Found {len(complete_df)} participants with complete files\n")
    
    # Extract visual features
    visual_features = []
    success_count = 0
    
    print("Extracting visual features...")
    for _, row in tqdm(complete_df.iterrows(), total=len(complete_df), desc="Visual features"):
        result = extract_visual_features(row['au_path'], row['participant_id'])
        if result is not None:
            visual_features.append(result)
            success_count += 1
    
    # Create DataFrame
    visual_df = pd.DataFrame(visual_features)
    
    # Save
    visual_df.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n✓ Visual features: {success_count}/{len(complete_df)} successful")
    print(f"  Saved to: {OUTPUT_PATH}")
    print(f"  Features extracted: {visual_df.shape[1]-1} (active AUs × 2 statistics)")
    
    print("\nPreview:")
    print(visual_df.head())
    print(f"\nShape: {visual_df.shape}")
    print(f"Columns: {list(visual_df.columns[:10])}...")


if __name__ == "__main__":
    main()

