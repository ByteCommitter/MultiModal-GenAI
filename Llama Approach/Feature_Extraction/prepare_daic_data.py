#!/usr/bin/env python3
"""
DAIC-WOZ Depression Dataset Preparation Script
Prepares unified metadata with file paths for all participants
"""

import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Configuration
BASE_DIR = Path("/home/dipanjan/rugraj/DIAC-WOZ")
OUTPUT_PATH = BASE_DIR / "daic_metadata.csv"

# Required files for each participant
REQUIRED_FILES = [
    "TRANSCRIPT.csv",
    "AUDIO.wav",
    "COVAREP.csv",
    "CLNF_AUs.txt",
    "CLNF_gaze.txt",
    "CLNF_pose.txt"
]

def load_metadata():
    """Load and combine train/dev/test metadata files"""
    print("Loading metadata files...")
    
    # Load train and dev (have full PHQ8 data)
    train_df = pd.read_csv(BASE_DIR / "train_split_Depression_AVEC2017.csv")
    train_df['split'] = 'train'
    
    dev_df = pd.read_csv(BASE_DIR / "dev_split_Depression_AVEC2017.csv")
    dev_df['split'] = 'dev'
    
    # Load test (only has participant_ID and Gender - note the lowercase 'p'!)
    test_df = pd.read_csv(BASE_DIR / "test_split_Depression_AVEC2017.csv")
    test_df['split'] = 'test'
    
    # Standardize column name (test uses 'participant_ID', others use 'Participant_ID')
    test_df.rename(columns={'participant_ID': 'Participant_ID'}, inplace=True)
    
    # Combine all splits
    combined_df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    
    print(f"✓ Loaded {len(train_df)} train, {len(dev_df)} dev, {len(test_df)} test participants")
    
    return combined_df

def check_participant_files(participant_id):
    """Check if all required files exist for a participant"""
    folder_path = BASE_DIR / f"{participant_id}_P"
    
    if not folder_path.exists():
        return None, {}
    
    file_paths = {}
    all_exist = True
    
    for file_type in REQUIRED_FILES:
        file_name = f"{participant_id}_{file_type}"
        file_path = folder_path / file_name
        
        if file_path.exists():
            file_paths[file_type] = str(file_path)
        else:
            file_paths[file_type] = None
            all_exist = False
    
    return all_exist, file_paths

def prepare_dataset():
    """Main function to prepare unified metadata"""
    print("\n" + "="*70)
    print("DAIC-WOZ Dataset Preparation")
    print("="*70 + "\n")
    
    # Load metadata
    metadata_df = load_metadata()
    
    # Prepare results list
    results = []
    missing_folders = []
    incomplete_participants = []
    
    print("\nChecking participant files...")
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing"):
        participant_id = row['Participant_ID']
        folder_path = BASE_DIR / f"{participant_id}_P"
        
        # Check files
        files_exist, file_paths = check_participant_files(participant_id)
        
        if files_exist is None:
            missing_folders.append(participant_id)
            files_exist = False
        elif not files_exist:
            incomplete_participants.append(participant_id)
        
        # Build result row
        result = {
            'participant_id': participant_id,
            'split': row['split'],
            'phq8_score': row.get('PHQ8_Score', None),
            'phq8_binary': row.get('PHQ8_Binary', None),
            'gender': row['Gender'],
            'folder_path': str(folder_path) if folder_path.exists() else None,
            'transcript_path': file_paths.get('TRANSCRIPT.csv'),
            'audio_path': file_paths.get('AUDIO.wav'),
            'covarep_path': file_paths.get('COVAREP.csv'),
            'au_path': file_paths.get('CLNF_AUs.txt'),
            'gaze_path': file_paths.get('CLNF_gaze.txt'),
            'pose_path': file_paths.get('CLNF_pose.txt'),
            'files_exist': files_exist
        }
        
        results.append(result)
    
    # Create final DataFrame
    final_df = pd.DataFrame(results)
    
    # Print warnings
    if missing_folders:
        print(f"\n⚠️  WARNING: {len(missing_folders)} participants have missing folders:")
        print(f"   {missing_folders[:10]}{'...' if len(missing_folders) > 10 else ''}")
    
    if incomplete_participants:
        print(f"\n⚠️  WARNING: {len(incomplete_participants)} participants have incomplete files:")
        print(f"   {incomplete_participants[:10]}{'...' if len(incomplete_participants) > 10 else ''}")
    
    # Print summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"\nTotal participants: {len(final_df)}")
    print(f"  - Train: {len(final_df[final_df['split'] == 'train'])}")
    print(f"  - Dev:   {len(final_df[final_df['split'] == 'dev'])}")
    print(f"  - Test:  {len(final_df[final_df['split'] == 'test'])}")
    
    print(f"\nFiles status:")
    print(f"  - Complete: {final_df['files_exist'].sum()}")
    print(f"  - Incomplete/Missing: {(~final_df['files_exist']).sum()}")
    
    # Depression statistics (only for train/dev with PHQ8 scores)
    train_dev_df = final_df[final_df['split'].isin(['train', 'dev'])].copy()
    if len(train_dev_df) > 0:
        print(f"\nDepression prevalence (train + dev only):")
        print(f"  - Not depressed (PHQ8_Binary=0): {(train_dev_df['phq8_binary'] == 0).sum()}")
        print(f"  - Depressed (PHQ8_Binary=1):     {(train_dev_df['phq8_binary'] == 1).sum()}")
        print(f"  - Mean PHQ8 score: {train_dev_df['phq8_score'].mean():.2f}")
        print(f"  - PHQ8 score range: {train_dev_df['phq8_score'].min():.0f} - {train_dev_df['phq8_score'].max():.0f}")
    
    print(f"\nGender distribution:")
    print(f"  - Female (0): {(final_df['gender'] == 0).sum()}")
    print(f"  - Male (1):   {(final_df['gender'] == 1).sum()}")
    
    print(f"\n⚠️  NOTE: Test split has NO PHQ8 labels (phq8_score and phq8_binary will be NaN)")
    
    # Save to CSV
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ Saved metadata to: {OUTPUT_PATH}")
    print("="*70 + "\n")
    
    return final_df

if __name__ == "__main__":
    df = prepare_dataset()
    
    # Quick verification
    print("Preview of saved data:")
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")
