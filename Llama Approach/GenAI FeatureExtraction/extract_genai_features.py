#!/usr/bin/env python3
"""
DAIC-WOZ Dataset Preparation Script
Prepares metadata and verifies file existence for depression detection dataset
"""

import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import warnings

# Configuration
BASE_DIR = Path("/home/dipanjan/rugraj/DIAC-WOZ")
OUTPUT_PATH = BASE_DIR / "daic_metadata.csv"

# Metadata files
METADATA_FILES = {
    'train': BASE_DIR / "train_split_Depression_AVEC2017.csv",
    'dev': BASE_DIR / "dev_split_Depression_AVEC2017.csv",
    'test': BASE_DIR / "test_split_Depression_AVEC2017.csv"
}

# Required files for each participant
REQUIRED_FILES = {
    'audio': '_AUDIO.wav',
    'covarep': '_COVAREP.csv',
    'transcript': '_TRANSCRIPT.csv',
    'au': '_CLNF_AUs.txt',
    'gaze': '_CLNF_gaze.txt',
    'pose': '_CLNF_pose.txt'
}


def load_metadata(split_name, filepath):
    """Load metadata CSV with error handling."""
    try:
        df = pd.read_csv(filepath)
        # CRITICAL: Convert Participant_ID to integer (it's read as float by default)
        df['Participant_ID'] = df['Participant_ID'].astype(int)
        df['split'] = split_name
        print(f"✓ Loaded {split_name} split: {len(df)} participants")
        return df
    except FileNotFoundError:
        print(f"✗ ERROR: Metadata file not found: {filepath}")
        return None
    except Exception as e:
        print(f"✗ ERROR loading {split_name} split: {e}")
        return None


def check_participant_files(participant_id, base_dir):
    """
    Check if participant folder and all required files exist.
    
    Returns:
        tuple: (folder_path, file_paths_dict, all_exist_flag)
    """
    folder_name = f"{participant_id}_P"
    folder_path = base_dir / folder_name
    
    file_paths = {}
    files_exist = []
    
    if not folder_path.exists():
        # Return None paths if folder doesn't exist
        return str(folder_path), {k: None for k in REQUIRED_FILES.keys()}, False
    
    # Check each required file
    for file_key, file_suffix in REQUIRED_FILES.items():
        file_name = f"{participant_id}{file_suffix}"
        file_path = folder_path / file_name
        
        if file_path.exists():
            file_paths[file_key] = str(file_path)
            files_exist.append(True)
        else:
            file_paths[file_key] = None
            files_exist.append(False)
    
    all_exist = all(files_exist)
    return str(folder_path), file_paths, all_exist


def prepare_unified_metadata(metadata_dfs, base_dir):
    """
    Create unified metadata DataFrame with file paths and verification.
    
    Args:
        metadata_dfs: List of DataFrames from train/dev/test splits
        base_dir: Base directory path
    
    Returns:
        DataFrame with unified metadata
    """
    # Combine all metadata
    combined_df = pd.concat(metadata_dfs, ignore_index=True)
    print(f"\n{'='*60}")
    print(f"Total participants across all splits: {len(combined_df)}")
    print(f"{'='*60}\n")
    
    # Prepare output data
    unified_data = []
    missing_folders = []
    missing_files = []
    
    print("Verifying participant files...")
    for idx, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc="Processing"):
        participant_id = row['Participant_ID']
        
        # Check files
        folder_path, file_paths, all_exist = check_participant_files(participant_id, base_dir)
        
        # Track missing data
        if not Path(folder_path).exists():
            missing_folders.append(participant_id)
        elif not all_exist:
            missing_files.append(participant_id)
        
        # Handle missing PHQ8_Score (convert to None/NaN)
        phq8_score = row.get('PHQ8_Score')
        if pd.isna(phq8_score) or phq8_score == '':
            phq8_score = None
        
        # Create unified record
        record = {
            'participant_id': participant_id,
            'split': row['split'],
            'phq8_score': phq8_score,
            'phq8_binary': row['PHQ8_Binary'],
            'gender': row.get('Gender'),
            'folder_path': folder_path,
            'transcript_path': file_paths['transcript'],
            'audio_path': file_paths['audio'],
            'covarep_path': file_paths['covarep'],
            'au_path': file_paths['au'],
            'gaze_path': file_paths['gaze'],
            'pose_path': file_paths['pose'],
            'files_exist': all_exist
        }
        
        unified_data.append(record)
    
    # Create DataFrame
    unified_df = pd.DataFrame(unified_data)
    
    # Print warnings for missing data
    if missing_folders:
        print(f"\n⚠ WARNING: {len(missing_folders)} participant folders not found:")
        print(f"   IDs: {missing_folders[:10]}{'...' if len(missing_folders) > 10 else ''}")
    
    if missing_files:
        print(f"\n⚠ WARNING: {len(missing_files)} participants missing some files:")
        print(f"   IDs: {missing_files[:10]}{'...' if len(missing_files) > 10 else ''}")
    
    # Check for missing PHQ8 scores
    missing_scores = unified_df['phq8_score'].isna().sum()
    if missing_scores > 0:
        print(f"\n⚠ WARNING: {missing_scores} participants have missing PHQ8_Score values")
    
    return unified_df


def print_summary_statistics(df):
    """Print comprehensive summary statistics."""
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}\n")
    
    # Overall counts
    print(f"Total Participants: {len(df)}")
    print(f"Participants with all files: {df['files_exist'].sum()}")
    print(f"Participants with missing files: {(~df['files_exist']).sum()}")
    
    # Split distribution
    print(f"\n--- Split Distribution ---")
    split_counts = df['split'].value_counts()
    for split, count in split_counts.items():
        print(f"{split.capitalize()}: {count}")
    
    # Depression prevalence
    print(f"\n--- Depression Prevalence (PHQ8_Binary) ---")
    if 'phq8_binary' in df.columns:
        depression_counts = df['phq8_binary'].value_counts()
        total_with_labels = df['phq8_binary'].notna().sum()
        
        for label, count in depression_counts.items():
            label_str = "Depressed (1)" if label == 1 else "Not Depressed (0)"
            percentage = (count / total_with_labels * 100) if total_with_labels > 0 else 0
            print(f"{label_str}: {count} ({percentage:.1f}%)")
    
    # PHQ8 Score statistics
    print(f"\n--- PHQ8 Score Statistics ---")
    phq8_valid = df['phq8_score'].dropna()
    if len(phq8_valid) > 0:
        print(f"Valid scores: {len(phq8_valid)}/{len(df)}")
        print(f"Mean: {phq8_valid.mean():.2f}")
        print(f"Median: {phq8_valid.median():.2f}")
        print(f"Std Dev: {phq8_valid.std():.2f}")
        print(f"Range: [{phq8_valid.min():.0f}, {phq8_valid.max():.0f}]")
    else:
        print("No valid PHQ8 scores found")
    
    # Gender distribution
    if 'gender' in df.columns:
        print(f"\n--- Gender Distribution ---")
        gender_counts = df['gender'].value_counts()
        for gender, count in gender_counts.items():
            print(f"{gender}: {count}")
    
    print(f"\n{'='*60}\n")


def main():
    """Main execution function."""
    print(f"\n{'='*60}")
    print("DAIC-WOZ Dataset Preparation")
    print(f"{'='*60}\n")
    
    print(f"Base Directory: {BASE_DIR}")
    print(f"Output File: {OUTPUT_PATH}\n")
    
    # Check base directory exists
    if not BASE_DIR.exists():
        print(f"✗ ERROR: Base directory does not exist: {BASE_DIR}")
        return
    
    # Load all metadata files
    print("Loading metadata files...")
    metadata_dfs = []
    
    for split_name, filepath in METADATA_FILES.items():
        df = load_metadata(split_name, filepath)
        if df is not None:
            metadata_dfs.append(df)
    
    if not metadata_dfs:
        print("\n✗ ERROR: No metadata files could be loaded. Exiting.")
        return
    
    # Prepare unified metadata
    unified_df = prepare_unified_metadata(metadata_dfs, BASE_DIR)
    
    # Print summary statistics
    print_summary_statistics(unified_df)
    
    # Save to CSV
    try:
        unified_df.to_csv(OUTPUT_PATH, index=False)
        print(f"✓ Successfully saved metadata to: {OUTPUT_PATH}")
        print(f"  Rows: {len(unified_df)}")
        print(f"  Columns: {len(unified_df.columns)}")
    except Exception as e:
        print(f"✗ ERROR saving CSV: {e}")
        return
    
    print(f"\n{'='*60}")
    print("Data preparation complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
