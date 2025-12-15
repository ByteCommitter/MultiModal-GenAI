#!/usr/bin/env python3
"""
DAIC-WOZ Dataset Validation and Setup Script
Validates dataset structure, loads metadata, and creates unified data splits.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
from tqdm import tqdm
import warnings
import sys

# Suppress pandas warnings for cleaner output
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path("/home/dipanjan/rugraj/DIAC-WOZ")
TRAIN_METADATA = BASE_DIR / "train_split_Depression_AVEC2017.csv"
DEV_METADATA = BASE_DIR / "dev_split_Depression_AVEC2017.csv"
TEST_METADATA = BASE_DIR / "full_test_split.csv"

REQUIRED_FILES = [
    "TRANSCRIPT.csv",
    "COVAREP.csv",
    "CLNF_AUs.txt",
    "CLNF_gaze.txt",
    "CLNF_pose.txt"
]

def print_header(text: str):
    """Print formatted section header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def scan_participant_folders() -> pd.DataFrame:
    """
    Scan dataset directory for all participant folders and validate file existence.
    
    Returns:
        DataFrame with columns: participant_id, folder_exists, transcript_exists,
        covarep_exists, au_exists, gaze_exists, pose_exists, all_files_present
    """
    print_header("STEP 1: Scanning Participant Folders")
    
    if not BASE_DIR.exists():
        print(f"❌ ERROR: Base directory not found: {BASE_DIR}")
        sys.exit(1)
    
    # Find all participant folders (XXX_P format)
    participant_folders = sorted([d for d in BASE_DIR.iterdir() 
                                 if d.is_dir() and d.name.endswith('_P')])
    
    print(f"Found {len(participant_folders)} participant folders")
    
    inventory = []
    
    for folder in tqdm(participant_folders, desc="Validating folders"):
        # Extract participant ID (e.g., "300_P" -> 300)
        try:
            participant_id = int(folder.name.replace('_P', ''))
        except ValueError:
            print(f"⚠️  Warning: Could not parse participant ID from folder: {folder.name}")
            continue
        
        # Check for each required file
        file_status = {
            'participant_id': participant_id,
            'folder_exists': True,
            'folder_path': str(folder)
        }
        
        # Check files with ID prefix (e.g., 300_TRANSCRIPT.csv)
        prefix = str(participant_id)
        file_status['transcript_exists'] = (folder / f"{prefix}_TRANSCRIPT.csv").exists()
        file_status['covarep_exists'] = (folder / f"{prefix}_COVAREP.csv").exists()
        file_status['au_exists'] = (folder / f"{prefix}_CLNF_AUs.txt").exists()
        file_status['gaze_exists'] = (folder / f"{prefix}_CLNF_gaze.txt").exists()
        file_status['pose_exists'] = (folder / f"{prefix}_CLNF_pose.txt").exists()
        
        # Check if all files present
        file_checks = [file_status[f'{f.split(".")[0].split("_")[-1].lower()}_exists'] 
                      for f in REQUIRED_FILES]
        file_status['all_files_present'] = all(file_checks)
        
        inventory.append(file_status)
    
    df_inventory = pd.DataFrame(inventory)
    
    # Summary statistics
    total = len(df_inventory)
    complete = df_inventory['all_files_present'].sum()
    print(f"\n✓ Total participants scanned: {total}")
    print(f"✓ Participants with complete data: {complete}")
    print(f"✓ Participants with missing files: {total - complete}")
    
    return df_inventory

def load_metadata_file(filepath: Path, split_name: str) -> pd.DataFrame:
    """
    Load and validate a metadata CSV file.
    
    Args:
        filepath: Path to metadata CSV
        split_name: Name of the split (train/dev/test)
    
    Returns:
        DataFrame with standardized columns
    """
    if not filepath.exists():
        print(f"⚠️  Warning: {split_name} metadata file not found: {filepath}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(filepath)
        print(f"  Loaded {split_name}: {len(df)} participants")
        
        # Validate expected columns
        expected_cols = ['Participant_ID', 'PHQ8_Binary']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            print(f"  ⚠️  Warning: Missing columns in {split_name}: {missing_cols}")
        
        # Add split identifier
        df['split'] = split_name
        
        return df
    
    except Exception as e:
        print(f"❌ Error loading {split_name} metadata: {e}")
        return pd.DataFrame()

def load_and_combine_metadata() -> pd.DataFrame:
    """
    Load all metadata files and combine into unified dataset.
    
    Returns:
        Combined DataFrame with standardized columns
    """
    print_header("STEP 2: Loading and Combining Metadata")
    
    # Load each split
    df_train = load_metadata_file(TRAIN_METADATA, 'train')
    df_dev = load_metadata_file(DEV_METADATA, 'dev')
    df_test = load_metadata_file(TEST_METADATA, 'test')
    
    # Combine all splits
    metadata_frames = [df for df in [df_train, df_dev, df_test] if not df.empty]
    
    if not metadata_frames:
        print("❌ ERROR: No metadata files could be loaded!")
        sys.exit(1)
    
    df_combined = pd.concat(metadata_frames, ignore_index=True)
    
    print(f"\n✓ Total participants in metadata: {len(df_combined)}")
    
    # Standardize column names
    column_mapping = {
        'Participant_ID': 'participant_id',
        'PHQ8_Binary': 'phq8_binary',
        'PHQ8_Score': 'phq8_score',
        'Gender': 'gender'
    }
    
    df_combined.rename(columns=column_mapping, inplace=True)
    
    # Handle missing PHQ8_Score values (empty cells -> NaN)
    if 'phq8_score' in df_combined.columns:
        df_combined['phq8_score'] = pd.to_numeric(df_combined['phq8_score'], errors='coerce')
        missing_scores = df_combined['phq8_score'].isna().sum()
        if missing_scores > 0:
            print(f"  ⚠️  Found {missing_scores} participants with missing PHQ8_Score")
    
    # Ensure participant_id is integer
    df_combined['participant_id'] = df_combined['participant_id'].astype(int)
    
    return df_combined

def merge_inventory_and_metadata(df_inventory: pd.DataFrame, 
                                 df_metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Merge inventory data with metadata and validate completeness.
    
    Args:
        df_inventory: DataFrame from folder scanning
        df_metadata: DataFrame from metadata files
    
    Returns:
        Merged DataFrame with complete information
    """
    print_header("STEP 3: Merging Inventory and Metadata")
    
    # Merge on participant_id
    df_merged = pd.merge(
        df_metadata,
        df_inventory,
        on='participant_id',
        how='outer',
        indicator=True
    )
    
    # Check for participants only in metadata
    only_metadata = df_merged[df_merged['_merge'] == 'left_only']
    if len(only_metadata) > 0:
        print(f"  ⚠️  {len(only_metadata)} participants in metadata but no folder found:")
        print(f"      IDs: {sorted(only_metadata['participant_id'].tolist())}")
    
    # Check for participants only in folders
    only_folders = df_merged[df_merged['_merge'] == 'right_only']
    if len(only_folders) > 0:
        print(f"  ⚠️  {len(only_folders)} participant folders with no metadata:")
        print(f"      IDs: {sorted(only_folders['participant_id'].tolist())}")
    
    # Keep only participants with both metadata and folders
    df_merged = df_merged[df_merged['_merge'] == 'both'].copy()
    df_merged.drop('_merge', axis=1, inplace=True)
    
    # Fill missing folder_exists with False for safety
    df_merged['all_files_present'] = df_merged['all_files_present'].fillna(False)
    
    print(f"\n✓ Successfully merged: {len(df_merged)} participants")
    
    return df_merged

def generate_summary_statistics(df_complete: pd.DataFrame, 
                               df_inventory: pd.DataFrame) -> Dict:
    """
    Generate comprehensive summary statistics.
    
    Args:
        df_complete: DataFrame with complete data
        df_inventory: DataFrame from folder scanning
    
    Returns:
        Dictionary with summary statistics
    """
    print_header("STEP 4: Summary Statistics")
    
    stats = {}
    
    # Overall statistics
    stats['total_folders'] = len(df_inventory)
    stats['complete_data'] = df_complete['all_files_present'].sum()
    stats['incomplete_data'] = len(df_complete) - stats['complete_data']
    
    print(f"\n📊 OVERALL STATISTICS:")
    print(f"  Total participant folders found: {stats['total_folders']}")
    print(f"  Participants with complete data: {stats['complete_data']}")
    print(f"  Participants with incomplete data: {stats['incomplete_data']}")
    
    # Filter to complete data only for split statistics
    df_complete_only = df_complete[df_complete['all_files_present']].copy()
    
    if len(df_complete_only) == 0:
        print("\n❌ WARNING: No participants with complete data!")
        return stats
    
    # Split-wise statistics
    print(f"\n📊 SPLIT-WISE BREAKDOWN:")
    for split in ['train', 'dev', 'test']:
        df_split = df_complete_only[df_complete_only['split'] == split]
        if len(df_split) > 0:
            n_total = len(df_split)
            n_depressed = (df_split['phq8_binary'] == 1).sum()
            n_healthy = (df_split['phq8_binary'] == 0).sum()
            pct_depressed = (n_depressed / n_total * 100) if n_total > 0 else 0
            
            stats[f'{split}_total'] = n_total
            stats[f'{split}_depressed'] = n_depressed
            stats[f'{split}_healthy'] = n_healthy
            
            print(f"  {split.upper()} split: {n_total} participants")
            print(f"    - Depressed (PHQ8=1): {n_depressed} ({pct_depressed:.1f}%)")
            print(f"    - Healthy (PHQ8=0): {n_healthy} ({100-pct_depressed:.1f}%)")
    
    # Overall depression prevalence
    total_complete = len(df_complete_only)
    total_depressed = (df_complete_only['phq8_binary'] == 1).sum()
    prevalence = (total_depressed / total_complete * 100) if total_complete > 0 else 0
    stats['overall_prevalence'] = prevalence
    
    print(f"\n  OVERALL DEPRESSION PREVALENCE: {prevalence:.1f}%")
    
    # Missing files analysis
    print(f"\n📊 MISSING FILES ANALYSIS:")
    file_cols = ['transcript_exists', 'covarep_exists', 'au_exists', 
                 'gaze_exists', 'pose_exists']
    
    for col in file_cols:
        if col in df_inventory.columns:
            missing = (~df_inventory[col]).sum()
            pct = (missing / len(df_inventory) * 100) if len(df_inventory) > 0 else 0
            file_name = col.replace('_exists', '').upper()
            print(f"  {file_name}: {missing} missing ({pct:.1f}%)")
            stats[f'missing_{file_name.lower()}'] = missing
    
    return stats

def save_outputs(df_inventory: pd.DataFrame, 
                df_complete: pd.DataFrame,
                stats: Dict):
    """
    Save all output files.
    
    Args:
        df_inventory: Complete inventory DataFrame
        df_complete: Merged metadata DataFrame
        stats: Summary statistics dictionary
    """
    print_header("STEP 5: Saving Outputs")
    
    # Save participants inventory (all participants)
    inventory_path = BASE_DIR / "participants_inventory.csv"
    df_inventory.to_csv(inventory_path, index=False)
    print(f"✓ Saved: {inventory_path}")
    
    # Save complete metadata (only participants with all files)
    df_complete_only = df_complete[df_complete['all_files_present']].copy()
    
    # Select and order columns for final output
    output_cols = ['participant_id', 'split', 'phq8_binary', 'phq8_score', 
                   'gender', 'folder_path', 'all_files_present']
    
    # Include only columns that exist
    output_cols = [col for col in output_cols if col in df_complete_only.columns]
    
    metadata_path = BASE_DIR / "daic_metadata_complete.csv"
    df_complete_only[output_cols].to_csv(metadata_path, index=False)
    print(f"✓ Saved: {metadata_path}")
    print(f"  ({len(df_complete_only)} participants with complete data)")
    
    # Save validation report
    report_path = BASE_DIR / "data_validation_report.txt"
    with open(report_path, 'w') as f:
        f.write("DAIC-WOZ DATASET VALIDATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OVERALL STATISTICS:\n")
        f.write(f"  Total participant folders: {stats.get('total_folders', 0)}\n")
        f.write(f"  Complete data: {stats.get('complete_data', 0)}\n")
        f.write(f"  Incomplete data: {stats.get('incomplete_data', 0)}\n\n")
        
        f.write("SPLIT-WISE BREAKDOWN:\n")
        for split in ['train', 'dev', 'test']:
            if f'{split}_total' in stats:
                f.write(f"  {split.upper()}: {stats[f'{split}_total']} participants\n")
                f.write(f"    Depressed: {stats[f'{split}_depressed']}\n")
                f.write(f"    Healthy: {stats[f'{split}_healthy']}\n")
        
        f.write(f"\nOVERALL DEPRESSION PREVALENCE: {stats.get('overall_prevalence', 0):.1f}%\n\n")
        
        f.write("MISSING FILES SUMMARY:\n")
        for key, value in stats.items():
            if key.startswith('missing_'):
                file_type = key.replace('missing_', '').upper()
                f.write(f"  {file_type}: {value} participants\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Generated by data_validation.py\n")
    
    print(f"✓ Saved: {report_path}")

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("  DAIC-WOZ DATASET VALIDATION AND SETUP")
    print("="*70)
    print(f"\nBase directory: {BASE_DIR}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Pandas version: {pd.__version__}")
    print(f"NumPy version: {np.__version__}")
    
    try:
        # Step 1: Scan participant folders
        df_inventory = scan_participant_folders()
        
        # Step 2: Load and combine metadata
        df_metadata = load_and_combine_metadata()
        
        # Step 3: Merge inventory and metadata
        df_complete = merge_inventory_and_metadata(df_inventory, df_metadata)
        
        # Step 4: Generate summary statistics
        stats = generate_summary_statistics(df_complete, df_inventory)
        
        # Step 5: Save outputs
        save_outputs(df_inventory, df_complete, stats)
        
        print_header("✅ VALIDATION COMPLETE")
        print("\nGenerated files:")
        print("  1. participants_inventory.csv - All participants (complete + incomplete)")
        print("  2. daic_metadata_complete.csv - Only participants with complete data")
        print("  3. data_validation_report.txt - Summary report")
        print("\nReady for model training! 🚀\n")
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
