#!/usr/bin/env python3
"""
DAIC-WOZ Feature Extraction Script
Extracts text, acoustic, and visual features from pre-existing files
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
OUTPUT_DIR = BASE_DIR

# Feature extraction functions
def extract_text_features(transcript_path, participant_id):
    """Extract linguistic features from transcript"""
    try:
        # Read transcript (tab-separated with header)
        df = pd.read_csv(transcript_path, sep='\t')
        
        # Filter only Participant utterances
        participant_df = df[df['speaker'] == 'Participant'].copy()
        
        if len(participant_df) == 0:
            return None
        
        # Clean and tokenize utterances
        participant_df['words'] = participant_df['value'].str.lower().str.split()
        participant_df['word_count'] = participant_df['words'].apply(len)
        
        # Calculate features
        total_words = participant_df['word_count'].sum()
        total_utterances = len(participant_df)
        avg_utterance_length = total_words / total_utterances if total_utterances > 0 else 0
        
        # First-person pronouns
        first_person_pronouns = {'i', 'me', 'my', 'mine', 'myself'}
        all_words = [word for words in participant_df['words'] for word in words]
        first_person_count = sum(1 for word in all_words if word in first_person_pronouns)
        first_person_ratio = first_person_count / total_words if total_words > 0 else 0
        
        # Response rate (participant turns / total turns)
        total_turns = len(df)
        response_rate = total_utterances / total_turns if total_turns > 0 else 0
        
        return {
            'participant_id': participant_id,
            'total_words': total_words,
            'total_utterances': total_utterances,
            'avg_utterance_length': avg_utterance_length,
            'first_person_ratio': first_person_ratio,
            'response_rate': response_rate
        }
    
    except Exception as e:
        print(f"  ⚠️  Error processing transcript {participant_id}: {str(e)}")
        return None


def extract_acoustic_features(covarep_path, participant_id):
    """Extract acoustic features from COVAREP file"""
    try:
        # Read COVAREP (NO header, comma-separated)
        df = pd.read_csv(covarep_path, header=None)
        
        if df.shape[1] != 74:
            print(f"  ⚠️  Unexpected column count for {participant_id}: {df.shape[1]}")
            return None
        
        # Extract columns 11-36 (0-indexed: columns 11 to 36 inclusive = indices 11:37)
        acoustic_cols = df.iloc[:, 11:37]
        
        # Replace -Inf and Inf with NaN
        acoustic_cols = acoustic_cols.replace([np.inf, -np.inf], np.nan)
        
        # Calculate mean and std for each column
        features = {'participant_id': participant_id}
        
        for i, col_idx in enumerate(range(11, 37)):
            col_data = acoustic_cols.iloc[:, i]
            features[f'covarep_f{col_idx}_mean'] = col_data.mean()
            features[f'covarep_f{col_idx}_std'] = col_data.std()
        
        return features
    
    except Exception as e:
        print(f"  ⚠️  Error processing COVAREP {participant_id}: {str(e)}")
        return None


def extract_visual_features(au_path, participant_id):
    """Extract visual features from Action Units"""
    try:
        # Read CLNF_AUs (has header, comma-separated despite .txt extension)
        df = pd.read_csv(au_path)
        
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
    """Main feature extraction pipeline"""
    print("\n" + "="*70)
    print("DAIC-WOZ Feature Extraction")
    print("="*70 + "\n")
    
    # Load metadata
    print("Loading metadata...")
    metadata_df = pd.read_csv(METADATA_PATH)
    
    # Filter to only participants with complete files
    complete_df = metadata_df[metadata_df['files_exist'] == True].copy()
    print(f"Found {len(complete_df)} participants with complete files\n")
    
    # Initialize result lists
    text_features = []
    acoustic_features = []
    visual_features = []
    
    # Counters
    text_success = 0
    acoustic_success = 0
    visual_success = 0
    
    # ==================== EXTRACT TEXT FEATURES ====================
    print("="*70)
    print("EXTRACTING TEXT FEATURES")
    print("="*70)
    
    for _, row in tqdm(complete_df.iterrows(), total=len(complete_df), desc="Text features"):
        result = extract_text_features(row['transcript_path'], row['participant_id'])
        if result is not None:
            text_features.append(result)
            text_success += 1
    
    text_df = pd.DataFrame(text_features)
    text_output_path = OUTPUT_DIR / "text_features.csv"
    text_df.to_csv(text_output_path, index=False)
    print(f"\n✓ Text features: {text_success}/{len(complete_df)} successful")
    print(f"  Saved to: {text_output_path}")
    
    # ==================== EXTRACT ACOUSTIC FEATURES ====================
    print("\n" + "="*70)
    print("EXTRACTING ACOUSTIC FEATURES")
    print("="*70)
    
    for _, row in tqdm(complete_df.iterrows(), total=len(complete_df), desc="Acoustic features"):
        result = extract_acoustic_features(row['covarep_path'], row['participant_id'])
        if result is not None:
            acoustic_features.append(result)
            acoustic_success += 1
    
    acoustic_df = pd.DataFrame(acoustic_features)
    acoustic_output_path = OUTPUT_DIR / "acoustic_features.csv"
    acoustic_df.to_csv(acoustic_output_path, index=False)
    print(f"\n✓ Acoustic features: {acoustic_success}/{len(complete_df)} successful")
    print(f"  Saved to: {acoustic_output_path}")
    print(f"  Features extracted: {acoustic_df.shape[1]-1} (26 features × 2 statistics)")
    
    # ==================== EXTRACT VISUAL FEATURES ====================
    print("\n" + "="*70)
    print("EXTRACTING VISUAL FEATURES")
    print("="*70)
    
    for _, row in tqdm(complete_df.iterrows(), total=len(complete_df), desc="Visual features"):
        result = extract_visual_features(row['au_path'], row['participant_id'])
        if result is not None:
            visual_features.append(result)
            visual_success += 1
    
    visual_df = pd.DataFrame(visual_features)
    visual_output_path = OUTPUT_DIR / "visual_features.csv"
    visual_df.to_csv(visual_output_path, index=False)
    print(f"\n✓ Visual features: {visual_success}/{len(complete_df)} successful")
    print(f"  Saved to: {visual_output_path}")
    print(f"  Features extracted: {visual_df.shape[1]-1} (active AUs × 2 statistics)")
    
    # ==================== FINAL SUMMARY ====================
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    print(f"Total participants processed: {len(complete_df)}")
    print(f"  ✓ Text features:     {text_success}/{len(complete_df)} ({text_success/len(complete_df)*100:.1f}%)")
    print(f"  ✓ Acoustic features: {acoustic_success}/{len(complete_df)} ({acoustic_success/len(complete_df)*100:.1f}%)")
    print(f"  ✓ Visual features:   {visual_success}/{len(complete_df)} ({visual_success/len(complete_df)*100:.1f}%)")
    
    print(f"\nFeature dimensions:")
    print(f"  - Text:     {text_df.shape[0]} samples × {text_df.shape[1]-1} features")
    print(f"  - Acoustic: {acoustic_df.shape[0]} samples × {acoustic_df.shape[1]-1} features")
    print(f"  - Visual:   {visual_df.shape[0]} samples × {visual_df.shape[1]-1} features")
    
    print("\n" + "="*70)
    print("Feature extraction complete!")
    print("="*70 + "\n")
    
    # Quick preview
    print("TEXT FEATURES PREVIEW:")
    print(text_df.head())
    print(f"\nACOUSTIC FEATURES PREVIEW (first 5 columns):")
    print(acoustic_df.iloc[:, :6].head())
    print(f"\nVISUAL FEATURES PREVIEW (first 5 columns):")
    print(visual_df.iloc[:, :6].head())


if __name__ == "__main__":
    main()
