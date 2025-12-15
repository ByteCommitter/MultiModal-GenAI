import argparse
import logging
import re
from pathlib import Path
import sys

import pandas as pd
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract baseline features from DAIC-WOZ sessions."
    )
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default="/home/dipanjan/rugraj/DIAC-WOZ/daic_metadata.csv",
        help="Path to unified metadata CSV.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/dipanjan/rugraj/DIAC-WOZ/",
        help="Output directory for feature CSVs.",
    )
    return parser.parse_args()


def extract_text_features(transcript_path: Path) -> dict:
    """
    Extract text features from TRANSCRIPT.csv (Participant speaker only).
    Handles both comma- and tab-separated formats.
    Returns dict with keys: total_words, total_utterances, avg_utterance_length, 
                             first_person_ratio, response_rate
    """
    try:
        # Try tab-separated first, then comma-separated
        df = pd.read_csv(transcript_path, sep="\t")
        if len(df.columns) < 4:
            df = pd.read_csv(transcript_path, sep=",")
    except Exception as e:
        logging.warning("Failed to read transcript %s: %s", transcript_path, e)
        return None

    # Normalize column names to lowercase for robust matching
    df.columns = df.columns.str.lower().str.strip()
    
    # Find speaker and value columns (case-insensitive)
    speaker_col = None
    value_col = None
    for col in df.columns:
        if "speaker" in col:
            speaker_col = col
        if "value" in col:
            value_col = col
    
    if speaker_col is None or value_col is None:
        logging.warning("Could not find 'speaker' or 'value' columns in %s (found: %s)", 
                       transcript_path, list(df.columns))
        return None

    # Filter for Participant utterances
    participant_utts = df[df[speaker_col].str.lower() == "participant"]
    all_utts = df

    if participant_utts.empty:
        logging.warning("No participant utterances in %s", transcript_path)
        return None

    # Count words (simple split on whitespace)
    participant_utts = participant_utts.copy()
    participant_utts["word_count"] = participant_utts[value_col].fillna("").str.split().str.len()
    total_words = participant_utts["word_count"].sum()

    if total_words == 0:
        logging.warning("Zero words found in %s", transcript_path)
        return None

    total_utterances = len(participant_utts)
    avg_utterance_length = total_words / total_utterances if total_utterances > 0 else 0

    # First-person pronouns
    first_person_words = {"i", "me", "my", "mine", "myself"}
    fp_count = 0
    for text in participant_utts[value_col].fillna(""):
        words = text.lower().split()
        fp_count += sum(1 for w in words if w in first_person_words)
    first_person_ratio = fp_count / total_words if total_words > 0 else 0

    # Response rate: participant utterances / total turns
    total_turns = len(all_utts)
    response_rate = total_utterances / total_turns if total_turns > 0 else 0

    return {
        "total_words": total_words,
        "total_utterances": total_utterances,
        "avg_utterance_length": avg_utterance_length,
        "first_person_ratio": first_person_ratio,
        "response_rate": response_rate,
    }


def extract_acoustic_features(covarep_path: Path) -> dict:
    """
    Extract acoustic features from COVAREP.csv (no header, 74 columns).
    Compute mean and std for columns 11-36 (indices 11:37).
    Replace -Inf with NaN before computing statistics.
    Returns dict with keys: covarep_f11_mean, covarep_f11_std, ...
    """
    try:
        df = pd.read_csv(covarep_path, header=None)
    except Exception as e:
        logging.warning("Failed to read covarep %s: %s", covarep_path, e)
        return None

    if df.shape[1] < 37:
        logging.warning("COVAREP has fewer than 37 columns: %s", covarep_path)
        return None

    # Extract columns 11-36 (0-indexed)
    features = df.iloc[:, 11:37].copy()

    # Replace -Inf with NaN
    features = features.replace([np.inf, -np.inf], np.nan)

    # Convert to numeric
    features = features.apply(pd.to_numeric, errors="coerce")

    result = {}
    for col_idx in range(11, 37):
        feat_num = col_idx  # feature number (11-36)
        col_data = features.iloc[:, col_idx - 11]
        mean_val = col_data.mean()
        std_val = col_data.std()
        result[f"covarep_f{feat_num}_mean"] = mean_val
        result[f"covarep_f{feat_num}_std"] = std_val

    return result


def extract_visual_features(au_path: Path) -> dict:
    """
    Extract visual (AU) features from CLNF_AUs.txt (comma- or space-separated, with header).
    Robustly detect AU column names and compute mean/std for AU01..AU28.
    Skip columns that are all-zero or contain no valid numeric data.
    """
    df = None
    # try encodings and delimiters (prefer comma)
    for enc in ("utf-8", "latin-1"):
        for delim in (",", r"\s+"):
            try:
                with open(au_path, "r", encoding=enc, errors="replace") as fh:
                    df = pd.read_csv(fh, sep=delim, engine="python", comment="#")
                # if we read and there are multiple columns, accept
                if df is not None and df.shape[1] > 1:
                    break
            except Exception as e:
                logging.debug("Read AU file %s with enc=%s delim=%s failed: %s", au_path, enc, delim, e)
                df = None
        if df is not None and df.shape[1] > 1:
            break

    if df is None:
        logging.warning("Failed to read AUs %s (all encodings/delimiters tried)", au_path)
        return None

    # Clean column names: strip whitespace and trailing commas/semicolons/quotes
    cleaned_cols = []
    for c in df.columns:
        cstr = str(c).strip()
        # remove trailing punctuation often present when wrong delimiter picked
        cstr = re.sub(r'^[\'"]+|[\'",;]+$', '', cstr)
        cleaned_cols.append(cstr)
    df.columns = cleaned_cols

    # Map columns to AU numbers using regex - support many naming variants (AU01_r, AU_01_r, au01r, etc.)
    col_map = {}  # au_num -> cleaned column name
    for col in df.columns:
        # remove spaces/underscores/hyphens in matching
        col_norm = re.sub(r'[\s_\\-]', '', col, flags=re.I)
        m = re.search(r'(?i)au0*(\d{1,2})', col_norm)
        if m:
            try:
                au_num = int(m.group(1))
            except ValueError:
                continue
            if 1 <= au_num <= 28:
                # prefer columns that include 'r' or '_r' but accept others
                col_map.setdefault(au_num, col)

    if not col_map:
        logging.warning("No AU columns detected in %s. Columns found: %s", au_path, df.columns[:40].tolist())
        return None

    result = {}
    for au_num in sorted(col_map.keys()):
        col_name = col_map[au_num]
        col_data = pd.to_numeric(df[col_name], errors="coerce")

        # If column contains no valid numeric values -> skip
        if col_data.dropna().empty:
            logging.debug("AU %02d (%s) has no numeric values in %s; skipping", au_num, col_name, au_path)
            continue

        # Skip only if all non-NaN values are zero (explicit requirement)
        non_na = col_data.dropna()
        if non_na.eq(0).all():
            logging.debug("AU %02d (%s) is all-zero in %s; skipping", au_num, col_name, au_path)
            continue

        mean_val = non_na.mean()
        std_val = non_na.std()
        result[f"au{au_num:02d}_mean"] = mean_val
        result[f"au{au_num:02d}_std"] = std_val

    if not result:
        logging.warning("No visual AU features extracted from %s after filtering (detected columns: %s)", au_path, list(col_map.values())[:10])
        return None

    return result


def extract_all_features(metadata_csv: Path, output_dir: Path):
    """
    Main function: load metadata, extract features for each participant.
    """
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    try:
        meta = pd.read_csv(metadata_csv)
    except Exception as e:
        logging.error("Failed to load metadata %s: %s", metadata_csv, e)
        raise

    text_rows = []
    acoustic_rows = []
    visual_rows = []

    # Extract text features
    logging.info("Extracting text features...")
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Text features"):
        pid = row["participant_id"]
        transcript_path = row["transcript_path"]

        if pd.isna(transcript_path) or not Path(transcript_path).exists():
            logging.warning("Transcript missing for participant %s", pid)
            continue

        feat = extract_text_features(Path(transcript_path))
        if feat is not None:
            feat["participant_id"] = pid
            text_rows.append(feat)

    text_df = pd.DataFrame(text_rows)
    if not text_df.empty:
        text_df = text_df[["participant_id"] + [c for c in text_df.columns if c != "participant_id"]]
        text_df.to_csv(output_dir / "text_features.csv", index=False)
        logging.info("Saved text features: %d participants", len(text_df))
    else:
        logging.warning("No text features extracted.")

    # Extract acoustic features
    logging.info("Extracting acoustic features...")
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Acoustic features"):
        pid = row["participant_id"]
        covarep_path = row["covarep_path"]

        if pd.isna(covarep_path) or not Path(covarep_path).exists():
            logging.warning("COVAREP missing for participant %s", pid)
            continue

        feat = extract_acoustic_features(Path(covarep_path))
        if feat is not None:
            feat["participant_id"] = pid
            acoustic_rows.append(feat)

    acoustic_df = pd.DataFrame(acoustic_rows)
    if not acoustic_df.empty:
        acoustic_df = acoustic_df[["participant_id"] + [c for c in acoustic_df.columns if c != "participant_id"]]
        acoustic_df.to_csv(output_dir / "acoustic_features.csv", index=False)
        logging.info("Saved acoustic features: %d participants", len(acoustic_df))
    else:
        logging.warning("No acoustic features extracted.")

    # Extract visual features
    logging.info("Extracting visual features...")
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Visual features"):
        pid = row["participant_id"]
        au_path = row["au_path"]

        if pd.isna(au_path) or not Path(au_path).exists():
            logging.warning("AU file missing for participant %s", pid)
            continue

        feat = extract_visual_features(Path(au_path))
        if feat is not None:
            feat["participant_id"] = pid
            visual_rows.append(feat)

    visual_df = pd.DataFrame(visual_rows)
    if not visual_df.empty:
        visual_df = visual_df[["participant_id"] + [c for c in visual_df.columns if c != "participant_id"]]
        visual_df.to_csv(output_dir / "visual_features.csv", index=False)
        logging.info("Saved visual features: %d participants", len(visual_df))
    else:
        logging.warning("No visual features extracted.")

    # Summary
    print("\n===== Feature Extraction Summary =====")
    print(f"Text features: {len(text_df)}/{len(meta)} sessions")
    print(f"Acoustic features: {len(acoustic_df)}/{len(meta)} sessions")
    print(f"Visual features: {len(visual_df)}/{len(meta)} sessions")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    metadata_csv = Path(args.metadata_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        extract_all_features(metadata_csv, output_dir)
    except Exception as e:
        logging.exception("Feature extraction failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
