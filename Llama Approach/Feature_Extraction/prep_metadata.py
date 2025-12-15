import argparse
import logging
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare unified DAIC-WOZ metadata CSV."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="/home/dipanjan/rugraj/DIAC-WOZ/",
        help="Absolute path to DAIC-WOZ dataset directory.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="/home/dipanjan/rugraj/DIAC-WOZ/daic_metadata.csv",
        help="Absolute path for output unified metadata CSV.",
    )
    return parser.parse_args()


def read_split_csv(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        logging.error("Split CSV not found: %s", path)
        raise FileNotFoundError(f"Split CSV not found: {path}")
    df = pd.read_csv(path)
    df = df.copy()
    df["split"] = split_name
    return df


def find_file_case_insensitive(folder: Path, substrings):
    """
    Search folder for a file whose name (lowercase) contains any of the provided substrings (also lowered).
    substrings can be a string or a list/tuple of strings (tries each).
    Returns first matching Path or None.
    """
    if isinstance(substrings, (list, tuple)):
        targets = [s.lower() for s in substrings]
    else:
        targets = [substrings.lower()]
    try:
        for p in folder.iterdir():
            name = p.name.lower()
            for t in targets:
                if t in name:
                    return p.resolve()
    except FileNotFoundError:
        return None
    return None


def prepare_metadata(dataset_dir: Path, output_csv: Path):
    # Expect these three files in dataset_dir
    train_csv = dataset_dir / "train_split_Depression_AVEC2017.csv"
    dev_csv = dataset_dir / "dev_split_Depression_AVEC2017.csv"
    test_csv = dataset_dir / "full_test_split.csv"

    # Read splits
    dfs = []
    for path, split in [(train_csv, "train"), (dev_csv, "dev"), (test_csv, "test")]:
        logging.info("Reading %s", path)
        dfs.append(read_split_csv(path, split))
    meta = pd.concat(dfs, ignore_index=True, sort=False)

    # Normalize column names if necessary
    # Expect: Participant_ID, PHQ8_Binary, PHQ8_Score, Gender, ...
    # Make sure Participant_ID exists
    if "Participant_ID" not in meta.columns:
        # try alternative names (defensive)
        possible = [c for c in meta.columns if "participant" in c.lower() and "id" in c.lower()]
        if possible:
            meta = meta.rename(columns={possible[0]: "Participant_ID"})
        else:
            logging.error("Participant_ID column not found in metadata CSVs.")
            raise KeyError("Participant_ID column not found.")

    # Clean PHQ8_Score: coerce to numeric (empty cells -> NaN)
    if "PHQ8_Score" in meta.columns:
        meta["PHQ8_Score"] = pd.to_numeric(meta["PHQ8_Score"], errors="coerce")
    else:
        meta["PHQ8_Score"] = pd.NA
        logging.warning("PHQ8_Score column missing; filling with NA.")

    # Ensure PHQ8_Binary exists; coerce to numeric 0/1 where possible
    if "PHQ8_Binary" in meta.columns:
        meta["PHQ8_Binary"] = pd.to_numeric(meta["PHQ8_Binary"], errors="coerce")
    else:
        meta["PHQ8_Binary"] = pd.NA
        logging.warning("PHQ8_Binary column missing; filling with NA.")

    # Gender
    if "Gender" not in meta.columns:
        # try find something similar
        poss = [c for c in meta.columns if "gender" in c.lower()]
        meta["Gender"] = meta[poss[0]] if poss else pd.NA

    # Build unified rows
    rows = []
    required_file_patterns = {
        "audio_path": ["audio.wav", "_audio.wav", "audio"],
        "covarep_path": ["covarep.csv", "covarep"],
        "transcript_path": ["transcript.csv", "transcript"],
        "au_path": ["clnf_aus.txt", "aus.txt", "_aus.txt", "clnf_aus"],
        "gaze_path": ["clnf_gaze.txt", "gaze.txt", "clnf_gaze"],
        "pose_path": ["clnf_pose.txt", "pose.txt", "clnf_pose"],
    }

    # Iterate participants
    logging.info("Inspecting %d participants", len(meta))
    for _, r in tqdm(meta.iterrows(), total=len(meta), desc="Preparing metadata"):
        pid_raw = r["Participant_ID"]
        # Participant IDs may be numeric or str; format folder as "{ID}_P"
        pid_str = str(int(pid_raw)) if pd.notna(pid_raw) and str(pid_raw).replace(".", "").isdigit() else str(pid_raw).strip()
        folder_name = f"{pid_str}_P"
        folder_path = dataset_dir / folder_name

        if not folder_path.exists():
            logging.warning("Folder missing for participant %s: expected %s", pid_str, folder_path)
            # still include row with None paths
            found = {k: None for k in required_file_patterns.keys()}
            files_exist = False
        else:
            found = {}
            for key, patterns in required_file_patterns.items():
                p = find_file_case_insensitive(folder_path, patterns)
                if p is None:
                    logging.warning("Missing %s for participant %s in folder %s", key, pid_str, folder_path)
                found[key] = str(p) if p is not None else None
            files_exist = all(v is not None for v in found.values())

        # Warn if PHQ8_Score missing
        if pd.isna(r.get("PHQ8_Score")):
            logging.warning("PHQ8_Score missing for participant %s (split=%s)", pid_str, r.get("split"))

        row = {
            "participant_id": pid_str,
            "split": r.get("split"),
            "phq8_score": r.get("PHQ8_Score"),
            "phq8_binary": r.get("PHQ8_Binary"),
            "gender": r.get("Gender"),
            "folder_path": str(folder_path.resolve()) if folder_path.exists() else None,
            "transcript_path": found.get("transcript_path"),
            "audio_path": found.get("audio_path"),
            "covarep_path": found.get("covarep_path"),
            "au_path": found.get("au_path"),
            "gaze_path": found.get("gaze_path"),
            "pose_path": found.get("pose_path"),
            "files_exist": files_exist,
        }
        rows.append(row)

    unified = pd.DataFrame(rows)

    # Summary stats
    total = len(unified)
    counts = unified["split"].value_counts().to_dict()
    valid_binary = pd.to_numeric(unified["phq8_binary"], errors="coerce")
    prevalence = valid_binary.dropna().mean() if not valid_binary.dropna().empty else pd.NA

    print("===== DAIC-WOZ Metadata Summary =====")
    print(f"Total participants processed: {total}")
    print(f"Per-split counts: {counts}")
    if prevalence is pd.NA:
        print("Depression prevalence (phq8_binary) could not be computed (no valid values).")
    else:
        print(f"Depression prevalence (phq8_binary==1): {prevalence:.3f} ({int(valid_binary.sum())} positives / {int(valid_binary.count())} with labels)")

    # Prevalence per split
    for split in unified["split"].unique():
        sub = unified[unified["split"] == split]
        vb = pd.to_numeric(sub["phq8_binary"], errors="coerce")
        if vb.dropna().empty:
            print(f"  {split}: no valid phq8_binary values")
        else:
            print(f"  {split}: prevalence={vb.dropna().mean():.3f} ({int(vb.sum())}/{int(vb.count())})")

    # Save CSV
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    unified.to_csv(output_csv, index=False)
    logging.info("Unified metadata saved to %s", output_csv)
    return unified


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()

    try:
        prepare_metadata(dataset_dir, output_csv)
    except Exception as e:
        logging.exception("Failed to prepare metadata: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
