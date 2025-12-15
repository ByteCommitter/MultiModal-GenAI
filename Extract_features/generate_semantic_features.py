import json
import logging
import re
import time
from pathlib import Path
import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TRANSFORMERS_VERBOSITY"] = "warning"

DATA_DIR = Path("/home/dipanjan/rugraj/DIAC-WOZ")
MODEL_PATH = Path("~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf").expanduser()
OUTPUT_CSV = DATA_DIR / "genai_features.csv"
ERROR_LOG = DATA_DIR / "genai_extraction_errors.log"
CHECKPOINT_EVERY = 20
UNLOAD_AFTER = 10


def parse_args():
    parser = argparse.ArgumentParser(description="Generate semantic features with local LLM")
    parser.add_argument("--metadata_csv", type=str, default=str(DATA_DIR / "daic_metadata.csv"))
    parser.add_argument("--output_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--limit", type=int, default=0, help="Process only this many participants (0 = all)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--few_shot", action="store_true", help="Include few-shot example")
    return parser.parse_args()


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        filename=ERROR_LOG,
        filemode="a",
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().handlers = [h for h in logging.getLogger().handlers if not isinstance(h, logging.StreamHandler)]
    logging.getLogger().addHandler(console)


def load_model_and_tokenizer():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model folder not found: {MODEL_PATH}")
    logging.info("Loading model from %s", MODEL_PATH)

    model_dir = MODEL_PATH
    try:
        snapshots_dir = MODEL_PATH / "snapshots"
        if snapshots_dir.exists() and snapshots_dir.is_dir():
            subs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
            if subs:
                subs_sorted = sorted(subs, key=lambda p: p.stat().st_mtime, reverse=True)
                model_dir = subs_sorted[0]
                logging.info("Using snapshot model dir: %s", model_dir)
    except Exception:
        logging.debug("No snapshots dir or failed to inspect snapshots", exc_info=True)

    try:
        def has_tokenizer_files(p):
            for fname in ("tokenizer.model", "tokenizer.json", "tokenizer_config.json", "spiece.model"):
                if (p / fname).exists():
                    return True
            return False

        if not has_tokenizer_files(model_dir):
            found = None
            for p in MODEL_PATH.iterdir():
                if p.is_dir() and has_tokenizer_files(p):
                    found = p
                    break
            if found:
                model_dir = found
                logging.info("Switched to model subdir with tokenizer files: %s", model_dir)
    except Exception:
        logging.debug("Tokenizer file discovery failed", exc_info=True)

    try:
        logging.info("Final model_dir used for loading: %s", model_dir)
        logging.info("Model dir top-level contents: %s", [p.name for p in model_dir.iterdir()] if model_dir.is_dir() else [])
    except Exception:
        logging.debug("Failed to list model_dir contents", exc_info=True)

    primary_exc = None
    fallback_exc = None

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        return tokenizer, model
    except Exception as e:
        primary_exc = e
        logging.exception("Primary tokenizer/model load failed: %s", e)

    try:
        logging.info("Attempting fallback load without trust_remote_code...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()
        logging.info("Fallback load succeeded.")
        return tokenizer, model
    except Exception as e:
        fallback_exc = e
        logging.exception("Fallback tokenizer/model load failed: %s", e)

    msg_lines = [
        f"Failed to load tokenizer/model from {MODEL_PATH}.",
        f"Resolved model_dir: {model_dir}",
        "Primary exception: " + (repr(primary_exc) if primary_exc is not None else "None"),
        "Fallback exception: " + (repr(fallback_exc) if fallback_exc is not None else "None"),
    ]
    full_msg = "\n".join(msg_lines)
    logging.error(full_msg)
    raise RuntimeError(full_msg)


def read_participant_utterances(transcript_path: Path, max_chars=1500):
    """
    Read transcript (tab/comma) and return concatenated participant utterances (first max_chars).
    """
    try:
        df = pd.read_csv(transcript_path, sep="\t")
        if df.shape[1] < 3:
            df = pd.read_csv(transcript_path, sep=",")
    except Exception:
        try:
            df = pd.read_csv(transcript_path, sep=",", encoding="latin-1")
        except Exception as e:
            logging.warning("Failed to read transcript %s: %s", transcript_path, e)
            return ""
    # normalize columns
    df.columns = df.columns.str.lower().str.strip()
    # find speaker/value columns
    speaker_col = None
    value_col = None
    for c in df.columns:
        if "speaker" in c:
            speaker_col = c
        if "value" in c:
            value_col = c
    if speaker_col is None or value_col is None:
        return ""
    participant_utts = df[df[speaker_col].str.lower() == "participant"][value_col].fillna("").astype(str)
    text = " ".join(participant_utts.tolist())
    return text[:max_chars]


def extract_json_from_text(text: str):
    """
    Find first JSON object in text and return parsed dict.
    """
    # greedy braces match using stack approach via regex scanning
    start = None
    for m in re.finditer(r"\{", text):
        start = m.start()
        # try to find matching closing brace
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
    # fallback to regex simple {...}
    m = re.search(r"(\{.*\})", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except Exception:
        return None


def make_prompt(first_utt: str, covarep_vals: dict, visual_vals: dict):
    # safe defaults
    f11 = covarep_vals.get("covarep_f11_mean", 0.0)
    f15 = covarep_vals.get("covarep_f15_mean", 0.0)
    f20 = covarep_vals.get("covarep_f20_mean", 0.0)
    au12 = visual_vals.get("au12_mean", 0.0)
    au_std_cols = [k for k in visual_vals.keys() if k.endswith("_std")]
    au_mean_std = 0.0
    if au_std_cols:
        vals = [visual_vals[c] for c in au_std_cols if visual_vals.get(c) is not None]
        if vals:
            au_mean_std = float(sum(vals) / len(vals))
    first_utt = first_utt.replace("\n", " ").strip()
    if len(first_utt) > 300:
        first_utt = first_utt[:300]
    
    # Ultra-minimal: force 5 comma-separated integers only
    prompt = (
        f"Rate depression 0-10:\n"
        f"{first_utt}\n"
        f"Pitch:{f11:.0f} Voice:{f15:.0f} Energy:{f20:.0f} Smile:{au12:.0f} Expr:{au_mean_std:.0f}\n"
        "Output 5 integers separated by commas:\n"
    )
    return prompt


def generate_response(tokenizer, model, prompt, max_new_tokens=60, temperature=0.0):
    """
    Generate text. If temperature==0, use greedy (do_sample=False). Returns decoded string.
    """
    try:
        # Add max_length to avoid truncation warning; use 2000 as safe upper bound
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000)
        try:
            param = next(model.parameters())
            device = param.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass
        do_sample = False if temperature == 0.0 else True
        with torch.no_grad():
            # Add top_p for sampling methods (ignored in greedy but suppresses warning)
            out = model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=do_sample, 
                temperature=temperature,
                top_p=0.9 if do_sample else 1.0
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        # strip prompt echo if present
        if prompt.strip() in text:
            # remove everything up to and including the prompt
            idx = text.find(prompt.strip())
            if idx != -1:
                text = text[idx + len(prompt.strip()):].strip()
        return text.strip()
    except Exception as e:
        logging.exception("Generation failed: %s", e)
        return ""


def get_json_response(tokenizer, model, prompt):
    """
    Generate response and extract 5 comma-separated integers (0-10).
    Returns (parsed_dict, raw_text, attempts_list)
    """
    attempts = []
    raw = generate_response(tokenizer, model, prompt, max_new_tokens=60, temperature=0.0)
    attempts.append(("greedy", raw))
    
    # Try to parse as comma-separated integers first
    # Extract: "X, Y, Z, A, B" or "X,Y,Z,A,B"
    parts = re.split(r'[,\s]+', raw.strip())
    nums = []
    for p in parts:
        try:
            n = int(float(p))
            n = max(0, min(10, n))
            nums.append(n)
            if len(nums) == 5:
                break 
        except ValueError:
            continue
    
    if len(nums) == 5:
        result = {
            "cognitive_negativity": nums[0],
            "emotional_flatness": nums[1],
            "low_engagement": nums[2],
            "psychomotor_slowing": nums[3],
            "overall_risk": nums[4],
        }
        logging.debug("Parsed comma-separated: %s", raw[:80])
        return result, raw, attempts
    
    # Fallback: try JSON parsing
    parsed = extract_json_from_text(raw)
    if parsed:
        return parsed, raw, attempts
    
    return None, raw, attempts


def get_feature_row(participant_id, tokenizer, model, acoustic_df, visual_df, transcript_path):
    row = {
        "participant_id": participant_id,
        "cognitive_negativity": None,
        "emotional_flatness": None,
        "low_engagement": None,
        "psychomotor_slowing": None,
        "overall_risk": None,
        "raw_response": None,
    }
    try:
        # transcript text
        utts = read_participant_utterances(Path(transcript_path))
        # acoustic features (single-row df)
        covarep_vals = {}
        if participant_id in acoustic_df.index:
            covarep_vals = acoustic_df.loc[participant_id].to_dict()
        # visual features
        visual_vals = {}
        if participant_id in visual_df.index:
            visual_vals = visual_df.loc[participant_id].to_dict()
        prompt = make_prompt(utts, covarep_vals, visual_vals)

        parsed, raw, attempts = get_json_response(tokenizer, model, prompt)
        row["raw_response"] = raw
        if parsed:
            # ensure keys present and coerce ints
            for k in ["cognitive_negativity", "emotional_flatness", "low_engagement", "psychomotor_slowing", "overall_risk"]:
                if k in parsed:
                    try:
                        v = int(float(parsed[k]))
                        v = max(0, min(10, v))
                    except Exception:
                        v = None
                    row[k] = v
                else:
                    row[k] = None
        else:
            # log attempts for debugging
            logging.warning("Parsing failed for %s after attempts: %s. Last raw: %s", participant_id, [a[0] for a in attempts], raw[:200])
            # leave row values as None (raw_response saved)
    except Exception as e:
        logging.exception("Failed processing participant %s: %s", participant_id, e)
    return row


def main():
    args = parse_args()
    # configure logging early
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)

    logging.info("Starting semantic feature extraction")
    logging.info("Args: %s", vars(args))

    # load metadata and feature CSVs
    try:
        meta = pd.read_csv(Path(args.metadata_csv))
    except Exception as e:
        logging.exception("Failed to read metadata CSV: %s", e)
        return

    # load acoustic and visual CSVs; index by participant_id (handle missing files)
    try:
        acoustic_df = pd.read_csv(DATA_DIR / "acoustic_features.csv").set_index("participant_id")
    except Exception as e:
        logging.warning("Could not load acoustic_features.csv: %s. Continuing with empty acoustic data.", e)
        acoustic_df = pd.DataFrame().set_index(pd.Index([], name="participant_id"))

    try:
        visual_df = pd.read_csv(DATA_DIR / "visual_features.csv").set_index("participant_id")
    except Exception as e:
        logging.warning("Could not load visual_features.csv: %s. Continuing with empty visual data.", e)
        visual_df = pd.DataFrame().set_index(pd.Index([], name="participant_id"))

    participants = meta["participant_id"].tolist()
    if args.limit and args.limit > 0:
        participants = participants[: args.limit]
        logging.info("Processing limited to first %d participants", args.limit)

    results = []

    # load model initially (handle load errors gracefully)
    try:
        tokenizer, model = load_model_and_tokenizer()
    except Exception as e:
        logging.exception("Model/tokenizer loading failed: %s", e)
        # Save an empty CSV with headers so downstream checks exist
        headers = ["participant_id", "cognitive_negativity", "emotional_flatness", "low_engagement", "psychomotor_slowing", "overall_risk", "raw_response"]
        pd.DataFrame(columns=headers).to_csv(Path(args.output_dir) / "genai_features.csv", index=False)
        logging.error("Wrote empty output CSV at %s due to model load failure. Exiting.", Path(args.output_dir) / "genai_features.csv")
        return

    processed = 0
    start_time = time.time()

    for i, pid in enumerate(tqdm(participants, desc="Semantic features")):
        # estimate time remaining printed by tqdm; keep processing
        # find transcript path from metadata row
        row_meta = meta[meta["participant_id"] == pid].iloc[0]
        transcript_path = row_meta.get("transcript_path")
        if pd.isna(transcript_path) or not Path(str(transcript_path)).exists():
            transcript_path = DATA_DIR / f"{pid}_P" / f"{pid}_TRANSCRIPT.csv"
        try:
            res = get_feature_row(str(pid), tokenizer, model, acoustic_df, visual_df, Path(transcript_path))
            results.append(res)
        except Exception as e:
            logging.exception("Unhandled error for %s: %s", pid, e)
        processed += 1

        # checkpoint
        if processed % CHECKPOINT_EVERY == 0:
            pd.DataFrame(results).to_csv(Path(args.output_dir) / "genai_features.csv", index=False)
            logging.info("Checkpoint saved after %d participants", processed)

        # free GPU memory after batch
        if processed % UNLOAD_AFTER == 0:
            try:
                logging.info("Unloading model to free GPU memory...")
                del model
                del tokenizer
                torch.cuda.empty_cache()
                time.sleep(1)
            except Exception:
                pass
            if processed < len(participants):
                tokenizer, model = load_model_and_tokenizer()

    pd.DataFrame(results).to_csv(Path(args.output_dir) / "genai_features.csv", index=False)
    elapsed = time.time() - start_time
    logging.info("Done. Processed %d participants in %.1f seconds", processed, elapsed)


if __name__ == "__main__":
    main()