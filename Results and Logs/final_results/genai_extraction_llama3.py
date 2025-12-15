#!/usr/bin/env python3
"""
GenAI Feature Extraction - Llama-3-8B-Instruct Edition
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import time
import glob

warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path("/home/dipanjan/rugraj/DIAC-WOZ")
METADATA_PATH = BASE_DIR / "daic_metadata.csv"
ACOUSTIC_PATH = BASE_DIR / "acoustic_features.csv"
VISUAL_PATH = BASE_DIR / "visual_features.csv"
OUTPUT_PATH = BASE_DIR / "genai_features.csv"
ERROR_LOG_PATH = BASE_DIR / "genai_extraction_errors.log"
CHECKPOINT_PATH = BASE_DIR / "genai_checkpoint.csv"

# Generation Params
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.6     
CHECKPOINT_INTERVAL = 10
CACHE_CLEAR_INTERVAL = 5

def find_llama3_path():
    """Automatically find the Llama-3 snapshot in HF cache"""
    cache_base = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    model_folder = cache_base / "models--meta-llama--Meta-Llama-3-8B-Instruct"
    
    if not model_folder.exists():
        raise FileNotFoundError(f"Llama-3 folder not found at {model_folder}")
    
    # Find the snapshot folder (usually inside 'snapshots')
    snapshot_dir = model_folder / "snapshots"
    subfolders = [f for f in snapshot_dir.iterdir() if f.is_dir()]
    
    if not subfolders:
        raise FileNotFoundError("No snapshot found inside Llama-3 folder")
        
    # Return the first snapshot (usually the valid hash)
    return str(subfolders[0])

def setup_logging():
    if ERROR_LOG_PATH.exists():
        pass
    return open(ERROR_LOG_PATH, 'a')

def log_error(log_file, participant_id, error_msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"[{timestamp}] Participant {participant_id}: {error_msg}\n")
    log_file.flush()

def load_model_and_tokenizer():
    model_path = find_llama3_path()
    print(f"Loading Llama-3 from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model onto GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    return model, tokenizer

def extract_participant_utterances(transcript_path, max_chars=2000):
    try:
        if not os.path.exists(transcript_path):
            return None
        df = pd.read_csv(transcript_path, sep='\t')
        df.columns = [c.lower() for c in df.columns]
        
        participant_df = df[df['speaker'].str.lower() == 'participant'].copy()
        if len(participant_df) == 0:
            return ""
        
        all_text = ' '.join(participant_df['value'].astype(str).tolist())
        return all_text[:max_chars]
    except Exception:
        return None

def create_prompt(participant_id, transcript_text, acoustic_features, visual_features):
    """Llama-3 Specific Prompt Template"""

    covarep_f11 = acoustic_features.get('covarep_f11_mean', 0.0)
    au12_mean = visual_features.get('au12_mean', 0.0)

    # Llama-3 Chat Format
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert clinical psychologist. 
Analyze the patient data for depression severity (0-10).
Think step-by-step.
Output format: First reasoning, then strictly valid JSON inside ### JSON RESULTS ### block.

CALIBRATION EXAMPLES:
1. HEALTHY (Score 0-2): Complains about traffic/work, but has hobbies/humor. (Pitch_Var>0.10, Smiles>1.0) -> {{ "cognitive_negativity": 1, "overall_risk": 1 }}
2. MILD (Score 3-5): "Numb", "Going through motions", low pleasure. (Pitch_Var~0.06) -> {{ "cognitive_negativity": 4, "overall_risk": 4 }}
3. SEVERE (Score 8-10): "Hopeless", "Burden", suicidal thoughts. (Pitch_Var<0.02, Smiles=0) -> {{ "cognitive_negativity": 10, "overall_risk": 10 }}

<|eot_id|><|start_header_id|>user<|end_header_id|>

PATIENT DATA:
Transcript: "{transcript_text}..."
Biomarkers: Pitch_Var={covarep_f11:.2f}, Smiles={au12_mean:.2f}

Step 1: Write reasoning.
Step 2: Output JSON block ### JSON RESULTS ###.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return prompt

def parse_llm_response(response_text):
    """Robust parser for CoT output"""
    try:
        marker = "### JSON RESULTS ###"
        start_idx = response_text.find(marker)
        
        if start_idx == -1:
            start_idx = response_text.rfind("{")
        else:
            start_idx += len(marker)
            
        json_fragment = response_text[start_idx:]
        dict_start = json_fragment.find("{")
        dict_end = json_fragment.rfind("}") + 1
        
        if dict_start == -1 or dict_end == 0: return None
            
        clean_json = json_fragment[dict_start:dict_end].replace("'", '"')
        parsed = json.loads(clean_json)

        result = {
            'cognitive_negativity': int(parsed.get('cognitive_negativity', 0)),
            'emotional_flatness': int(parsed.get('emotional_flatness', 0)),
            'overall_risk': int(parsed.get('overall_risk', 0)),
            'low_engagement': int(parsed.get('low_engagement', 0)),
            'psychomotor_slowing': int(parsed.get('psychomotor_slowing', 0)),
            'llm_reasoning': response_text[:300].replace("\n", " ").strip()
        }
        
        for k in result:
            if isinstance(result[k], int):
                result[k] = max(0, min(10, result[k]))
            
        return result
    except Exception:
        return None

def process_participant(participant_id, row, acoustic_df, visual_df, model, tokenizer, log_file):
    try:
        transcript_text = extract_participant_utterances(row['transcript_path'])
        if not transcript_text: return None

        acoustic_row = acoustic_df[acoustic_df['participant_id'] == participant_id]
        acoustic_features = acoustic_row.iloc[0].to_dict() if len(acoustic_row) > 0 else {}
        visual_row = visual_df[visual_df['participant_id'] == participant_id]
        visual_features = visual_row.iloc[0].to_dict() if len(visual_row) > 0 else {}

        prompt = create_prompt(participant_id, transcript_text, acoustic_features, visual_features)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        
        # Stop token for Llama-3 to prevent endless yapping
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=0.9,
                eos_token_id=terminators
            )
        
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Parse output - Llama-3 output usually follows the system prompt structure directly
        # We split by 'assistant' header usually, but tokenizer.decode removes special tokens.
        # So we just grab the text. 
        # Since we prompted with headers, the 'full_output' contains the whole conversation.
        # We need to find where our prompt ends.
        
        # Simple heuristic: Look for "PATIENT DATA" then look for the reasoning start
        # Or simpler: The model is generating the 'assistant' response.
        # 'outputs' contains the input tokens too. We slice them off.
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response_only = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        parsed = parse_llm_response(response_only)
        
        if parsed is None:
            log_error(log_file, participant_id, f"Parse fail: {response_only[:100]}")
            return None

        result = {
            'participant_id': participant_id,
            **parsed,
            'raw_response': response_only[:500] 
        }
        return result

    except Exception as e:
        log_error(log_file, participant_id, str(e))
        return None

def main():
    print("GenAI Extraction: Llama-3-8B Edition")
    log_file = setup_logging()

    # Load Data
    meta = pd.read_csv(METADATA_PATH)
    meta['participant_id'] = meta['participant_id'].astype(str)
    acoustic = pd.read_csv(ACOUSTIC_PATH)
    acoustic['participant_id'] = acoustic['participant_id'].astype(str)
    
    try: visual = pd.read_csv(VISUAL_PATH); visual['participant_id'] = visual['participant_id'].astype(str)
    except: visual = pd.DataFrame({'participant_id': []})

    model, tokenizer = load_model_and_tokenizer()

    results = []
    processed_ids = set()

    if CHECKPOINT_PATH.exists() and os.path.getsize(CHECKPOINT_PATH) > 0:
        try:
            ckpt = pd.read_csv(CHECKPOINT_PATH)
            ckpt['participant_id'] = ckpt['participant_id'].astype(str)
            results = ckpt.to_dict('records')
            processed_ids = set(ckpt['participant_id'])
            print(f"Resuming from {len(processed_ids)} processed.")
        except: pass

    if len(processed_ids) == 0 and OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()

    print("Starting Extraction...")
    pbar = tqdm(meta.iterrows(), total=len(meta))
    
    batch_counter = 0
    for idx, row in pbar:
        pid = str(row['participant_id'])
        if pid in processed_ids: continue

        res = process_participant(pid, row, acoustic, visual, model, tokenizer, log_file)
        if res:
            results.append(res)
            batch_counter += 1

            if batch_counter % CHECKPOINT_INTERVAL == 0:
                pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
            if batch_counter % CACHE_CLEAR_INTERVAL == 0:
                torch.cuda.empty_cache()

    final_df = pd.DataFrame(results)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Done. Saved to {OUTPUT_PATH}")
    if CHECKPOINT_PATH.exists(): CHECKPOINT_PATH.unlink()

if __name__ == "__main__":
    main()
