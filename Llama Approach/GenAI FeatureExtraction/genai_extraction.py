#!/usr/bin/env python3
"""
GenAI Feature Extraction - Chain of Thought (CoT) Approach
Model: Llama-2-7b-chat-hf
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

warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path("/home/dipanjan/rugraj/DIAC-WOZ")
METADATA_PATH = BASE_DIR / "daic_metadata.csv"
ACOUSTIC_PATH = BASE_DIR / "acoustic_features.csv"
VISUAL_PATH = BASE_DIR / "visual_features.csv"
OUTPUT_PATH = BASE_DIR / "genai_features.csv" # Overwriting previous file
ERROR_LOG_PATH = BASE_DIR / "genai_extraction_errors.log"
CHECKPOINT_PATH = BASE_DIR / "genai_checkpoint.csv"

# Model path
MODEL_PATH = "/home/dipanjan/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"

# Parameters
MAX_NEW_TOKENS = 512  # Increased to allow for reasoning text
TEMPERATURE = 0.6     
CHECKPOINT_INTERVAL = 10
CACHE_CLEAR_INTERVAL = 5

def setup_logging():
    if ERROR_LOG_PATH.exists():
        pass
    return open(ERROR_LOG_PATH, 'a')

def log_error(log_file, participant_id, error_msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"[{timestamp}] Participant {participant_id}: {error_msg}\n")
    log_file.flush()

def load_model_and_tokenizer():
    print("Loading Llama-2-7b-chat-hf model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model onto GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
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
        
        # Join text
        all_text = ' '.join(participant_df['value'].astype(str).tolist())
        return all_text[:max_chars]
    except Exception:
        return None

def create_prompt(participant_id, transcript_text, acoustic_features, visual_features):
    """Chain of Thought Prompt"""

    covarep_f11 = acoustic_features.get('covarep_f11_mean', 0.0)
    au12_mean = visual_features.get('au12_mean', 0.0)

    prompt = f"""[INST] <<SYS>>
You are an expert psychiatrist. Do not guess. Think step-by-step.
<</SYS>>

PATIENT DATA:
- Transcript snippet: "{transcript_text}..."
- Pitch Variance: {covarep_f11:.2f} (Higher = more expressive)
- Smile Intensity: {au12_mean:.2f} (Higher = more smiling)

INSTRUCTIONS:
1. Analyze the transcript for *Protective Factors* (humor, job engagement, hobbies, social connection).
2. Analyze for *Risk Factors* (hopelessness, self-hate, giving up).
3. Compare the two. If Protective Factors > Risk Factors, the patient is HEALTHY (0-3).
4. Only diagnose DEPRESSION (7-10) if there is clear evidence of functional impairment.

OUTPUT FORMAT:
First, write your reasoning. 
Then, output the JSON inside a block labeled ### JSON RESULTS ###.

Example Output:
Reasoning: Patient discusses traffic and work stress, but laughs about a movie. Shows normal range of affect. No signs of anhedonia.
### JSON RESULTS ###
{{
"reasoning_summary": "Normal complaints, healthy social drive",
"cognitive_negativity": 1,
"emotional_flatness": 1,
"overall_risk": 1
}}

Now analyze the current patient:
[/INST]
"""
    return prompt

def parse_llm_response(response_text):
    """Parse JSON specifically from the ### JSON RESULTS ### block"""
    try:
        # 1. Find the delimiter
        marker = "### JSON RESULTS ###"
        start_idx = response_text.find(marker)
        
        if start_idx == -1:
            # Fallback: try finding the last open brace
            start_idx = response_text.rfind("{")
        else:
            start_idx += len(marker)
            
        json_fragment = response_text[start_idx:]
        
        # 2. Extract strictly between first { and last }
        dict_start = json_fragment.find("{")
        dict_end = json_fragment.rfind("}") + 1
        
        if dict_start == -1 or dict_end == 0:
            return None
            
        clean_json = json_fragment[dict_start:dict_end]
        
        # 3. Clean common LLM formatting errors
        clean_json = clean_json.replace("'", '"')
        
        parsed = json.loads(clean_json)

        # 4. Standardize output keys
        # We simplify to the core metrics + the reasoning snippet
        result = {
            'cognitive_negativity': int(parsed.get('cognitive_negativity', 0)),
            'emotional_flatness': int(parsed.get('emotional_flatness', 0)),
            'overall_risk': int(parsed.get('overall_risk', 0)),
            'llm_reasoning': str(parsed.get('reasoning_summary', 'No summary'))[:200]
        }
        
        # Clamp values 0-10
        for k in ['cognitive_negativity', 'emotional_flatness', 'overall_risk']:
            result[k] = max(0, min(10, result[k]))
            
        return result

    except Exception as e:
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

        # Generation
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                top_p=0.9
            )
        
        # Decode only new tokens
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Isolate the response (remove the prompt part)
        response_only = full_output.split("[/INST]")[-1]

        parsed = parse_llm_response(response_only)
        
        if parsed is None:
            log_error(log_file, participant_id, f"Parse fail. Resp: {response_only[:100]}")
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
    print("GenAI Extraction: Chain of Thought Mode")
    log_file = setup_logging()

    # Load Data
    meta = pd.read_csv(METADATA_PATH)
    meta['participant_id'] = meta['participant_id'].astype(str)
    
    acoustic = pd.read_csv(ACOUSTIC_PATH)
    acoustic['participant_id'] = acoustic['participant_id'].astype(str)
    
    try:
        visual = pd.read_csv(VISUAL_PATH)
        visual['participant_id'] = visual['participant_id'].astype(str)
    except:
        visual = pd.DataFrame({'participant_id': []})

    model, tokenizer = load_model_and_tokenizer()

    results = []
    processed_ids = set()

    # Resume?
    if CHECKPOINT_PATH.exists() and os.path.getsize(CHECKPOINT_PATH) > 0:
        try:
            ckpt = pd.read_csv(CHECKPOINT_PATH)
            ckpt['participant_id'] = ckpt['participant_id'].astype(str)
            results = ckpt.to_dict('records')
            processed_ids = set(ckpt['participant_id'])
            print(f"Resuming from {len(processed_ids)} processed.")
        except: pass

    # Clean old output if starting fresh
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




















