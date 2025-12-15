#!/usr/bin/env python3
"""
GenAI Extraction V4: Multimodal Cross-Reference
Idea: Explicitly highlight mismatches between Text (What is said) and Audio/Video (How it is said).
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

# CONFIG
BASE_DIR = Path("/home/dipanjan/rugraj/DIAC-WOZ")
METADATA_PATH = BASE_DIR / "daic_metadata.csv"
ACOUSTIC_PATH = BASE_DIR / "acoustic_features.csv"
VISUAL_PATH = BASE_DIR / "visual_features.csv"
OUTPUT_PATH = BASE_DIR / "genai_features.csv"
CHECKPOINT_PATH = BASE_DIR / "genai_checkpoint.csv"

# Llama-3 Params
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.6

def find_llama3_path():
    cache_base = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    model_folder = cache_base / "models--meta-llama--Meta-Llama-3-8B-Instruct"
    snapshot_dir = model_folder / "snapshots"
    if not snapshot_dir.exists(): raise FileNotFoundError("Llama-3 not found.")
    return str([f for f in snapshot_dir.iterdir() if f.is_dir()][0])

def load_model():
    path = find_llama3_path()
    print(f"Loading Llama-3 from {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=torch.float16)
    return model, tokenizer

def extract_transcript(path):
    try:
        if not os.path.exists(path): return None
        df = pd.read_csv(path, sep='\t')
        df.columns = [c.lower() for c in df.columns]
        p_df = df[df['speaker'].str.lower() == 'participant']
        if len(p_df) == 0: return ""
        return ' '.join(p_df['value'].astype(str).tolist())[:2500]
    except: return None

# --- NEW: INTERPRET THE NUMBERS INTO WORDS ---
def interpret_features(acoustic, visual):
    # These thresholds are heuristics based on DIAC-WOZ distribution
    pitch_var = acoustic.get('covarep_f11_mean', 0.0)
    energy = acoustic.get('covarep_f20_mean', 0.0)
    smile = visual.get('au12_mean', 0.0)
    
    # Pitch Interpretation
    if pitch_var < 0.04: pitch_desc = "VERY MONOTONE (Flat voice)"
    elif pitch_var < 0.08: pitch_desc = "SOMEWHAT MONOTONE"
    else: pitch_desc = "NORMAL / EXPRESSIVE"
        
    # Energy Interpretation
    if energy < -10: energy_desc = "LOW ENERGY (Quiet/Lethargic)"
    else: energy_desc = "NORMAL ENERGY"
    
    # Visual Interpretation
    if smile < 0.2: vis_desc = "FLAT AFFECT (Rarely smiles)"
    elif smile < 1.0: vis_desc = "REDUCED EXPRESSION"
    else: vis_desc = "NORMAL EXPRESSIVENESS"
    
    return pitch_var, pitch_desc, energy, energy_desc, smile, vis_desc

def create_multimodal_prompt(participant_id, transcript, acoustic, visual):
    # Get Interpreted strings
    pv, p_desc, en, e_desc, sm, s_desc = interpret_features(acoustic, visual)
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert psychiatrist analyzing a clinical interview.
Your specific task is to find **CROSS-MODAL MISMATCHES**.

Depression often presents as "Smiling Depression" or "Masking":
- The patient SAYS positive things ("I'm fine", "Work is good").
- But their VOICE and FACE show flatness (Monotone, No Smiles).

If you see a mismatch (Positive Text + Negative Bio-markers), rate HIGH RISK.

<|eot_id|><|start_header_id|>user<|end_header_id|>

PATIENT DATA:
1. TRANSCRIPT EXCERPT:
"{transcript}..."

2. ACOUSTIC ANALYSIS:
- Pitch Variation: {pv:.3f} -> **{p_desc}**
- Energy Level: {en:.1f} -> **{e_desc}**

3. VISUAL ANALYSIS:
- Smile Frequency: {sm:.2f} -> **{s_desc}**

TASK:
1. Analyze if the text matches the bio-markers.
2. If text is positive but bio-markers are flat, flag as "Incongruent/Masking" (High Risk).
3. Output valid JSON in ### JSON RESULTS ### block.

JSON Keys needed: cognitive_negativity, emotional_flatness, low_engagement, psychomotor_slowing, overall_risk (0-10).

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    return prompt

def parse_response(text):
    try:
        marker = "### JSON RESULTS ###"
        if marker in text: 
            json_str = text.split(marker)[1]
        else:
            json_str = text[text.rfind("{"):]
        
        json_str = json_str.split("}")[0] + "}"
        json_str = json_str.replace("'", '"')
        parsed = json.loads(json_str)
        
        # Cleanup
        res = {k: int(parsed.get(k, 0)) for k in ['cognitive_negativity', 'emotional_flatness', 'overall_risk', 'low_engagement', 'psychomotor_slowing']}
        res['llm_reasoning'] = text.split(marker)[0][:500].strip()
        return res
    except: return None

def process_row(pid, row, ac_df, vis_df, model, tok, log):
    try:
        txt = extract_transcript(row['transcript_path'])
        if not txt: return None
        
        ac = ac_df[ac_df['participant_id']==pid].iloc[0].to_dict() if pid in ac_df['participant_id'].values else {}
        vis = vis_df[vis_df['participant_id']==pid].iloc[0].to_dict() if pid in vis_df['participant_id'].values else {}
        
        prompt = create_multimodal_prompt(pid, txt, ac, vis)
        
        in_ids = tok(prompt, return_tensors='pt').to(model.device)
        terminators = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
        
        with torch.no_grad():
            out = model.generate(**in_ids, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, do_sample=True, eos_token_id=terminators)
        
        resp = tok.decode(out[0][in_ids['input_ids'].shape[1]:], skip_special_tokens=True)
        parsed = parse_response(resp)
        
        if parsed:
            return {'participant_id': pid, **parsed, 'raw_response': resp[:200]}
    except Exception as e:
        return None

def main():
    print("GenAI V4: Multimodal Mismatch Detection")
    # Load Data
    meta = pd.read_csv(METADATA_PATH)
    meta['participant_id'] = meta['participant_id'].astype(str)
    ac = pd.read_csv(ACOUSTIC_PATH)
    ac['participant_id'] = ac['participant_id'].astype(str)
    vis = pd.read_csv(VISUAL_PATH)
    vis['participant_id'] = vis['participant_id'].astype(str)
    
    model, tokenizer = load_model()
    
    if OUTPUT_PATH.exists(): OUTPUT_PATH.unlink()
    
    results = []
    for idx, row in tqdm(meta.iterrows(), total=len(meta)):
        pid = str(row['participant_id'])
        res = process_row(pid, row, ac, vis, model, tokenizer, None)
        if res: results.append(res)
        
        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT_PATH, index=False)
            
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
