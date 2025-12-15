"""
Hierarchical Multimodal Feature Extraction Pipeline for Depression Detection
Using Llama-3-8B-Instruct on NVIDIA H100

Author: Research Pipeline
Date: 2024
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hierarchical_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HierarchicalFeatureExtractor:
    """Main class for hierarchical multimodal feature extraction"""
    
    def __init__(self, base_dir: str, checkpoint_file: str = "checkpoint.json"):
        self.base_dir = Path(base_dir)
        self.checkpoint_file = self.base_dir / checkpoint_file
        self.processed_participants = self.load_checkpoint()
        
        # Initialize model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Find and load Llama-3 model
        self.model_path = self.find_llama_model()
        logger.info(f"Loading model from: {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.model.eval()
        
        # Load data
        self.metadata = pd.read_csv(self.base_dir / "daic_metadata.csv")
        self.acoustic_features = pd.read_csv(self.base_dir / "acoustic_features.csv")
        self.visual_features = pd.read_csv(self.base_dir / "visual_features.csv")
        
        logger.info(f"Loaded {len(self.metadata)} participants")
        
    def find_llama_model(self) -> str:
        """Automatically find Llama-3-8B-Instruct in huggingface cache"""
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_pattern = "models--meta-llama--Meta-Llama-3-8B-Instruct"
        
        model_dir = cache_dir / model_pattern
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Llama-3 model not found in {cache_dir}. "
                "Please download it first using: "
                "huggingface-cli login && huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct"
            )
        
        # Find the snapshot directory
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            raise FileNotFoundError(f"No snapshots found in {model_dir}")
        
        # Get the latest snapshot
        snapshots = [d for d in snapshots_dir.iterdir() if d.is_dir()]
        if not snapshots:
            raise FileNotFoundError(f"No snapshot directories found in {snapshots_dir}")
        
        latest_snapshot = max(snapshots, key=lambda x: x.stat().st_mtime)
        logger.info(f"Found Llama-3 model at: {latest_snapshot}")
        return str(latest_snapshot)
    
    def load_checkpoint(self) -> set:
        """Load processed participants from checkpoint"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
                logger.info(f"Resuming from checkpoint: {len(data['processed'])} participants already processed")
                return set(data['processed'])
        return set()
    
    def save_checkpoint(self, participant_id: str):
        """Save checkpoint after processing each participant"""
        self.processed_participants.add(str(participant_id))
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'processed': list(self.processed_participants),
                'last_updated': datetime.now().isoformat()
            }, f)
    
    def generate_llm_response(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate response from Llama-3 with error handling"""
        try:
            # Format prompt for Llama-3-Instruct
            formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a clinical psychology expert analyzing patient data for depression indicators. Respond ONLY with valid JSON, no additional text.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.3,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            # Clear GPU cache
            del inputs, outputs
            torch.cuda.empty_cache()
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            raise
    
    def extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response, handling conversational filler and truncation"""
        try:
            # Try direct JSON parse first
            return json.loads(response)
        except json.JSONDecodeError:
            # Extract JSON from text (between first { and last })
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to fix common truncation issues
                    json_str = self.fix_truncated_json(json_str)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        pass
            
            # Try to find JSON code block
            code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
            if code_block:
                try:
                    return json.loads(code_block.group(1))
                except json.JSONDecodeError:
                    pass
            
            logger.error(f"Failed to parse JSON from response: {response[:200]}...")
            raise ValueError("Could not extract valid JSON from response")
    
    def fix_truncated_json(self, json_str: str) -> str:
        """Attempt to fix truncated JSON by closing open structures"""
        # Count open/close brackets and braces
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # Remove trailing incomplete content after last complete field
        # Look for patterns like: "field": [... or "field": "...
        truncation_patterns = [
            # Incomplete array: ,\s*"field":\s*\[(?:[^\]](?!]))*
            r',\s*"[^"]*":\s*\[(?:[^\]](?!]))*', 
            # Incomplete string: ,\s*"field":\s*"(?:[^"](?!"))*
            r',\s*"[^"]*":\s*"(?:[^"](?!"))*'
        ]

        # Apply truncation removal (from end to beginning)
        for pattern in truncation_patterns:
            match = re.search(pattern + '$', json_str, re.DOTALL)
            if match:
                json_str = json_str[:match.start()]
                
        # Append closing characters
        json_str += ']' * (open_brackets - json_brackets)
        json_str += '}' * (open_braces - close_braces)
        
        return json_str
    
    def load_transcript(self, participant_id: str) -> str:
        """Load and preprocess transcript for a participant"""
        transcript_path = self.base_dir / f"{participant_id}_P" / f"{participant_id}_TRANSCRIPT.csv"
        
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")
        
        # Read with tab separator
        df = pd.read_csv(transcript_path, sep='\t')
        
        # Filter for participant speech only
        if 'speaker' in df.columns:
            participant_speech = df[df['speaker'] == 'Participant']['value'].tolist()
        else:
            # Fallback: use 'value' column directly
            participant_speech = df['value'].tolist()
        
        # Clean and join
        participant_speech = [str(text).strip() for text in participant_speech if pd.notna(text)]
        return " ".join(participant_speech)
    
    def stage1_text_analysis(self, transcript: str) -> Dict[str, Any]:
        """Stage 1: Text-based feature extraction"""
        prompt = f"""TRANSCRIPT (participant only): {transcript[:4000]}

Extract these features (respond in JSON):
{{
  "linguistic_patterns": {{ "negative_words_density": <0-10>, "first_person_focus": <0-10>, "absolutist_thinking": <0-10>, "past_tense_bias": <0-10>, "hedging_language": <0-10> }},
  "response_quality": {{ "elaboration_level": <0-10>, "coherence": <0-10>, "engagement": <0-10> }},
  "emotional_content": {{ "sadness_expressions": <0-10>, "anhedonia_indicators": <0-10>, "hopelessness": <0-10>, "anxiety_markers": <0-10> }},
  "cognitive_style": {{ "rumination": <0-10>, "self_criticism": <0-10>, "catastrophizing": <0-10> }}
}}"""
        
        response = self.generate_llm_response(prompt)
        return self.extract_json_from_response(response)
    
    def stage1_acoustic_analysis(self, participant_id: str) -> Dict[str, Any]:
        """Stage 1: Acoustic feature extraction"""
        # Get acoustic features for participant
        acoustic_data = self.acoustic_features[
            self.acoustic_features['participant_id'] == participant_id
        ]
        
        if acoustic_data.empty:
            logger.warning(f"No acoustic data for {participant_id}")
            return self.get_default_acoustic_features()
        
        # Extract relevant statistics (note: lowercase column names, use iloc[0] for single row)
        f11_mean = acoustic_data['covarep_f11_mean'].iloc[0] if 'covarep_f11_mean' in acoustic_data.columns else 0.05
        f20_mean = acoustic_data['covarep_f20_mean'].iloc[0] if 'covarep_f20_mean' in acoustic_data.columns else 0.5
        
        prompt = f"""ACOUSTIC MEASUREMENTS:
- Pitch Mean: {f11_mean:.4f} (Low < 0.04 indicates Monotone)
- Energy Mean: {f20_mean:.4f}
- Overall vocal quality suggests {'low energy and monotone speech' if f11_mean < 0.04 else 'normal prosody'}

Interpret for depression (JSON):
{{
  "prosody": {{ "monotony": <0-10>, "vocal_energy": <0-10>, "speech_hesitation": <0-10>, "vocal_strain": <0-10> }},
  "temporal_dynamics": {{ "speaking_rate_abnormality": <0-10>, "response_latency": <0-10> }}
}}"""
        
        response = self.generate_llm_response(prompt)
        return self.extract_json_from_response(response)
    
    def stage1_visual_analysis(self, participant_id: str) -> Dict[str, Any]:
        """Stage 1: Visual feature extraction"""
        # Get visual features for participant
        visual_data = self.visual_features[
            self.visual_features['participant_id'] == participant_id
        ]
        
        if visual_data.empty:
            logger.warning(f"No visual data for {participant_id}")
            return self.get_default_visual_features()
        
        # Extract action units (note: lowercase column names)
        au12_mean = visual_data['au12_mean'].iloc[0] if 'au12_mean' in visual_data.columns else 0.1
        au04_mean = visual_data['au04_mean'].iloc[0] if 'au04_mean' in visual_data.columns else 0.1
        
        prompt = f"""FACIAL MEASUREMENTS:
- Smile (AU12): {au12_mean:.4f} (Low < 0.2 indicates Flat Affect)
- Brow Furrow (AU04): {au04_mean:.4f}
- Overall expressiveness: {'Reduced' if au12_mean < 0.2 else 'Normal'}

Interpret for depression (JSON):
{{
  "facial_affect": {{ "expressiveness": <0-10>, "positive_affect_deficit": <0-10>, "negative_affect_presence": <0-10> }},
  "nonverbal_behavior": {{ "gaze_aversion": <0-10>, "reduced_animation": <0-10> }}
}}"""
        
        response = self.generate_llm_response(prompt)
        return self.extract_json_from_response(response)
    
    def stage2_cross_modal_integration(
        self, 
        text_results: Dict, 
        acoustic_results: Dict, 
        visual_results: Dict
    ) -> Dict[str, Any]:
        """Stage 2: Cross-modal integration analysis"""
        prompt = f"""TEXT RESULTS: {json.dumps(text_results, indent=2)}
ACOUSTIC RESULTS: {json.dumps(acoustic_results, indent=2)}
VISUAL RESULTS: {json.dumps(visual_results, indent=2)}

Identify patterns:
1. CONGRUENCE (Do modalities agree?)
2. COMPENSATION (Is one masking another?)
3. MISMATCH (Positive text vs Flat voice?)

Respond in JSON:
{{
  "cross_modal_patterns": {{ "text_acoustic_mismatch": <0-10>, "text_visual_mismatch": <0-10>, "acoustic_visual_coherence": <0-10>, "multimodal_depression_signal": <0-10> }},
  "clinical_reasoning": "<text>",
  "confidence": <0-10>
}}"""
        
        response = self.generate_llm_response(prompt, max_tokens=512)
        return self.extract_json_from_response(response)
    
    def stage3_temporal_analysis(self, transcript: str) -> Dict[str, Any]:
        """Stage 3: Temporal progression analysis"""
        # Split transcript into thirds
        words = transcript.split()
        n = len(words)
        early = " ".join(words[:n//3])
        middle = " ".join(words[n//3:2*n//3])
        late = " ".join(words[2*n//3:])
        
        prompt = f"""Analyze symptom progression across interview stages:

EARLY (0-33%): {early[:800]}
MIDDLE (33-66%): {middle[:800]}
LATE (66-100%): {late[:800]}

Track changes in depression indicators (JSON):
{{
  "temporal_patterns": {{ "symptom_progression": <-10 to +10, negative=worsening>, "engagement_trajectory": <-10 to +10>, "emotional_shift": <-10 to +10> }},
  "stage_comparison": "<text describing changes>",
  "prognostic_indicator": <0-10>
}}"""
        
        response = self.generate_llm_response(prompt, max_tokens=512)
        return self.extract_json_from_response(response)
    
    def stage4_meta_reasoning(
        self,
        text_results: Dict,
        acoustic_results: Dict,
        visual_results: Dict,
        cross_modal_results: Dict,
        temporal_results: Dict
    ) -> Dict[str, Any]:
        """Stage 4: Final meta-reasoning synthesis"""
        prompt = f"""COMPREHENSIVE ASSESSMENT DATA:

TEXT: {json.dumps(text_results, indent=2)}
ACOUSTIC: {json.dumps(acoustic_results, indent=2)}
VISUAL: {json.dumps(visual_results, indent=2)}
CROSS-MODAL: {json.dumps(cross_modal_results, indent=2)}
TEMPORAL: {json.dumps(temporal_results, indent=2)}

Provide final supervisor assessment (JSON only, be concise):
{{
  "overall_depression_severity": <0-10>,
  "confidence_score": <0-10>,
  "primary_indicators": ["max 3-4 key findings"],
  "risk_factors": ["max 3-4 concerns"],
  "reliability_assessment": <0-10>,
  "clinical_summary": "<one sentence only>"
}}"""
        
        response = self.generate_llm_response(prompt, max_tokens=1024)
        return self.extract_json_from_response(response)
    
    def get_default_acoustic_features(self) -> Dict[str, Any]:
        """Return default acoustic features when data is missing"""
        return {
            "prosody": {"monotony": 5, "vocal_energy": 5, "speech_hesitation": 5, "vocal_strain": 5},
            "temporal_dynamics": {"speaking_rate_abnormality": 5, "response_latency": 5}
        }
    
    def get_default_visual_features(self) -> Dict[str, Any]:
        """Return default visual features when data is missing"""
        return {
            "facial_affect": {"expressiveness": 5, "positive_affect_deficit": 5, "negative_affect_presence": 5},
            "nonverbal_behavior": {"gaze_aversion": 5, "reduced_animation": 5}
        }
    
    def flatten_results(
        self, 
        participant_id: str,
        text: Dict,
        acoustic: Dict,
        visual: Dict,
        cross_modal: Dict,
        temporal: Dict,
        meta: Dict
    ) -> Dict[str, Any]:
        """Flatten nested JSON results into a single row"""
        row = {"participant_id": participant_id}
        
        # Flatten each stage with prefixes
        def flatten_dict(d: Dict, prefix: str = ""):
            for key, value in d.items():
                if isinstance(value, dict):
                    flatten_dict(value, f"{prefix}{key}_")
                elif isinstance(value, list):
                    row[f"{prefix}{key}"] = json.dumps(value)
                else:
                    row[f"{prefix}{key}"] = value
        
        flatten_dict(text, "text_")
        flatten_dict(acoustic, "acoustic_")
        flatten_dict(visual, "visual_")
        flatten_dict(cross_modal, "cross_modal_")
        flatten_dict(temporal, "temporal_")
        flatten_dict(meta, "meta_")
        
        return row
    
    def process_participant(self, participant_id: str) -> Optional[Dict[str, Any]]:
        """Process a single participant through all stages"""
        logger.info(f"Processing participant: {participant_id}")
        
        try:
            # Load transcript
            transcript = self.load_transcript(participant_id)
            
            # Stage 1: Modality-specific analyses
            logger.info(f"  Stage 1: Modality-specific analysis...")
            text_results = self.stage1_text_analysis(transcript)
            acoustic_results = self.stage1_acoustic_analysis(participant_id)
            visual_results = self.stage1_visual_analysis(participant_id)
            
            # Stage 2: Cross-modal integration
            logger.info(f"  Stage 2: Cross-modal integration...")
            cross_modal_results = self.stage2_cross_modal_integration(
                text_results, acoustic_results, visual_results
            )
            
            # Stage 3: Temporal analysis
            logger.info(f"  Stage 3: Temporal analysis...")
            temporal_results = self.stage3_temporal_analysis(transcript)
            
            # Stage 4: Meta-reasoning
            logger.info(f"  Stage 4: Meta-reasoning...")
            meta_results = self.stage4_meta_reasoning(
                text_results, acoustic_results, visual_results,
                cross_modal_results, temporal_results
            )
            
            # Flatten results
            flattened = self.flatten_results(
                participant_id, text_results, acoustic_results, visual_results,
                cross_modal_results, temporal_results, meta_results
            )
            
            # Save checkpoint
            self.save_checkpoint(participant_id)
            
            logger.info(f"  ✓ Completed participant {participant_id}")
            return flattened
            
        except Exception as e:
            logger.error(f"  ✗ Error processing participant {participant_id}: {e}")
            return None
    
    def run_pipeline(self, output_file: str = "genai_hierarchical_features.csv"):
        """Run the complete pipeline for all participants"""
        logger.info("=" * 80)
        logger.info("Starting Hierarchical Multimodal Feature Extraction Pipeline")
        logger.info("=" * 80)
        
        results = []
        participant_ids = self.metadata['participant_id'].unique()
        
        # Filter out already processed participants
        remaining = [pid for pid in participant_ids if str(pid) not in self.processed_participants]
        logger.info(f"Total participants: {len(participant_ids)}")
        logger.info(f"Already processed: {len(self.processed_participants)}")
        logger.info(f"Remaining: {len(remaining)}")
        
        for idx, participant_id in enumerate(remaining, 1):
            logger.info(f"\n[{idx}/{len(remaining)}] Processing {participant_id}...")
            
            result = self.process_participant(participant_id)
            if result:
                results.append(result)
                
                # Save intermediate results every 10 participants
                if idx % 10 == 0:
                    df = pd.DataFrame(results)
                    df.to_csv(self.base_dir / output_file, index=False)
                    logger.info(f"  💾 Saved intermediate results ({len(results)} participants)")
        
        # Final save
        if results:
            df = pd.DataFrame(results)
            output_path = self.base_dir / output_file
            df.to_csv(output_path, index=False)
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Pipeline completed successfully!")
            logger.info(f"Results saved to: {output_path}")
            logger.info(f"Total participants processed: {len(results)}")
            logger.info(f"Total features extracted: {len(df.columns)}")
            logger.info(f"{'=' * 80}")
        else:
            logger.warning("No results to save!")


def main():
    """Main execution function"""
    BASE_DIR = "/home/dipanjan/rugraj/DIAC-WOZ"
    
    try:
        extractor = HierarchicalFeatureExtractor(BASE_DIR)
        extractor.run_pipeline()
        
    except KeyboardInterrupt:
        logger.info("\n\nPipeline interrupted by user. Progress has been saved to checkpoint.")
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
,  # Incomplete array
            r',\s*"[^"]*":\s*"(?:[^"](?!"))*
    
    def load_transcript(self, participant_id: str) -> str:
        """Load and preprocess transcript for a participant"""
        transcript_path = self.base_dir / f"{participant_id}_P" / f"{participant_id}_TRANSCRIPT.csv"
        
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")
        
        # Read with tab separator
        df = pd.read_csv(transcript_path, sep='\t')
        
        # Filter for participant speech only
        if 'speaker' in df.columns:
            participant_speech = df[df['speaker'] == 'Participant']['value'].tolist()
        else:
            # Fallback: use 'value' column directly
            participant_speech = df['value'].tolist()
        
        # Clean and join
        participant_speech = [str(text).strip() for text in participant_speech if pd.notna(text)]
        return " ".join(participant_speech)
    
    def stage1_text_analysis(self, transcript: str) -> Dict[str, Any]:
        """Stage 1: Text-based feature extraction"""
        prompt = f"""TRANSCRIPT (participant only): {transcript[:4000]}

Extract these features (respond in JSON):
{{
  "linguistic_patterns": {{ "negative_words_density": <0-10>, "first_person_focus": <0-10>, "absolutist_thinking": <0-10>, "past_tense_bias": <0-10>, "hedging_language": <0-10> }},
  "response_quality": {{ "elaboration_level": <0-10>, "coherence": <0-10>, "engagement": <0-10> }},
  "emotional_content": {{ "sadness_expressions": <0-10>, "anhedonia_indicators": <0-10>, "hopelessness": <0-10>, "anxiety_markers": <0-10> }},
  "cognitive_style": {{ "rumination": <0-10>, "self_criticism": <0-10>, "catastrophizing": <0-10> }}
}}"""
        
        response = self.generate_llm_response(prompt)
        return self.extract_json_from_response(response)
    
    def stage1_acoustic_analysis(self, participant_id: str) -> Dict[str, Any]:
        """Stage 1: Acoustic feature extraction"""
        # Get acoustic features for participant
        acoustic_data = self.acoustic_features[
            self.acoustic_features['participant_id'] == participant_id
        ]
        
        if acoustic_data.empty:
            logger.warning(f"No acoustic data for {participant_id}")
            return self.get_default_acoustic_features()
        
        # Extract relevant statistics (note: lowercase column names, use iloc[0] for single row)
        f11_mean = acoustic_data['covarep_f11_mean'].iloc[0] if 'covarep_f11_mean' in acoustic_data.columns else 0.05
        f20_mean = acoustic_data['covarep_f20_mean'].iloc[0] if 'covarep_f20_mean' in acoustic_data.columns else 0.5
        
        prompt = f"""ACOUSTIC MEASUREMENTS:
- Pitch Mean: {f11_mean:.4f} (Low < 0.04 indicates Monotone)
- Energy Mean: {f20_mean:.4f}
- Overall vocal quality suggests {'low energy and monotone speech' if f11_mean < 0.04 else 'normal prosody'}

Interpret for depression (JSON):
{{
  "prosody": {{ "monotony": <0-10>, "vocal_energy": <0-10>, "speech_hesitation": <0-10>, "vocal_strain": <0-10> }},
  "temporal_dynamics": {{ "speaking_rate_abnormality": <0-10>, "response_latency": <0-10> }}
}}"""
        
        response = self.generate_llm_response(prompt)
        return self.extract_json_from_response(response)
    
    def stage1_visual_analysis(self, participant_id: str) -> Dict[str, Any]:
        """Stage 1: Visual feature extraction"""
        # Get visual features for participant
        visual_data = self.visual_features[
            self.visual_features['participant_id'] == participant_id
        ]
        
        if visual_data.empty:
            logger.warning(f"No visual data for {participant_id}")
            return self.get_default_visual_features()
        
        # Extract action units (note: lowercase column names)
        au12_mean = visual_data['au12_mean'].iloc[0] if 'au12_mean' in visual_data.columns else 0.1
        au04_mean = visual_data['au04_mean'].iloc[0] if 'au04_mean' in visual_data.columns else 0.1
        
        prompt = f"""FACIAL MEASUREMENTS:
- Smile (AU12): {au12_mean:.4f} (Low < 0.2 indicates Flat Affect)
- Brow Furrow (AU04): {au04_mean:.4f}
- Overall expressiveness: {'Reduced' if au12_mean < 0.2 else 'Normal'}

Interpret for depression (JSON):
{{
  "facial_affect": {{ "expressiveness": <0-10>, "positive_affect_deficit": <0-10>, "negative_affect_presence": <0-10> }},
  "nonverbal_behavior": {{ "gaze_aversion": <0-10>, "reduced_animation": <0-10> }}
}}"""
        
        response = self.generate_llm_response(prompt)
        return self.extract_json_from_response(response)
    
    def stage2_cross_modal_integration(
        self, 
        text_results: Dict, 
        acoustic_results: Dict, 
        visual_results: Dict
    ) -> Dict[str, Any]:
        """Stage 2: Cross-modal integration analysis"""
        prompt = f"""TEXT RESULTS: {json.dumps(text_results, indent=2)}
ACOUSTIC RESULTS: {json.dumps(acoustic_results, indent=2)}
VISUAL RESULTS: {json.dumps(visual_results, indent=2)}

Identify patterns:
1. CONGRUENCE (Do modalities agree?)
2. COMPENSATION (Is one masking another?)
3. MISMATCH (Positive text vs Flat voice?)

Respond in JSON:
{{
  "cross_modal_patterns": {{ "text_acoustic_mismatch": <0-10>, "text_visual_mismatch": <0-10>, "acoustic_visual_coherence": <0-10>, "multimodal_depression_signal": <0-10> }},
  "clinical_reasoning": "<text>",
  "confidence": <0-10>
}}"""
        
        response = self.generate_llm_response(prompt, max_tokens=512)
        return self.extract_json_from_response(response)
    
    def stage3_temporal_analysis(self, transcript: str) -> Dict[str, Any]:
        """Stage 3: Temporal progression analysis"""
        # Split transcript into thirds
        words = transcript.split()
        n = len(words)
        early = " ".join(words[:n//3])
        middle = " ".join(words[n//3:2*n//3])
        late = " ".join(words[2*n//3:])
        
        prompt = f"""Analyze symptom progression across interview stages:

EARLY (0-33%): {early[:800]}
MIDDLE (33-66%): {middle[:800]}
LATE (66-100%): {late[:800]}

Track changes in depression indicators (JSON):
{{
  "temporal_patterns": {{ "symptom_progression": <-10 to +10, negative=worsening>, "engagement_trajectory": <-10 to +10>, "emotional_shift": <-10 to +10> }},
  "stage_comparison": "<text describing changes>",
  "prognostic_indicator": <0-10>
}}"""
        
        response = self.generate_llm_response(prompt, max_tokens=512)
        return self.extract_json_from_response(response)
    
    def stage4_meta_reasoning(
        self,
        text_results: Dict,
        acoustic_results: Dict,
        visual_results: Dict,
        cross_modal_results: Dict,
        temporal_results: Dict
    ) -> Dict[str, Any]:
        """Stage 4: Final meta-reasoning synthesis"""
        prompt = f"""COMPREHENSIVE ASSESSMENT DATA:

TEXT: {json.dumps(text_results, indent=2)}
ACOUSTIC: {json.dumps(acoustic_results, indent=2)}
VISUAL: {json.dumps(visual_results, indent=2)}
CROSS-MODAL: {json.dumps(cross_modal_results, indent=2)}
TEMPORAL: {json.dumps(temporal_results, indent=2)}

Provide final supervisor assessment (JSON only, be concise):
{{
  "overall_depression_severity": <0-10>,
  "confidence_score": <0-10>,
  "primary_indicators": ["max 3-4 key findings"],
  "risk_factors": ["max 3-4 concerns"],
  "reliability_assessment": <0-10>,
  "clinical_summary": "<one sentence only>"
}}"""
        
        response = self.generate_llm_response(prompt, max_tokens=1024)
        return self.extract_json_from_response(response)
    
    def get_default_acoustic_features(self) -> Dict[str, Any]:
        """Return default acoustic features when data is missing"""
        return {
            "prosody": {"monotony": 5, "vocal_energy": 5, "speech_hesitation": 5, "vocal_strain": 5},
            "temporal_dynamics": {"speaking_rate_abnormality": 5, "response_latency": 5}
        }
    
    def get_default_visual_features(self) -> Dict[str, Any]:
        """Return default visual features when data is missing"""
        return {
            "facial_affect": {"expressiveness": 5, "positive_affect_deficit": 5, "negative_affect_presence": 5},
            "nonverbal_behavior": {"gaze_aversion": 5, "reduced_animation": 5}
        }
    
    def flatten_results(
        self, 
        participant_id: str,
        text: Dict,
        acoustic: Dict,
        visual: Dict,
        cross_modal: Dict,
        temporal: Dict,
        meta: Dict
    ) -> Dict[str, Any]:
        """Flatten nested JSON results into a single row"""
        row = {"participant_id": participant_id}
        
        # Flatten each stage with prefixes
        def flatten_dict(d: Dict, prefix: str = ""):
            for key, value in d.items():
                if isinstance(value, dict):
                    flatten_dict(value, f"{prefix}{key}_")
                elif isinstance(value, list):
                    row[f"{prefix}{key}"] = json.dumps(value)
                else:
                    row[f"{prefix}{key}"] = value
        
        flatten_dict(text, "text_")
        flatten_dict(acoustic, "acoustic_")
        flatten_dict(visual, "visual_")
        flatten_dict(cross_modal, "cross_modal_")
        flatten_dict(temporal, "temporal_")
        flatten_dict(meta, "meta_")
        
        return row
    
    def process_participant(self, participant_id: str) -> Optional[Dict[str, Any]]:
        """Process a single participant through all stages"""
        logger.info(f"Processing participant: {participant_id}")
        
        try:
            # Load transcript
            transcript = self.load_transcript(participant_id)
            
            # Stage 1: Modality-specific analyses
            logger.info(f"  Stage 1: Modality-specific analysis...")
            text_results = self.stage1_text_analysis(transcript)
            acoustic_results = self.stage1_acoustic_analysis(participant_id)
            visual_results = self.stage1_visual_analysis(participant_id)
            
            # Stage 2: Cross-modal integration
            logger.info(f"  Stage 2: Cross-modal integration...")
            cross_modal_results = self.stage2_cross_modal_integration(
                text_results, acoustic_results, visual_results
            )
            
            # Stage 3: Temporal analysis
            logger.info(f"  Stage 3: Temporal analysis...")
            temporal_results = self.stage3_temporal_analysis(transcript)
            
            # Stage 4: Meta-reasoning
            logger.info(f"  Stage 4: Meta-reasoning...")
            meta_results = self.stage4_meta_reasoning(
                text_results, acoustic_results, visual_results,
                cross_modal_results, temporal_results
            )
            
            # Flatten results
            flattened = self.flatten_results(
                participant_id, text_results, acoustic_results, visual_results,
                cross_modal_results, temporal_results, meta_results
            )
            
            # Save checkpoint
            self.save_checkpoint(participant_id)
            
            logger.info(f"  ✓ Completed participant {participant_id}")
            return flattened
            
        except Exception as e:
            logger.error(f"  ✗ Error processing participant {participant_id}: {e}")
            return None
    
    def run_pipeline(self, output_file: str = "genai_hierarchical_features.csv"):
        """Run the complete pipeline for all participants"""
        logger.info("=" * 80)
        logger.info("Starting Hierarchical Multimodal Feature Extraction Pipeline")
        logger.info("=" * 80)
        
        results = []
        participant_ids = self.metadata['participant_id'].unique()
        
        # Filter out already processed participants
        remaining = [pid for pid in participant_ids if str(pid) not in self.processed_participants]
        logger.info(f"Total participants: {len(participant_ids)}")
        logger.info(f"Already processed: {len(self.processed_participants)}")
        logger.info(f"Remaining: {len(remaining)}")
        
        for idx, participant_id in enumerate(remaining, 1):
            logger.info(f"\n[{idx}/{len(remaining)}] Processing {participant_id}...")
            
            result = self.process_participant(participant_id)
            if result:
                results.append(result)
                
                # Save intermediate results every 10 participants
                if idx % 10 == 0:
                    df = pd.DataFrame(results)
                    df.to_csv(self.base_dir / output_file, index=False)
                    logger.info(f"  💾 Saved intermediate results ({len(results)} participants)")
        
        # Final save
        if results:
            df = pd.DataFrame(results)
            output_path = self.base_dir / output_file
            df.to_csv(output_path, index=False)
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Pipeline completed successfully!")
            logger.info(f"Results saved to: {output_path}")
            logger.info(f"Total participants processed: {len(results)}")
            logger.info(f"Total features extracted: {len(df.columns)}")
            logger.info(f"{'=' * 80}")
        else:
            logger.warning("No results to save!")


def main():
    """Main execution function"""
    BASE_DIR = "/home/dipanjan/rugraj/DIAC-WOZ"
    
    try:
        extractor = HierarchicalFeatureExtractor(BASE_DIR)
        extractor.run_pipeline()
        
    except KeyboardInterrupt:
        logger.info("\n\nPipeline interrupted by user. Progress has been saved to checkpoint.")
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
,     # Incomplete string
            r',\s*"[^"]*":\s*\{(?:[^\}](?!\}))*
    
    def load_transcript(self, participant_id: str) -> str:
        """Load and preprocess transcript for a participant"""
        transcript_path = self.base_dir / f"{participant_id}_P" / f"{participant_id}_TRANSCRIPT.csv"
        
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")
        
        # Read with tab separator
        df = pd.read_csv(transcript_path, sep='\t')
        
        # Filter for participant speech only
        if 'speaker' in df.columns:
            participant_speech = df[df['speaker'] == 'Participant']['value'].tolist()
        else:
            # Fallback: use 'value' column directly
            participant_speech = df['value'].tolist()
        
        # Clean and join
        participant_speech = [str(text).strip() for text in participant_speech if pd.notna(text)]
        return " ".join(participant_speech)
    
    def stage1_text_analysis(self, transcript: str) -> Dict[str, Any]:
        """Stage 1: Text-based feature extraction"""
        prompt = f"""TRANSCRIPT (participant only): {transcript[:4000]}

Extract these features (respond in JSON):
{{
  "linguistic_patterns": {{ "negative_words_density": <0-10>, "first_person_focus": <0-10>, "absolutist_thinking": <0-10>, "past_tense_bias": <0-10>, "hedging_language": <0-10> }},
  "response_quality": {{ "elaboration_level": <0-10>, "coherence": <0-10>, "engagement": <0-10> }},
  "emotional_content": {{ "sadness_expressions": <0-10>, "anhedonia_indicators": <0-10>, "hopelessness": <0-10>, "anxiety_markers": <0-10> }},
  "cognitive_style": {{ "rumination": <0-10>, "self_criticism": <0-10>, "catastrophizing": <0-10> }}
}}"""
        
        response = self.generate_llm_response(prompt)
        return self.extract_json_from_response(response)
    
    def stage1_acoustic_analysis(self, participant_id: str) -> Dict[str, Any]:
        """Stage 1: Acoustic feature extraction"""
        # Get acoustic features for participant
        acoustic_data = self.acoustic_features[
            self.acoustic_features['participant_id'] == participant_id
        ]
        
        if acoustic_data.empty:
            logger.warning(f"No acoustic data for {participant_id}")
            return self.get_default_acoustic_features()
        
        # Extract relevant statistics (note: lowercase column names, use iloc[0] for single row)
        f11_mean = acoustic_data['covarep_f11_mean'].iloc[0] if 'covarep_f11_mean' in acoustic_data.columns else 0.05
        f20_mean = acoustic_data['covarep_f20_mean'].iloc[0] if 'covarep_f20_mean' in acoustic_data.columns else 0.5
        
        prompt = f"""ACOUSTIC MEASUREMENTS:
- Pitch Mean: {f11_mean:.4f} (Low < 0.04 indicates Monotone)
- Energy Mean: {f20_mean:.4f}
- Overall vocal quality suggests {'low energy and monotone speech' if f11_mean < 0.04 else 'normal prosody'}

Interpret for depression (JSON):
{{
  "prosody": {{ "monotony": <0-10>, "vocal_energy": <0-10>, "speech_hesitation": <0-10>, "vocal_strain": <0-10> }},
  "temporal_dynamics": {{ "speaking_rate_abnormality": <0-10>, "response_latency": <0-10> }}
}}"""
        
        response = self.generate_llm_response(prompt)
        return self.extract_json_from_response(response)
    
    def stage1_visual_analysis(self, participant_id: str) -> Dict[str, Any]:
        """Stage 1: Visual feature extraction"""
        # Get visual features for participant
        visual_data = self.visual_features[
            self.visual_features['participant_id'] == participant_id
        ]
        
        if visual_data.empty:
            logger.warning(f"No visual data for {participant_id}")
            return self.get_default_visual_features()
        
        # Extract action units (note: lowercase column names)
        au12_mean = visual_data['au12_mean'].iloc[0] if 'au12_mean' in visual_data.columns else 0.1
        au04_mean = visual_data['au04_mean'].iloc[0] if 'au04_mean' in visual_data.columns else 0.1
        
        prompt = f"""FACIAL MEASUREMENTS:
- Smile (AU12): {au12_mean:.4f} (Low < 0.2 indicates Flat Affect)
- Brow Furrow (AU04): {au04_mean:.4f}
- Overall expressiveness: {'Reduced' if au12_mean < 0.2 else 'Normal'}

Interpret for depression (JSON):
{{
  "facial_affect": {{ "expressiveness": <0-10>, "positive_affect_deficit": <0-10>, "negative_affect_presence": <0-10> }},
  "nonverbal_behavior": {{ "gaze_aversion": <0-10>, "reduced_animation": <0-10> }}
}}"""
        
        response = self.generate_llm_response(prompt)
        return self.extract_json_from_response(response)
    
    def stage2_cross_modal_integration(
        self, 
        text_results: Dict, 
        acoustic_results: Dict, 
        visual_results: Dict
    ) -> Dict[str, Any]:
        """Stage 2: Cross-modal integration analysis"""
        prompt = f"""TEXT RESULTS: {json.dumps(text_results, indent=2)}
ACOUSTIC RESULTS: {json.dumps(acoustic_results, indent=2)}
VISUAL RESULTS: {json.dumps(visual_results, indent=2)}

Identify patterns:
1. CONGRUENCE (Do modalities agree?)
2. COMPENSATION (Is one masking another?)
3. MISMATCH (Positive text vs Flat voice?)

Respond in JSON:
{{
  "cross_modal_patterns": {{ "text_acoustic_mismatch": <0-10>, "text_visual_mismatch": <0-10>, "acoustic_visual_coherence": <0-10>, "multimodal_depression_signal": <0-10> }},
  "clinical_reasoning": "<text>",
  "confidence": <0-10>
}}"""
        
        response = self.generate_llm_response(prompt, max_tokens=512)
        return self.extract_json_from_response(response)
    
    def stage3_temporal_analysis(self, transcript: str) -> Dict[str, Any]:
        """Stage 3: Temporal progression analysis"""
        # Split transcript into thirds
        words = transcript.split()
        n = len(words)
        early = " ".join(words[:n//3])
        middle = " ".join(words[n//3:2*n//3])
        late = " ".join(words[2*n//3:])
        
        prompt = f"""Analyze symptom progression across interview stages:

EARLY (0-33%): {early[:800]}
MIDDLE (33-66%): {middle[:800]}
LATE (66-100%): {late[:800]}

Track changes in depression indicators (JSON):
{{
  "temporal_patterns": {{ "symptom_progression": <-10 to +10, negative=worsening>, "engagement_trajectory": <-10 to +10>, "emotional_shift": <-10 to +10> }},
  "stage_comparison": "<text describing changes>",
  "prognostic_indicator": <0-10>
}}"""
        
        response = self.generate_llm_response(prompt, max_tokens=512)
        return self.extract_json_from_response(response)
    
    def stage4_meta_reasoning(
        self,
        text_results: Dict,
        acoustic_results: Dict,
        visual_results: Dict,
        cross_modal_results: Dict,
        temporal_results: Dict
    ) -> Dict[str, Any]:
        """Stage 4: Final meta-reasoning synthesis"""
        prompt = f"""COMPREHENSIVE ASSESSMENT DATA:

TEXT: {json.dumps(text_results, indent=2)}
ACOUSTIC: {json.dumps(acoustic_results, indent=2)}
VISUAL: {json.dumps(visual_results, indent=2)}
CROSS-MODAL: {json.dumps(cross_modal_results, indent=2)}
TEMPORAL: {json.dumps(temporal_results, indent=2)}

Provide final supervisor assessment (JSON only, be concise):
{{
  "overall_depression_severity": <0-10>,
  "confidence_score": <0-10>,
  "primary_indicators": ["max 3-4 key findings"],
  "risk_factors": ["max 3-4 concerns"],
  "reliability_assessment": <0-10>,
  "clinical_summary": "<one sentence only>"
}}"""
        
        response = self.generate_llm_response(prompt, max_tokens=1024)
        return self.extract_json_from_response(response)
    
    def get_default_acoustic_features(self) -> Dict[str, Any]:
        """Return default acoustic features when data is missing"""
        return {
            "prosody": {"monotony": 5, "vocal_energy": 5, "speech_hesitation": 5, "vocal_strain": 5},
            "temporal_dynamics": {"speaking_rate_abnormality": 5, "response_latency": 5}
        }
    
    def get_default_visual_features(self) -> Dict[str, Any]:
        """Return default visual features when data is missing"""
        return {
            "facial_affect": {"expressiveness": 5, "positive_affect_deficit": 5, "negative_affect_presence": 5},
            "nonverbal_behavior": {"gaze_aversion": 5, "reduced_animation": 5}
        }
    
    def flatten_results(
        self, 
        participant_id: str,
        text: Dict,
        acoustic: Dict,
        visual: Dict,
        cross_modal: Dict,
        temporal: Dict,
        meta: Dict
    ) -> Dict[str, Any]:
        """Flatten nested JSON results into a single row"""
        row = {"participant_id": participant_id}
        
        # Flatten each stage with prefixes
        def flatten_dict(d: Dict, prefix: str = ""):
            for key, value in d.items():
                if isinstance(value, dict):
                    flatten_dict(value, f"{prefix}{key}_")
                elif isinstance(value, list):
                    row[f"{prefix}{key}"] = json.dumps(value)
                else:
                    row[f"{prefix}{key}"] = value
        
        flatten_dict(text, "text_")
        flatten_dict(acoustic, "acoustic_")
        flatten_dict(visual, "visual_")
        flatten_dict(cross_modal, "cross_modal_")
        flatten_dict(temporal, "temporal_")
        flatten_dict(meta, "meta_")
        
        return row
    
    def process_participant(self, participant_id: str) -> Optional[Dict[str, Any]]:
        """Process a single participant through all stages"""
        logger.info(f"Processing participant: {participant_id}")
        
        try:
            # Load transcript
            transcript = self.load_transcript(participant_id)
            
            # Stage 1: Modality-specific analyses
            logger.info(f"  Stage 1: Modality-specific analysis...")
            text_results = self.stage1_text_analysis(transcript)
            acoustic_results = self.stage1_acoustic_analysis(participant_id)
            visual_results = self.stage1_visual_analysis(participant_id)
            
            # Stage 2: Cross-modal integration
            logger.info(f"  Stage 2: Cross-modal integration...")
            cross_modal_results = self.stage2_cross_modal_integration(
                text_results, acoustic_results, visual_results
            )
            
            # Stage 3: Temporal analysis
            logger.info(f"  Stage 3: Temporal analysis...")
            temporal_results = self.stage3_temporal_analysis(transcript)
            
            # Stage 4: Meta-reasoning
            logger.info(f"  Stage 4: Meta-reasoning...")
            meta_results = self.stage4_meta_reasoning(
                text_results, acoustic_results, visual_results,
                cross_modal_results, temporal_results
            )
            
            # Flatten results
            flattened = self.flatten_results(
                participant_id, text_results, acoustic_results, visual_results,
                cross_modal_results, temporal_results, meta_results
            )
            
            # Save checkpoint
            self.save_checkpoint(participant_id)
            
            logger.info(f"  ✓ Completed participant {participant_id}")
            return flattened
            
        except Exception as e:
            logger.error(f"  ✗ Error processing participant {participant_id}: {e}")
            return None
    
    def run_pipeline(self, output_file: str = "genai_hierarchical_features.csv"):
        """Run the complete pipeline for all participants"""
        logger.info("=" * 80)
        logger.info("Starting Hierarchical Multimodal Feature Extraction Pipeline")
        logger.info("=" * 80)
        
        results = []
        participant_ids = self.metadata['participant_id'].unique()
        
        # Filter out already processed participants
        remaining = [pid for pid in participant_ids if str(pid) not in self.processed_participants]
        logger.info(f"Total participants: {len(participant_ids)}")
        logger.info(f"Already processed: {len(self.processed_participants)}")
        logger.info(f"Remaining: {len(remaining)}")
        
        for idx, participant_id in enumerate(remaining, 1):
            logger.info(f"\n[{idx}/{len(remaining)}] Processing {participant_id}...")
            
            result = self.process_participant(participant_id)
            if result:
                results.append(result)
                
                # Save intermediate results every 10 participants
                if idx % 10 == 0:
                    df = pd.DataFrame(results)
                    df.to_csv(self.base_dir / output_file, index=False)
                    logger.info(f"  💾 Saved intermediate results ({len(results)} participants)")
        
        # Final save
        if results:
            df = pd.DataFrame(results)
            output_path = self.base_dir / output_file
            df.to_csv(output_path, index=False)
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Pipeline completed successfully!")
            logger.info(f"Results saved to: {output_path}")
            logger.info(f"Total participants processed: {len(results)}")
            logger.info(f"Total features extracted: {len(df.columns)}")
            logger.info(f"{'=' * 80}")
        else:
            logger.warning("No results to save!")


def main():
    """Main execution function"""
    BASE_DIR = "/home/dipanjan/rugraj/DIAC-WOZ"
    
    try:
        extractor = HierarchicalFeatureExtractor(BASE_DIR)
        extractor.run_pipeline()
        
    except KeyboardInterrupt:
        logger.info("\n\nPipeline interrupted by user. Progress has been saved to checkpoint.")
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
,  # Incomplete object
        ]
        
        for pattern in truncation_patterns:
            json_str = re.sub(pattern, '', json_str)
        
        # Close any open arrays
        json_str += ']' * (open_brackets - close_brackets)
        
        # Close any open objects
        json_str += '}' * (open_braces - close_braces)
        
        return json_str
    
    def load_transcript(self, participant_id: str) -> str:
        """Load and preprocess transcript for a participant"""
        transcript_path = self.base_dir / f"{participant_id}_P" / f"{participant_id}_TRANSCRIPT.csv"
        
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript not found: {transcript_path}")
        
        # Read with tab separator
        df = pd.read_csv(transcript_path, sep='\t')
        
        # Filter for participant speech only
        if 'speaker' in df.columns:
            participant_speech = df[df['speaker'] == 'Participant']['value'].tolist()
        else:
            # Fallback: use 'value' column directly
            participant_speech = df['value'].tolist()
        
        # Clean and join
        participant_speech = [str(text).strip() for text in participant_speech if pd.notna(text)]
        return " ".join(participant_speech)
    
    def stage1_text_analysis(self, transcript: str) -> Dict[str, Any]:
        """Stage 1: Text-based feature extraction"""
        prompt = f"""TRANSCRIPT (participant only): {transcript[:4000]}

Extract these features (respond in JSON):
{{
  "linguistic_patterns": {{ "negative_words_density": <0-10>, "first_person_focus": <0-10>, "absolutist_thinking": <0-10>, "past_tense_bias": <0-10>, "hedging_language": <0-10> }},
  "response_quality": {{ "elaboration_level": <0-10>, "coherence": <0-10>, "engagement": <0-10> }},
  "emotional_content": {{ "sadness_expressions": <0-10>, "anhedonia_indicators": <0-10>, "hopelessness": <0-10>, "anxiety_markers": <0-10> }},
  "cognitive_style": {{ "rumination": <0-10>, "self_criticism": <0-10>, "catastrophizing": <0-10> }}
}}"""
        
        response = self.generate_llm_response(prompt)
        return self.extract_json_from_response(response)
    
    def stage1_acoustic_analysis(self, participant_id: str) -> Dict[str, Any]:
        """Stage 1: Acoustic feature extraction"""
        # Get acoustic features for participant
        acoustic_data = self.acoustic_features[
            self.acoustic_features['participant_id'] == participant_id
        ]
        
        if acoustic_data.empty:
            logger.warning(f"No acoustic data for {participant_id}")
            return self.get_default_acoustic_features()
        
        # Extract relevant statistics (note: lowercase column names, use iloc[0] for single row)
        f11_mean = acoustic_data['covarep_f11_mean'].iloc[0] if 'covarep_f11_mean' in acoustic_data.columns else 0.05
        f20_mean = acoustic_data['covarep_f20_mean'].iloc[0] if 'covarep_f20_mean' in acoustic_data.columns else 0.5
        
        prompt = f"""ACOUSTIC MEASUREMENTS:
- Pitch Mean: {f11_mean:.4f} (Low < 0.04 indicates Monotone)
- Energy Mean: {f20_mean:.4f}
- Overall vocal quality suggests {'low energy and monotone speech' if f11_mean < 0.04 else 'normal prosody'}

Interpret for depression (JSON):
{{
  "prosody": {{ "monotony": <0-10>, "vocal_energy": <0-10>, "speech_hesitation": <0-10>, "vocal_strain": <0-10> }},
  "temporal_dynamics": {{ "speaking_rate_abnormality": <0-10>, "response_latency": <0-10> }}
}}"""
        
        response = self.generate_llm_response(prompt)
        return self.extract_json_from_response(response)
    
    def stage1_visual_analysis(self, participant_id: str) -> Dict[str, Any]:
        """Stage 1: Visual feature extraction"""
        # Get visual features for participant
        visual_data = self.visual_features[
            self.visual_features['participant_id'] == participant_id
        ]
        
        if visual_data.empty:
            logger.warning(f"No visual data for {participant_id}")
            return self.get_default_visual_features()
        
        # Extract action units (note: lowercase column names)
        au12_mean = visual_data['au12_mean'].iloc[0] if 'au12_mean' in visual_data.columns else 0.1
        au04_mean = visual_data['au04_mean'].iloc[0] if 'au04_mean' in visual_data.columns else 0.1
        
        prompt = f"""FACIAL MEASUREMENTS:
- Smile (AU12): {au12_mean:.4f} (Low < 0.2 indicates Flat Affect)
- Brow Furrow (AU04): {au04_mean:.4f}
- Overall expressiveness: {'Reduced' if au12_mean < 0.2 else 'Normal'}

Interpret for depression (JSON):
{{
  "facial_affect": {{ "expressiveness": <0-10>, "positive_affect_deficit": <0-10>, "negative_affect_presence": <0-10> }},
  "nonverbal_behavior": {{ "gaze_aversion": <0-10>, "reduced_animation": <0-10> }}
}}"""
        
        response = self.generate_llm_response(prompt)
        return self.extract_json_from_response(response)
    
    def stage2_cross_modal_integration(
        self, 
        text_results: Dict, 
        acoustic_results: Dict, 
        visual_results: Dict
    ) -> Dict[str, Any]:
        """Stage 2: Cross-modal integration analysis"""
        prompt = f"""TEXT RESULTS: {json.dumps(text_results, indent=2)}
ACOUSTIC RESULTS: {json.dumps(acoustic_results, indent=2)}
VISUAL RESULTS: {json.dumps(visual_results, indent=2)}

Identify patterns:
1. CONGRUENCE (Do modalities agree?)
2. COMPENSATION (Is one masking another?)
3. MISMATCH (Positive text vs Flat voice?)

Respond in JSON:
{{
  "cross_modal_patterns": {{ "text_acoustic_mismatch": <0-10>, "text_visual_mismatch": <0-10>, "acoustic_visual_coherence": <0-10>, "multimodal_depression_signal": <0-10> }},
  "clinical_reasoning": "<text>",
  "confidence": <0-10>
}}"""
        
        response = self.generate_llm_response(prompt, max_tokens=512)
        return self.extract_json_from_response(response)
    
    def stage3_temporal_analysis(self, transcript: str) -> Dict[str, Any]:
        """Stage 3: Temporal progression analysis"""
        # Split transcript into thirds
        words = transcript.split()
        n = len(words)
        early = " ".join(words[:n//3])
        middle = " ".join(words[n//3:2*n//3])
        late = " ".join(words[2*n//3:])
        
        prompt = f"""Analyze symptom progression across interview stages:

EARLY (0-33%): {early[:800]}
MIDDLE (33-66%): {middle[:800]}
LATE (66-100%): {late[:800]}

Track changes in depression indicators (JSON):
{{
  "temporal_patterns": {{ "symptom_progression": <-10 to +10, negative=worsening>, "engagement_trajectory": <-10 to +10>, "emotional_shift": <-10 to +10> }},
  "stage_comparison": "<text describing changes>",
  "prognostic_indicator": <0-10>
}}"""
        
        response = self.generate_llm_response(prompt, max_tokens=512)
        return self.extract_json_from_response(response)
    
    def stage4_meta_reasoning(
        self,
        text_results: Dict,
        acoustic_results: Dict,
        visual_results: Dict,
        cross_modal_results: Dict,
        temporal_results: Dict
    ) -> Dict[str, Any]:
        """Stage 4: Final meta-reasoning synthesis"""
        prompt = f"""COMPREHENSIVE ASSESSMENT DATA:

TEXT: {json.dumps(text_results, indent=2)}
ACOUSTIC: {json.dumps(acoustic_results, indent=2)}
VISUAL: {json.dumps(visual_results, indent=2)}
CROSS-MODAL: {json.dumps(cross_modal_results, indent=2)}
TEMPORAL: {json.dumps(temporal_results, indent=2)}

Provide final supervisor assessment (JSON only, be concise):
{{
  "overall_depression_severity": <0-10>,
  "confidence_score": <0-10>,
  "primary_indicators": ["max 3-4 key findings"],
  "risk_factors": ["max 3-4 concerns"],
  "reliability_assessment": <0-10>,
  "clinical_summary": "<one sentence only>"
}}"""
        
        response = self.generate_llm_response(prompt, max_tokens=1024)
        return self.extract_json_from_response(response)
    
    def get_default_acoustic_features(self) -> Dict[str, Any]:
        """Return default acoustic features when data is missing"""
        return {
            "prosody": {"monotony": 5, "vocal_energy": 5, "speech_hesitation": 5, "vocal_strain": 5},
            "temporal_dynamics": {"speaking_rate_abnormality": 5, "response_latency": 5}
        }
    
    def get_default_visual_features(self) -> Dict[str, Any]:
        """Return default visual features when data is missing"""
        return {
            "facial_affect": {"expressiveness": 5, "positive_affect_deficit": 5, "negative_affect_presence": 5},
            "nonverbal_behavior": {"gaze_aversion": 5, "reduced_animation": 5}
        }
    
    def flatten_results(
        self, 
        participant_id: str,
        text: Dict,
        acoustic: Dict,
        visual: Dict,
        cross_modal: Dict,
        temporal: Dict,
        meta: Dict
    ) -> Dict[str, Any]:
        """Flatten nested JSON results into a single row"""
        row = {"participant_id": participant_id}
        
        # Flatten each stage with prefixes
        def flatten_dict(d: Dict, prefix: str = ""):
            for key, value in d.items():
                if isinstance(value, dict):
                    flatten_dict(value, f"{prefix}{key}_")
                elif isinstance(value, list):
                    row[f"{prefix}{key}"] = json.dumps(value)
                else:
                    row[f"{prefix}{key}"] = value
        
        flatten_dict(text, "text_")
        flatten_dict(acoustic, "acoustic_")
        flatten_dict(visual, "visual_")
        flatten_dict(cross_modal, "cross_modal_")
        flatten_dict(temporal, "temporal_")
        flatten_dict(meta, "meta_")
        
        return row
    
    def process_participant(self, participant_id: str) -> Optional[Dict[str, Any]]:
        """Process a single participant through all stages"""
        logger.info(f"Processing participant: {participant_id}")
        
        try:
            # Load transcript
            transcript = self.load_transcript(participant_id)
            
            # Stage 1: Modality-specific analyses
            logger.info(f"  Stage 1: Modality-specific analysis...")
            text_results = self.stage1_text_analysis(transcript)
            acoustic_results = self.stage1_acoustic_analysis(participant_id)
            visual_results = self.stage1_visual_analysis(participant_id)
            
            # Stage 2: Cross-modal integration
            logger.info(f"  Stage 2: Cross-modal integration...")
            cross_modal_results = self.stage2_cross_modal_integration(
                text_results, acoustic_results, visual_results
            )
            
            # Stage 3: Temporal analysis
            logger.info(f"  Stage 3: Temporal analysis...")
            temporal_results = self.stage3_temporal_analysis(transcript)
            
            # Stage 4: Meta-reasoning
            logger.info(f"  Stage 4: Meta-reasoning...")
            meta_results = self.stage4_meta_reasoning(
                text_results, acoustic_results, visual_results,
                cross_modal_results, temporal_results
            )
            
            # Flatten results
            flattened = self.flatten_results(
                participant_id, text_results, acoustic_results, visual_results,
                cross_modal_results, temporal_results, meta_results
            )
            
            # Save checkpoint
            self.save_checkpoint(participant_id)
            
            logger.info(f"  ✓ Completed participant {participant_id}")
            return flattened
            
        except Exception as e:
            logger.error(f"  ✗ Error processing participant {participant_id}: {e}")
            return None
    
    def run_pipeline(self, output_file: str = "genai_hierarchical_features.csv"):
        """Run the complete pipeline for all participants"""
        logger.info("=" * 80)
        logger.info("Starting Hierarchical Multimodal Feature Extraction Pipeline")
        logger.info("=" * 80)
        
        results = []
        participant_ids = self.metadata['participant_id'].unique()
        
        # Filter out already processed participants
        remaining = [pid for pid in participant_ids if str(pid) not in self.processed_participants]
        logger.info(f"Total participants: {len(participant_ids)}")
        logger.info(f"Already processed: {len(self.processed_participants)}")
        logger.info(f"Remaining: {len(remaining)}")
        
        for idx, participant_id in enumerate(remaining, 1):
            logger.info(f"\n[{idx}/{len(remaining)}] Processing {participant_id}...")
            
            result = self.process_participant(participant_id)
            if result:
                results.append(result)
                
                # Save intermediate results every 10 participants
                if idx % 10 == 0:
                    df = pd.DataFrame(results)
                    df.to_csv(self.base_dir / output_file, index=False)
                    logger.info(f"  💾 Saved intermediate results ({len(results)} participants)")
        
        # Final save
        if results:
            df = pd.DataFrame(results)
            output_path = self.base_dir / output_file
            df.to_csv(output_path, index=False)
            logger.info(f"\n{'=' * 80}")
            logger.info(f"Pipeline completed successfully!")
            logger.info(f"Results saved to: {output_path}")
            logger.info(f"Total participants processed: {len(results)}")
            logger.info(f"Total features extracted: {len(df.columns)}")
            logger.info(f"{'=' * 80}")
        else:
            logger.warning("No results to save!")


def main():
    """Main execution function"""
    BASE_DIR = "/home/dipanjan/rugraj/DIAC-WOZ"
    
    try:
        extractor = HierarchicalFeatureExtractor(BASE_DIR)
        extractor.run_pipeline()
        
    except KeyboardInterrupt:
        logger.info("\n\nPipeline interrupted by user. Progress has been saved to checkpoint.")
    except Exception as e:
        logger.error(f"\n\nFatal error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
