#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Meditron3-8B-GGUF - Medical Exam Tester
Optimized for Italian Medical Specialization Entrance Exams

Features:
- Meditron3-8B in GGUF format with multiple quantization levels
- Supports Q8_0, Q6_K, Q5_K_M, Q4_K_M quantizations
- Enhanced raw response saving with full context
- Checkpoint system for resumption
- Same test methodology as MedGemma tests
"""

import os
import sys
import time
import json
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
import random
from pathlib import Path

# Check dependencies
try:
    from llama_cpp import Llama
except ImportError:
    print("ERROR: llama-cpp-python not found.")
    print("Install with: pip install llama-cpp-python")
    print("For GPU support: CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --force-reinstall --no-cache-dir")
    sys.exit(1)

# Local models path
MODELS_PATH = r"/home/gueststudente/FiloWhite/"

# Available quantization levels (local files)
QUANTIZATION_LEVELS = {
    "Q8_0": os.path.join(MODELS_PATH, "Meditron3-8B.Q8_0.gguf"),
    "Q6_K": os.path.join(MODELS_PATH, "Meditron3-8B.Q6_K.gguf"),
    "Q5_K_M": os.path.join(MODELS_PATH, "Meditron3-8B.Q5_K_M.gguf"),
    "Q4_K_M": os.path.join(MODELS_PATH, "Meditron3-8B.Q4_K_M.gguf")
}


class Meditron3GGUFMedicalExamTester:
    """
    Tester for Meditron3-8B-GGUF on Italian Medical Specialization Exams
    """
    
    def __init__(self, quantization: str, model_path: str = None, n_gpu_layers: int = -1):
        """
        Initialize the tester with Meditron3-8B-GGUF
        
        Args:
            quantization: Quantization level (Q8_0, Q6_K, Q5_K_M, Q4_K_M)
            model_path: Optional path to local GGUF file. If None, downloads from HuggingFace
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
        """
        self.quantization = quantization
        self.model_name = f"Meditron3-8B-{quantization}"
        
        # Configure logging for this quantization
        log_file = f"meditron3_{quantization}_medical_exam_test.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ],
            force=True
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Loading model: {self.model_name}")
        self.logger.info(f"Quantization: {quantization}")
        
        try:
            # Use local file from QUANTIZATION_LEVELS
            gguf_path = model_path if model_path else QUANTIZATION_LEVELS[quantization]
            
            if not os.path.exists(gguf_path):
                raise FileNotFoundError(f"Model file not found: {gguf_path}")
            
            self.logger.info(f"Loading from local path: {gguf_path}")
            
            # Load model with llama-cpp
            self.logger.info(f"Loading GGUF model with n_gpu_layers={n_gpu_layers}...")
            self.model = Llama(
                model_path=gguf_path,
                n_ctx=4096,           # Context window
                n_gpu_layers=n_gpu_layers,  # GPU layers (-1 = all)
                n_threads=8,          # CPU threads
                verbose=False
            )
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
        
        # Setup checkpoint and raw responses files
        safe_model_name = f"Meditron3_8B_{quantization}"
        self.checkpoint_file = f"checkpoint_{safe_model_name}.json"
        self.raw_responses_file = f"raw_responses_{safe_model_name}.jsonl"
        
        # Open raw responses file in append mode
        self.raw_responses_handle = open(self.raw_responses_file, 'a', encoding='utf-8')
        
        # Statistics
        self.total_processed = 0
        self.extraction_failures = 0
    
    def load_checkpoint(self) -> Dict:
        """Load checkpoint if it exists"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                self.logger.info(f"Checkpoint loaded: Year {checkpoint.get('current_year')}, "
                           f"Run {checkpoint.get('current_run')}, "
                           f"Question {checkpoint.get('current_question')}")
                return checkpoint
        return {
            "current_year": None,
            "current_run": 0,
            "current_question": 0,
            "results": {},
            "consistency_data": [],
            "metrics_data": []
        }
    
    def save_checkpoint(self, checkpoint: Dict):
        """Save checkpoint"""
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    def delete_checkpoint(self):
        """Delete checkpoint at the end"""
        if os.path.exists(self.checkpoint_file):
            try:
                os.remove(self.checkpoint_file)
                self.logger.info("Checkpoint deleted")
            except PermissionError as e:
                self.logger.warning(f"Could not delete checkpoint: {e}")
    
    def _shuffle_options(self, options: List[str]) -> Tuple[List[str], str, Dict[str, str]]:
        """
        Shuffle options randomly and return new correct answer letter + mapping.
        
        Args:
            options: List of 5 options where index 0 is always the correct answer
            
        Returns:
            Tuple of (shuffled_options, new_correct_letter, shuffled_to_original_map)
            shuffled_to_original_map: maps shuffled letter -> original letter (e.g. {'C': 'A'})
        """
        # Create list of (original_index, option_text)
        indexed_options = list(enumerate(options))
        
        # Shuffle
        random.shuffle(indexed_options)
        
        letters = ['A', 'B', 'C', 'D', 'E']
        new_correct_letter = 'A'  # default
        
        # Build mapping: shuffled position -> original letter
        shuffled_to_original = {}
        for new_idx, (orig_idx, _) in enumerate(indexed_options):
            shuffled_letter = letters[new_idx]
            original_letter = letters[orig_idx]
            shuffled_to_original[shuffled_letter] = original_letter
            if orig_idx == 0:  # Original correct answer was at index 0
                new_correct_letter = shuffled_letter
        
        # Extract just the shuffled options
        shuffled_options = [opt for _, opt in indexed_options]
        
        return shuffled_options, new_correct_letter, shuffled_to_original
    
    def process_question(self, question: str, options: List[str], 
                        question_idx: int, year: int, run: int,
                        correct_answer: str = 'A') -> Tuple[str, float]:
        """
        Process a single question with Meditron3-GGUF
        
        Args:
            question: Question text
            options: List of answer options (already shuffled)
            question_idx: Question index (0-based)
            year: Current year
            run: Current run number
            correct_answer: The letter of the correct answer after shuffling
        
        Returns:
            Tuple with answer letter (A-E or ND) and processing time
        """
        start_time = time.time()
        raw_response = ""
        answer_letter = "ND"
        
        try:
            # Build the prompt with Llama 3 chat template format for Meditron3
            prompt_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a medical doctor taking a multiple-choice exam.
You must reply with ONLY one line in this exact format:
The final answer is [LETTER]: [exact text of the chosen option]

Example:
The final answer is C: Hypertension

Do NOT explain. Do NOT reason. Do NOT write anything else.<|eot_id|><|start_header_id|>user<|end_header_id|>

QUESTION:
{question}

OPTIONS:
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}
E) {options[4]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The final answer is"""

            # Generate response with llama-cpp
            max_tokens = 50  # Enough for letter + brief explanation
            
            output = self.model(
                prompt_text,
                max_tokens=max_tokens,
                temperature=0.1,      # Small temperature for variability
                top_p=0.95,
                echo=False,           # Don't include prompt in output
                stop=["<|eot_id|>", "<|end_of_text|>", "QUESTION:", "\n\n"]  # Stop tokens
            )
            
            raw_response = output["choices"][0]["text"].strip()
            
            # Check if model entered thinking mode (<unused94>thought)
            if raw_response.strip().startswith("<unused"):
                self.logger.debug("Model entered thinking mode, retrying with more tokens...")
                # Retry with many more tokens so model can finish reasoning AND produce answer
                output = self.model(
                    prompt_text,
                    max_tokens=2048,  # Extended for thinking + answer
                    temperature=0.1,
                    top_p=0.95,
                    echo=False,
                    stop=["<|eot_id|>", "<|end_of_text|>", "QUESTION:"]
                )
                raw_response = output["choices"][0]["text"].strip()
            
            # Extract the letter
            answer_letter = self._extract_letter(raw_response)
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raw_response = f"ERROR: {str(e)}"
            answer_letter = "ND"
        
        elapsed_time = time.time() - start_time
        
        # Save raw response with full context
        raw_entry = {
            "timestamp": datetime.now().isoformat(),
            "year": year,
            "run": run,
            "question_idx": question_idx + 1,
            "question": question,
            "options": {
                "A": options[0],
                "B": options[1],
                "C": options[2],
                "D": options[3],
                "E": options[4]
            },
            "correct_answer": correct_answer,  # Correct answer after shuffling
            "raw_response": raw_response,
            "extracted_letter": answer_letter,
            "is_correct": answer_letter == correct_answer,
            "processing_time_seconds": elapsed_time
        }
        
        self.raw_responses_handle.write(json.dumps(raw_entry, ensure_ascii=False) + '\n')
        self.raw_responses_handle.flush()
        
        self.total_processed += 1
        if answer_letter == "ND":
            self.extraction_failures += 1
        
        return answer_letter, elapsed_time
    
    def _extract_letter(self, response_text: str) -> str:
        """
        Extract the answer letter from model response
        Prioritizes explicit ANSWER: pattern, then falls back to other methods
        """
        response_text = response_text.strip()
        
        # 0. Handle <unused94>thought blocks: extract text AFTER the block, or strip it
        if '<unused94>' in response_text or '<unused' in response_text:
            # Try to find text after </thought> closing tag
            after_thought = re.split(r'</thought>', response_text, maxsplit=1)
            if len(after_thought) > 1 and after_thought[1].strip():
                response_text = after_thought[1].strip()
            else:
                # No closing tag or nothing after it — strip the thought prefix and search within
                cleaned = re.sub(r'<unused\d*>thought\s*', '', response_text, flags=re.DOTALL).strip()
                if cleaned:
                    response_text = cleaned
        
        # 1. PRIORITY: Look for explicit "ANSWER: X" pattern (what we asked for)
        answer_pattern = r'ANSWER\s*:\s*([A-E])\b'
        match = re.search(answer_pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # 2. Look for "The final answer is X" pattern (Meditron3 format)
        final_answer_pattern = r'(?:The\s+)?final\s+answer\s+is\s+([A-E])\b'
        match = re.search(final_answer_pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # 3. Pattern "A: content" or "A) content" at start
        letter_content_pattern = r'^([A-E])\s*[:\)]'
        match = re.match(letter_content_pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # 4. Check if response starts directly with a letter (well-behaved model)
        if response_text and response_text[0].upper() in ['A', 'B', 'C', 'D', 'E']:
            if len(response_text) == 1 or response_text[1] in ['.', ':', ')', ' ', '\n', ',']:
                return response_text[0].upper()
        
        # 5. Other common answer formats
        patterns = [
            r'^([A-E])\s*$',                                    # Just the letter
            r'^([A-E])[.:\)\s]',                                # Letter with punctuation
            r'(?:the answer is|la risposta è)[\s:]*([A-E])\b',  # "the answer is X"
            r'(?:correct answer|risposta corretta)[\s:]*([A-E])\b',  # "correct answer: X"
            r'\*\*([A-E])\*\*',                                 # **X** markdown bold
            r'^([A-E])\)',                                      # A) at start
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
        
        # 6. Look for isolated letter at the very start of first line only
        first_line = response_text.split('\n')[0].strip()
        if first_line and len(first_line) <= 3:
            for letter in ['A', 'B', 'C', 'D', 'E']:
                if letter in first_line.upper():
                    return letter
        
        # 7. Look for "Option X" pattern
        option_match = re.search(r'(?:option|opzione)\s*([A-E])\b', response_text, re.IGNORECASE)
        if option_match:
            return option_match.group(1).upper()
        
        self.logger.warning(f"Could not extract letter from: '{response_text[:150]}'")
        return "ND"
    
    def run_multi_year_test(self, dataset_path: str, years: List[int], num_repetitions: int):
        """
        Run test on multiple years with checkpoint support
        
        Args:
            dataset_path: Path to folder with Excel files
            years: List of years to test
            num_repetitions: Number of repetitions per year
        """
        checkpoint = self.load_checkpoint()
        
        # Initialize data structures
        if not checkpoint.get("results"):
            checkpoint["results"] = {}
        if not checkpoint.get("consistency_data"):
            checkpoint["consistency_data"] = []
        if not checkpoint.get("metrics_data"):
            checkpoint["metrics_data"] = []
        
        try:
            for year in years:
                # Skip if already completed
                if checkpoint.get("current_year") and year < int(str(checkpoint["current_year"]).split("_")[0] if "_" in str(checkpoint["current_year"]) else checkpoint["current_year"]):
                    self.logger.info(f"Year {year} already completed, skipping")
                    continue
                
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"STARTING TEST FOR YEAR {year}")
                self.logger.info(f"{'='*80}\n")
                
                # Load Excel file
                excel_file = os.path.join(dataset_path, f"{year}_answers_converted_[checked].xlsx")
                self.logger.info(f"Loading file: {excel_file}")
                df = pd.read_excel(excel_file)
                
                # Initialize results for this year
                if str(year) not in checkpoint["results"]:
                    checkpoint["results"][str(year)] = []
                
                # Determine starting run
                start_run = 0
                if checkpoint.get("current_year") == year:
                    start_run = checkpoint.get("current_run", 0)
                
                # Execute repetitions
                for run_idx in range(start_run, num_repetitions):
                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"Year {year} - Run {run_idx + 1}/{num_repetitions}")
                    self.logger.info(f"{'='*60}")
                    
                    checkpoint["current_year"] = year
                    checkpoint["current_run"] = run_idx
                    
                    run_results = self._execute_run(df, year, run_idx, checkpoint)
                    
                    # Save results
                    checkpoint["results"][str(year)].append(run_results["metrics"])
                    checkpoint["consistency_data"].append(run_results["consistency_row"])
                    checkpoint["metrics_data"].append(run_results["metrics_row"])
                    
                    # Reset question counter
                    checkpoint["current_question"] = 0
                    
                    # Save checkpoint
                    self.save_checkpoint(checkpoint)
                    
                    self.logger.info(f"Run {run_idx + 1} completed - Accuracy: {run_results['metrics']['accuracy']:.2f}%, "
                               f"Score: {run_results['metrics']['score']:.2f}")
            
            # Save final results
            self._save_final_results(checkpoint)
            
            # Close raw responses file
            self.raw_responses_handle.close()
            self.logger.info(f"Raw responses saved to: {self.raw_responses_file}")
            self.logger.info(f"Total processed: {self.total_processed}, Extraction failures: {self.extraction_failures}")
            
            # Delete checkpoint
            self.delete_checkpoint()
            
            self.logger.info("\n" + "="*80)
            self.logger.info("TEST COMPLETED SUCCESSFULLY")
            self.logger.info("="*80)
            
        except Exception as e:
            self.logger.error(f"Error during execution: {str(e)}")
            self.logger.info("Checkpoint saved. You can resume execution.")
            raise
    
    def _execute_run(self, df: pd.DataFrame, year: int, run_idx: int, checkpoint: Dict) -> Dict:
        """Execute a single test run"""
        start_run_time = time.time()
        
        # Initialize metrics
        correct_answers = 0
        wrong_answers = 0
        total_time = 0
        answers_list = []
        
        # Determine starting question
        start_question = checkpoint.get("current_question", 0) if checkpoint.get("current_run") == run_idx else 0
        
        # Resume partial answers if needed
        if start_question > 0 and checkpoint.get("partial_answers"):
            answers_list = checkpoint["partial_answers"]
            correct_answers = checkpoint.get("partial_correct", 0)
            wrong_answers = checkpoint.get("partial_wrong", 0)
            total_time = checkpoint.get("partial_time", 0)
        
        # Process each question
        for idx in range(start_question, len(df)):
            row = df.iloc[idx]
            question_num = idx + 1
            
            if question_num % 10 == 0 or idx == start_question:
                self.logger.info(f"Processing question {question_num}/{len(df)}")
            
            # Extract question and options (correct answer is ALWAYS at index 0 / Risposta A)
            question = row['Domanda']
            original_options = [
                row['Risposta A'],  # Index 0 = correct answer
                row['Risposta B'],
                row['Risposta C'],
                row['Risposta D'],
                row['Risposta E']
            ]
            
            # Shuffle options to eliminate position bias
            # This randomizes the order and tracks where the correct answer moved
            shuffled_options, correct_answer, shuffle_map = self._shuffle_options(original_options)
            
            # Process question with shuffled options
            answer, elapsed_time = self.process_question(
                question, shuffled_options, idx, year, run_idx + 1, correct_answer
            )
            
            # Track for consistency: 1 = correct, original letter (B-E) = wrong, ND = no answer
            if answer == correct_answer:
                consistency_value = 1  # Correct (always originally A)
            elif answer in ['A', 'B', 'C', 'D', 'E']:
                # Wrong: get the ORIGINAL letter the model selected (before shuffle)
                consistency_value = shuffle_map[answer]  # e.g., if model said C which was originally B -> "B"
            else:
                consistency_value = "ND"  # No answer extracted
            answers_list.append(consistency_value)
            total_time += elapsed_time
            
            # Evaluate correctness
            if answer == correct_answer:
                correct_answers += 1
            elif answer in ['A', 'B', 'C', 'D', 'E']:
                wrong_answers += 1
            
            # Save partial state
            checkpoint["current_question"] = idx + 1
            checkpoint["partial_answers"] = answers_list
            checkpoint["partial_correct"] = correct_answers
            checkpoint["partial_wrong"] = wrong_answers
            checkpoint["partial_time"] = total_time
            
            # Save checkpoint every 10 questions
            if question_num % 10 == 0:
                self.save_checkpoint(checkpoint)
        
        # Calculate final metrics
        total_questions = len(df)
        accuracy = (correct_answers / total_questions) * 100
        score = correct_answers - (wrong_answers * 0.25)
        time_per_question = total_time / total_questions
        consistency = self._calculate_consistency(answers_list)
        
        # Clean partial data
        checkpoint.pop("partial_answers", None)
        checkpoint.pop("partial_correct", None)
        checkpoint.pop("partial_wrong", None)
        checkpoint.pop("partial_time", None)
        
        run_time = time.time() - start_run_time
        self.logger.info(f"Run completed in {run_time:.1f}s ({time_per_question:.2f}s per question)")
        
        return {
            "metrics": {
                "run_exam": f"{year}_run_{run_idx + 1}",
                "year": year,
                "run": run_idx + 1,
                "accuracy": accuracy,
                "score": score,
                "time_per_question": time_per_question,
                "consistency": consistency,
                "correct_answers": correct_answers,
                "wrong_answers": wrong_answers,
                "total_questions": total_questions
            },
            "consistency_row": {
                "RUN_EXAM": f"{year}_run_{run_idx + 1}",
                **{f"Domanda_{i+1}": ans for i, ans in enumerate(answers_list)}  # 1=correct, 0=wrong, -1=ND
            },
            "metrics_row": {
                "RUN_EXAM": f"{year}_run_{run_idx + 1}",
                "ACCURACY": accuracy,
                "SCORE": score,
                "TIME_PER_QUESTION": time_per_question,
                "CONSISTENCY": consistency
            }
        }
    
    def _calculate_consistency(self, answers: List) -> float:
        """Calculate percentage of valid answers (not ND)"""
        valid_answers = sum(1 for ans in answers if ans != "ND")
        return (valid_answers / len(answers)) * 100 if answers else 0
    
    def _save_final_results(self, checkpoint: Dict):
        """Save final results to JSON and Excel"""
        safe_model_name = f"Meditron3_8B_{self.quantization}"
        
        # 1. Save JSON
        json_output = {
            "model": f"QuantFactory/Meditron3-8B-GGUF",
            "quantization": self.quantization,
            "test_completed": datetime.now().isoformat(),
            "total_processed": self.total_processed,
            "extraction_failures": self.extraction_failures,
            "years": checkpoint["results"]
        }
        
        json_file = f"results_{safe_model_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
        self.logger.info(f"JSON saved: {json_file}")
        
        # 2. Save Consistency Excel
        consistency_df = pd.DataFrame(checkpoint["consistency_data"])
        consistency_file = f"consistency_{safe_model_name}.xlsx"
        consistency_df.to_excel(consistency_file, index=False)
        self.logger.info(f"Consistency Excel saved: {consistency_file}")
        
        # 3. Save Metrics Excel
        metrics_df = pd.DataFrame(checkpoint["metrics_data"])
        metrics_file = f"metrics_{safe_model_name}.xlsx"
        metrics_df.to_excel(metrics_file, index=False)
        self.logger.info(f"Metrics Excel saved: {metrics_file}")


def main():
    """Main entry point"""
    print("="*80)
    print("MEDITRON3-8B-GGUF - MEDICAL EXAM TESTER")
    print("="*80)
    print()
    print("Available quantization levels:")
    print("  [1] Q8_0  - 8-bit (highest quality, ~8.5GB)")
    print("  [2] Q6_K  - 6-bit (good quality, ~6.5GB)")
    print("  [3] Q5_K_M - 5-bit medium (balanced, ~5.5GB)")
    print("  [4] Q4_K_M - 4-bit medium (smallest, ~4.5GB)")
    print("  [5] Run ALL quantizations sequentially")
    print()
    
    choice = input("Select quantization [1-5]: ").strip()
    
    if choice == "5":
        quant_to_test = ["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"]
    elif choice == "1":
        quant_to_test = ["Q8_0"]
    elif choice == "2":
        quant_to_test = ["Q6_K"]
    elif choice == "3":
        quant_to_test = ["Q5_K_M"]
    elif choice == "4":
        quant_to_test = ["Q4_K_M"]
    else:
        print("Invalid selection")
        sys.exit(1)
    
    # Configuration
    dataset_path = r"/home/gueststudente/FiloWhite/Dataset"
    years = [2020, 2021, 2022, 2023, 2024]
    num_repetitions = 50
    
    print(f"\nDataset path: {dataset_path}")
    print(f"Years: {years}")
    print(f"Repetitions per year: {num_repetitions}")
    print(f"Quantizations to test: {quant_to_test}")
    print(f"Total runs per quantization: {len(years) * num_repetitions}")
    print()
    
    # GPU layers (-1 = all on GPU)
    n_gpu_layers = -1
    gpu_choice = input("Offload all layers to GPU? [Y/n]: ").strip().lower()
    if gpu_choice == "n":
        try:
            n_gpu_layers = int(input("Enter number of GPU layers (0 for CPU only): ").strip())
        except ValueError:
            n_gpu_layers = 0
    
    input("\nPress ENTER to start the test (or Ctrl+C to cancel)...")
    print()
    
    for quant in quant_to_test:
        print(f"\n{'#'*80}")
        print(f"# TESTING QUANTIZATION: {quant}")
        print(f"{'#'*80}\n")
        
        try:
            tester = Meditron3GGUFMedicalExamTester(
                quantization=quant,
                n_gpu_layers=n_gpu_layers
            )
            tester.run_multi_year_test(dataset_path, years, num_repetitions)
            
            print(f"\n{quant} TEST COMPLETED!")
            
        except KeyboardInterrupt:
            print(f"\nTest interrupted by user for {quant}. Checkpoint saved.")
            print("You can resume the test by running the script again.")
            break
        except Exception as e:
            print(f"\nFatal error for {quant}: {e}")
            print("Continuing to next quantization...")
            continue
    
    print("\nAll tests finished!")


if __name__ == "__main__":
    main()
