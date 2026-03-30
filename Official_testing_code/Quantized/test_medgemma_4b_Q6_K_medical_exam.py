#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MedGemma 1.5 4B IT (GGUF Q6_K) - Medical Exam Tester
Optimized for Italian Medical Specialization Entrance Exams

Run with: /home/gueststudente/miniconda3/envs/testing_llm_med/bin/python test_medgemma_4b_Q6_K_medical_exam.py

Features:
- MedGemma 1.5 4B IT GGUF Q6_K quantization via llama-cpp-python
- Full reasoning support (model may use thinking blocks)
- Enhanced raw response saving with full context
- Improved answer extraction with multiple patterns + content matching
- Checkpoint system for resumption
- Same test methodology as other model tests
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
    sys.exit(1)

# Model configuration
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "medgemma-gguf", "medgemma-1.5-4b-it.Q6_K.gguf")
QUANT_NAME = "Q6_K"


class MedGemmaGGUFMedicalExamTester:
    """
    Tester for MedGemma 1.5 4B IT GGUF on Italian Medical Specialization Exams
    """
    
    def __init__(self):
        self.model_name = f"MedGemma-1.5-4B-IT-GGUF-{QUANT_NAME}"
        
        safe_model_name = f"MedGemma_4B_{QUANT_NAME}"
        log_file = f"medgemma_4b_{QUANT_NAME}_medical_exam_test.log"
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
        self.logger.info(f"Model path: {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            self.logger.error(f"Model file not found: {MODEL_PATH}")
            raise FileNotFoundError(f"GGUF model not found at {MODEL_PATH}")
        
        try:
            self.logger.info("Loading GGUF model (this may take a moment)...")
            self.model = Llama(
                model_path=MODEL_PATH,
                n_gpu_layers=-1,    # Offload all layers to GPU
                n_ctx=2560,         # Context window (112 prompt + 2048 gen + margin)
                n_batch=1024,       # Larger batch for faster prompt processing
                flash_attn=True,    # Flash attention for speed
                verbose=False       # Suppress llama.cpp output
            )
            self.logger.info("Model loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
        
        self.checkpoint_file = f"checkpoint_{safe_model_name}.json"
        self.raw_responses_file = f"raw_responses_{safe_model_name}.jsonl"
        
        self.raw_responses_handle = open(self.raw_responses_file, 'a', encoding='utf-8')
        
        self.total_processed = 0
        self.extraction_failures = 0
    
    def load_checkpoint(self) -> Dict:
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
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    
    def delete_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            try:
                os.remove(self.checkpoint_file)
                self.logger.info("Checkpoint deleted")
            except PermissionError as e:
                self.logger.warning(f"Could not delete checkpoint: {e}")
    
    def _shuffle_options(self, options: List[str]) -> Tuple[List[str], str, Dict[str, str]]:
        indexed_options = list(enumerate(options))
        random.shuffle(indexed_options)
        
        letters = ['A', 'B', 'C', 'D', 'E']
        new_correct_letter = 'A'
        
        shuffled_to_original = {}
        for new_idx, (orig_idx, _) in enumerate(indexed_options):
            shuffled_letter = letters[new_idx]
            original_letter = letters[orig_idx]
            shuffled_to_original[shuffled_letter] = original_letter
            if orig_idx == 0:
                new_correct_letter = shuffled_letter
        
        shuffled_options = [opt for _, opt in indexed_options]
        return shuffled_options, new_correct_letter, shuffled_to_original
    
    def process_question(self, question: str, options: List[str], 
                        question_idx: int, year: int, run: int,
                        correct_answer: str = 'A') -> Tuple[str, float]:
        start_time = time.time()
        raw_response = ""
        answer_letter = "ND"
        
        try:
            prompt_text = f"""You are a highly skilled italian medical doctor taking a multiple-choice exam.
After your reasoning, you MUST produce this FINAL ANSWER FORMAT (exactly one line):
ANSWER: [LETTER]: [exact text of the chosen option]

Where [LETTER] is exactly one of: A, B, C, D, or E.

QUESTION:
{question}

OPTIONS:
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}
E) {options[4]}"""

            response = self.model.create_chat_completion(
                messages=[
                    {"role": "user", "content": prompt_text}
                ],
                max_tokens=2048,
                temperature=0,
            )
            
            raw_response = response['choices'][0]['message']['content'].strip()
            
            answer_letter = self._extract_letter(raw_response, options)
            
            if answer_letter == "ND":
                self.logger.debug(f"Extraction failed, retrying with forceful prompt. Original: '{raw_response[:100]}'")
                retry_prompt = f"""Answer the multiple-choice question with ONLY a single letter: A, B, C, D, or E. Nothing else.

{question}

A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}
E) {options[4]}

The correct answer letter is:"""

                retry_response = self.model.create_chat_completion(
                    messages=[
                        {"role": "user", "content": retry_prompt}
                    ],
                    max_tokens=20,
                    temperature=0.01,
                    top_p=0.95,
                )
                retry_text = retry_response['choices'][0]['message']['content'].strip()
                retry_letter = self._extract_letter(retry_text, options)
                if retry_letter != "ND":
                    answer_letter = retry_letter
                    raw_response = raw_response + " [RETRY] " + retry_text
                    self.logger.debug(f"Retry succeeded: {answer_letter}")
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            raw_response = f"ERROR: {str(e)}"
            answer_letter = "ND"
        
        elapsed_time = time.time() - start_time
        
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
            "correct_answer": correct_answer,
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
    
    def _extract_letter(self, response_text: str, options: List[str] = None) -> str:
        response_text = response_text.strip()
        
        # Handle <unused94>thought blocks (MedGemma reasoning mode)
        if '<unused94>' in response_text:
            after_thought = re.split(r'</thought>', response_text, maxsplit=1)
            if len(after_thought) > 1 and after_thought[1].strip():
                response_text = after_thought[1].strip()
            else:
                cleaned = re.sub(r'<unused94>thought\s*', '', response_text, flags=re.DOTALL).strip()
                if cleaned:
                    response_text = cleaned
        
        # Handle <think> blocks for compatibility
        if '<think>' in response_text:
            after_think = re.split(r'</think>', response_text, maxsplit=1)
            if len(after_think) > 1 and after_think[1].strip():
                response_text = after_think[1].strip()
            else:
                cleaned = re.sub(r'<think>\s*', '', response_text, flags=re.DOTALL).strip()
                if cleaned:
                    response_text = cleaned
        
        answer_pattern = r'ANSWER\s*:\s*([A-E])\b'
        match = re.search(answer_pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        if response_text and response_text[0].upper() in ['A', 'B', 'C', 'D', 'E']:
            if len(response_text) == 1 or response_text[1] in ['.', ':', ')', ' ', '\n', ',']:
                return response_text[0].upper()
        
        final_answer_pattern = r'(?:The\s+)?final\s+answer\s+is\s+([A-E])\b'
        match = re.search(final_answer_pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        letter_content_pattern = r'^([A-E])\s*[:\)]'
        match = re.match(letter_content_pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        patterns = [
            r'^([A-E])\s*$',
            r'^([A-E])[.:\)\s]',
            r'(?:the answer is|la risposta è)[\s:]*([A-E])\b',
            r'(?:risposta corretta)[\s:è]+([A-E])\b',
            r'(?:correct answer)[\s:]+([A-E])\b',
            r'(?:La\s+)?(?:risposta|diagnosi)[\s\w]*?(?:è|:)[^A-E]*([A-E])\)',
            r'\*\*([A-E])\*\*',
            r'(?:option|opzione)\s*([A-E])\b',
            r'\b([A-E])\)\s+\w',
            r'\b([A-E])\.',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).upper()
        
        first_line = response_text.split('\n')[0].strip()
        if first_line and len(first_line) <= 3:
            for letter in ['A', 'B', 'C', 'D', 'E']:
                if letter in first_line.upper():
                    return letter
        
        if options and len(options) == 5:
            result = self._match_option_text(response_text, options)
            if result:
                return result
        
        self.logger.warning(f"Could not extract letter from: '{response_text[:200]}'")
        return "ND"
    
    def _match_option_text(self, response_text: str, options: List[str]) -> Optional[str]:
        letters = ['A', 'B', 'C', 'D', 'E']
        response_lower = response_text.lower()
        
        scores = []
        for i, opt_text in enumerate(options):
            words = [w.lower() for w in re.findall(r'\b\w{4,}\b', opt_text)]
            if not words:
                scores.append(0)
                continue
            matches = sum(1 for w in words if w in response_lower)
            score = matches / len(words) if words else 0
            scores.append(score)
        
        max_score = max(scores)
        if max_score < 0.7:
            return None
        
        sorted_scores = sorted(scores, reverse=True)
        if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1]) < 0.3:
            return None
        
        best_idx = scores.index(max_score)
        self.logger.debug(f"Content-matched option {letters[best_idx]} (score={max_score:.2f})")
        return letters[best_idx]
    
    def run_multi_year_test(self, dataset_path: str, years: List[int], num_repetitions: int):
        checkpoint = self.load_checkpoint()
        
        if not checkpoint.get("results"):
            checkpoint["results"] = {}
        if not checkpoint.get("consistency_data"):
            checkpoint["consistency_data"] = []
        if not checkpoint.get("metrics_data"):
            checkpoint["metrics_data"] = []
        
        try:
            for year in years:
                if checkpoint.get("current_year") and year < int(str(checkpoint["current_year"]).split("_")[0] if "_" in str(checkpoint["current_year"]) else checkpoint["current_year"]):
                    self.logger.info(f"Year {year} already completed, skipping")
                    continue
                
                self.logger.info(f"\n{'='*80}")
                self.logger.info(f"STARTING TEST FOR YEAR {year}")
                self.logger.info(f"{'='*80}\n")
                
                excel_file = os.path.join(dataset_path, f"{year}_answers_converted_[checked].xlsx")
                self.logger.info(f"Loading file: {excel_file}")
                df = pd.read_excel(excel_file)
                
                if str(year) not in checkpoint["results"]:
                    checkpoint["results"][str(year)] = []
                
                start_run = 0
                if checkpoint.get("current_year") == year:
                    start_run = checkpoint.get("current_run", 0)
                
                for run_idx in range(start_run, num_repetitions):
                    self.logger.info(f"\n{'='*60}")
                    self.logger.info(f"Year {year} - Run {run_idx + 1}/{num_repetitions}")
                    self.logger.info(f"{'='*60}")
                    
                    checkpoint["current_year"] = year
                    checkpoint["current_run"] = run_idx
                    
                    run_results = self._execute_run(df, year, run_idx, checkpoint)
                    
                    checkpoint["results"][str(year)].append(run_results["metrics"])
                    checkpoint["consistency_data"].append(run_results["consistency_row"])
                    checkpoint["metrics_data"].append(run_results["metrics_row"])
                    
                    checkpoint["current_question"] = 0
                    self.save_checkpoint(checkpoint)
                    
                    self.logger.info(f"Run {run_idx + 1} completed - Accuracy: {run_results['metrics']['accuracy']:.2f}%, "
                               f"Score: {run_results['metrics']['score']:.2f}")
            
            self._save_final_results(checkpoint)
            
            self.raw_responses_handle.close()
            self.logger.info(f"Raw responses saved to: {self.raw_responses_file}")
            self.logger.info(f"Total processed: {self.total_processed}, Extraction failures: {self.extraction_failures}")
            
            self.delete_checkpoint()
            
            self.logger.info("\n" + "="*80)
            self.logger.info("TEST COMPLETED SUCCESSFULLY")
            self.logger.info("="*80)
            
        except Exception as e:
            self.logger.error(f"Error during execution: {str(e)}")
            self.logger.info("Checkpoint saved. You can resume execution.")
            raise
    
    def _execute_run(self, df: pd.DataFrame, year: int, run_idx: int, checkpoint: Dict) -> Dict:
        start_run_time = time.time()
        
        correct_answers = 0
        wrong_answers = 0
        total_time = 0
        answers_list = []
        
        start_question = checkpoint.get("current_question", 0) if checkpoint.get("current_run") == run_idx else 0
        
        if start_question > 0 and checkpoint.get("partial_answers"):
            answers_list = checkpoint["partial_answers"]
            correct_answers = checkpoint.get("partial_correct", 0)
            wrong_answers = checkpoint.get("partial_wrong", 0)
            total_time = checkpoint.get("partial_time", 0)
        
        for idx in range(start_question, len(df)):
            row = df.iloc[idx]
            question_num = idx + 1
            
            if question_num % 10 == 0 or idx == start_question:
                self.logger.info(f"Processing question {question_num}/{len(df)}")
            
            question = row['Domanda']
            original_options = [
                row['Risposta A'],
                row['Risposta B'],
                row['Risposta C'],
                row['Risposta D'],
                row['Risposta E']
            ]
            
            shuffled_options, correct_answer, shuffle_map = self._shuffle_options(original_options)
            
            answer, elapsed_time = self.process_question(
                question, shuffled_options, idx, year, run_idx + 1, correct_answer
            )
            
            if answer == correct_answer:
                consistency_value = 1
            elif answer in ['A', 'B', 'C', 'D', 'E']:
                consistency_value = shuffle_map[answer]
            else:
                consistency_value = "ND"
            answers_list.append(consistency_value)
            total_time += elapsed_time
            
            if answer == correct_answer:
                correct_answers += 1
            elif answer in ['A', 'B', 'C', 'D', 'E']:
                wrong_answers += 1
            
            checkpoint["current_question"] = idx + 1
            checkpoint["partial_answers"] = answers_list
            checkpoint["partial_correct"] = correct_answers
            checkpoint["partial_wrong"] = wrong_answers
            checkpoint["partial_time"] = total_time
            
            if question_num % 10 == 0:
                self.save_checkpoint(checkpoint)
        
        total_questions = len(df)
        accuracy = (correct_answers / total_questions) * 100
        score = correct_answers - (wrong_answers * 0.25)
        time_per_question = total_time / total_questions
        consistency = self._calculate_consistency(answers_list)
        
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
                **{f"Domanda_{i+1}": ans for i, ans in enumerate(answers_list)}
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
        valid_answers = sum(1 for ans in answers if ans != "ND")
        return (valid_answers / len(answers)) * 100 if answers else 0
    
    def _save_final_results(self, checkpoint: Dict):
        safe_model_name = f"MedGemma_4B_{QUANT_NAME}"
        
        json_output = {
            "model": f"mradermacher/medgemma-1.5-4b-it-GGUF ({QUANT_NAME})",
            "quantization": QUANT_NAME,
            "test_completed": datetime.now().isoformat(),
            "total_processed": self.total_processed,
            "extraction_failures": self.extraction_failures,
            "years": checkpoint["results"]
        }
        
        json_file = f"results_{safe_model_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
        self.logger.info(f"JSON saved: {json_file}")
        
        consistency_df = pd.DataFrame(checkpoint["consistency_data"])
        consistency_file = f"consistency_{safe_model_name}.xlsx"
        consistency_df.to_excel(consistency_file, index=False)
        self.logger.info(f"Consistency Excel saved: {consistency_file}")
        
        metrics_df = pd.DataFrame(checkpoint["metrics_data"])
        metrics_file = f"metrics_{safe_model_name}.xlsx"
        metrics_df.to_excel(metrics_file, index=False)
        self.logger.info(f"Metrics Excel saved: {metrics_file}")


def main():
    print("="*80)
    print(f"MEDGEMMA 1.5 4B IT GGUF ({QUANT_NAME}) - MEDICAL EXAM TESTER")
    print("="*80)
    print()
    print(f"Model: mradermacher/medgemma-1.5-4b-it-GGUF")
    print(f"Quantization: {QUANT_NAME}")
    print(f"Model path: {MODEL_PATH}")
    print()
    
    dataset_path = "Dataset"
    years = [2020, 2021, 2022, 2023, 2024]
    num_repetitions = 50
    
    print(f"Dataset path: {dataset_path}")
    print(f"Years: {years}")
    print(f"Repetitions per year: {num_repetitions}")
    print(f"Total runs: {len(years) * num_repetitions}")
    print()
    
    input("Press ENTER to start the test (or Ctrl+C to cancel)...")
    print()
    
    try:
        tester = MedGemmaGGUFMedicalExamTester()
        tester.run_multi_year_test(dataset_path, years, num_repetitions)
        
        print("\nTEST COMPLETED!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user. Checkpoint saved.")
        print("You can resume the test by running the script again.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
