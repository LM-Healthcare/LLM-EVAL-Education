#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import logging
import pandas as pd
import getpass
from typing import Dict, List, Any, Optional, Tuple
import requests
import random
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mistral_medical_exam_test_v2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MistralMedicalExamTester:
    """
    Classe per testare il modello Mistral Large su domande d'esame di specializzazione medica
    Versione 2: Multi-anno con checkpoint e output Excel
    """
    
    def __init__(self, api_key: str):
        """
        Inizializza il tester con l'API key di Mistral
        
        Args:
            api_key: API key di Mistral
        """
        self.api_key = api_key
        self.model_name = "mistral-large-2512"
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        self.checkpoint_file = "checkpoint_mistral.json"
        self.raw_responses_file = "raw_responses_mistral.jsonl"
        logger.info(f"Initialized tester with model {self.model_name}")
        
    def load_checkpoint(self) -> Dict:
        """Load checkpoint if it exists"""
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
                logger.info(f"Checkpoint loaded: Year {checkpoint.get('current_year')}, Run {checkpoint.get('current_run')}, Question {checkpoint.get('current_question')}")
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
                logger.info("Checkpoint deleted")
            except PermissionError as e:
                logger.warning(f"Could not delete checkpoint: {e}. You can delete it manually.")
    
    def _shuffle_options(self, options: List[str]) -> Tuple[List[str], str, Dict[str, str]]:
        """
        Shuffle options randomly and return new correct answer letter + mapping.
        """
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
        
    def process_question(self, question: str, options: List[str], correct_answer: str = 'A', year: int = 0, run: int = 0, question_idx: int = 0) -> Tuple[str, float]:
        """
        Processa una singola domanda con il modello Mistral
        
        Args:
            question: Testo della domanda
            options: Lista delle opzioni di risposta
        
        Returns:
            Tuple con la lettera della risposta (A-E o ND) e il tempo di elaborazione
        """
        try:
            start_time = time.time()
            
            # Build the prompt
            prompt = f"""
You are a highly skilled medical doctor taking a specialization entrance exam in Italy.
Carefully analyze the following question and possible answers in Italian.
Choose the correct answer among the provided options.

QUESTION:
{question}

OPTIONS:
A: {options[0]}
B: {options[1]}
C: {options[2]}
D: {options[3]}
E: {options[4]}

IMPORTANT: Provide your answer in this exact format:
[LETTER]: [content of the chosen option]

For example: "A: Fever and chills"

Do not provide additional explanations.
"""
            
            # Prepare the request
            data = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert physician taking a medical specialization exam in Italy. Carefully analyze the question and each possible answer. Respond ONLY with the letter corresponding to the correct answer (A, B, C, D, or E). Do not provide explanations or additional text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,  # Small temperature for slight variability
                "max_tokens": 50
            }
            
            # API call with retry
            max_retries = 5
            base_delay = 2
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(self.api_url, headers=self.headers, json=data)
                    
                    if response.status_code == 200:
                        break
                    
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', base_delay * (2 ** attempt)))
                        jitter = random.uniform(0, 0.1 * retry_after)
                        wait_time = retry_after + jitter
                        
                        logger.warning(f"Rate limit reached (429). Waiting {wait_time:.2f} seconds before retrying. Attempt {attempt+1}/{max_retries}")
                        time.sleep(wait_time)
                        continue
                    
                    logger.error(f"API call error: {response.status_code} - {response.text}")
                    raise Exception(f"API error: {response.status_code}")
                    
                except Exception as e:
                    wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"API call error: {str(e)}. Waiting {wait_time:.2f} seconds before retrying. Attempt {attempt+1}/{max_retries}")
                    
                    if attempt == max_retries - 1:
                        logger.error(f"Maximum number of retries reached. Unable to complete request.")
                        raise
                    
                    time.sleep(wait_time)
            
            if response.status_code != 200:
                logger.error(f"API call error after {max_retries} attempts: {response.status_code} - {response.text}")
                raise Exception(f"API error: {response.status_code}")
            
            # Extract the answer letter
            response_json = response.json()
            response_text = response_json["choices"][0]["message"]["content"].strip()
            logger.debug(f"Original model response: '{response_text}'")
            
            # Letter extraction
            answer_letter = self._extract_letter(response_text)
            
            # Save raw response for analysis (full format like Meditron3)
            elapsed_time = time.time() - start_time
            raw_entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "year": year,
                "run": run,
                "question_idx": question_idx,
                "question": question,
                "options": {
                    "A": options[0],
                    "B": options[1],
                    "C": options[2],
                    "D": options[3],
                    "E": options[4]
                },
                "correct_answer": correct_answer,
                "raw_response": response_text,
                "extracted_letter": answer_letter,
                "is_correct": answer_letter == correct_answer,
                "processing_time_seconds": elapsed_time
            }
            with open(self.raw_responses_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(raw_entry, ensure_ascii=False) + '\n')
            
            # Delay to respect rate limit (Mistral: 20 req/min = 3s between requests)
            time.sleep(3 + random.uniform(0, 1))
                
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            answer_letter = "ND"
            
        elapsed_time = time.time() - start_time
        return answer_letter, elapsed_time
    
    def _extract_letter(self, response_text: str) -> str:
        """Extract the letter from the model response"""
        response_text = response_text.strip()
        
        # 0. PRIORITY: Pattern "A: content" (our requested format)
        letter_content_pattern = r'^([A-E])\s*[:\-]'
        match = re.match(letter_content_pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # 1. Search for "ANSWER: X" pattern
        answer_pattern = r'ANSWER\s*:\s*([A-E])\b'
        match = re.search(answer_pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # 2. Search for "The final answer is X" pattern
        final_pattern = r'(?:The\s+)?final\s+answer\s+is\s+([A-E])\b'
        match = re.search(final_pattern, response_text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        # 3. Search for explicit patterns
        patterns = [
            r"(?:answer|option|choice|letter)[\s:]*(A|B|C|D|E)",
            r"(?:the|I choose|I select)[\s:]*(A|B|C|D|E)",
            r"^(A|B|C|D|E)$"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                return matches[0].upper()
        
        # 4. Search for isolated letters
        for letter in ['A', 'B', 'C', 'D', 'E']:
            if letter in response_text or letter.lower() in response_text.lower():
                return letter
        
        # 5. Search for numerical references
        number_map = {'1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'E'}
        for text, letter in number_map.items():
            if text in response_text.lower():
                return letter
        
        logger.warning(f"Could not extract a letter from response: '{response_text}'")
        return "ND"
    
    def run_multi_year_test(self, dataset_path: str, years: List[int], num_repetitions: int):
        """
        Esegue il test su più anni con checkpoint e output ottimizzati
        
        Args:
            dataset_path: Percorso della cartella con i file Excel
            years: Lista degli anni da testare
            num_repetitions: Numero di ripetizioni per ogni anno
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
                # Skip if we already completed this year
                if checkpoint.get("current_year") and year < int(str(checkpoint["current_year"]).split("_")[0] if "_" in str(checkpoint["current_year"]) else checkpoint["current_year"]):
                    logger.info(f"Year {year} already completed, skipping")
                    continue
                
                logger.info(f"\n{'='*80}")
                logger.info(f"STARTING TEST FOR YEAR {year}")
                logger.info(f"{'='*80}\n")
                
                # Load the Excel file
                excel_file = os.path.join(dataset_path, f"{year}_answers_converted_[checked].xlsx")
                logger.info(f"Loading file: {excel_file}")
                df = pd.read_excel(excel_file)
                
                # Initialize results for this year
                if str(year) not in checkpoint["results"]:
                    checkpoint["results"][str(year)] = []
                
                # Determine which run to resume from
                start_run = 0
                if checkpoint.get("current_year") == year:
                    start_run = checkpoint.get("current_run", 0)
                
                # Execute repetitions
                for run_idx in range(start_run, num_repetitions):
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Year {year} - Run {run_idx + 1}/{num_repetitions}")
                    logger.info(f"{'='*60}")
                    
                    checkpoint["current_year"] = year
                    checkpoint["current_run"] = run_idx
                    
                    run_results = self._execute_run(df, year, run_idx, checkpoint)
                    
                    # Save results
                    checkpoint["results"][str(year)].append(run_results["metrics"])
                    checkpoint["consistency_data"].append(run_results["consistency_row"])
                    checkpoint["metrics_data"].append(run_results["metrics_row"])
                    
                    # Reset current question for next run
                    checkpoint["current_question"] = 0
                    
                    # Save checkpoint
                    self.save_checkpoint(checkpoint)
                    
                    logger.info(f"Run {run_idx + 1} completed - Accuracy: {run_results['metrics']['accuracy']:.2f}%, Score: {run_results['metrics']['score']:.2f}")
            
            # Save final results
            self._save_final_results(checkpoint)
            
            # Delete checkpoint
            self.delete_checkpoint()
            
            logger.info("\n" + "="*80)
            logger.info("TEST COMPLETED SUCCESSFULLY")
            logger.info("="*80)
            
        except Exception as e:
            logger.error(f"Error during execution: {str(e)}")
            logger.info("Checkpoint saved. You can resume execution.")
            raise
    
    def _execute_run(self, df: pd.DataFrame, year: int, run_idx: int, checkpoint: Dict) -> Dict:
        """Execute a single test run"""
        start_run_time = time.time()
        
        # Initialize metrics
        correct_answers = 0
        wrong_answers = 0
        total_time = 0
        answers_list = []
        
        # Determine which question to resume from
        start_question = checkpoint.get("current_question", 0) if checkpoint.get("current_run") == run_idx else 0
        
        # If resuming, load previous answers
        if start_question > 0 and checkpoint.get("partial_answers"):
            answers_list = checkpoint["partial_answers"]
            correct_answers = checkpoint.get("partial_correct", 0)
            wrong_answers = checkpoint.get("partial_wrong", 0)
            total_time = checkpoint.get("partial_time", 0)
        
        # Process each question
        for idx in range(start_question, len(df)):
            row = df.iloc[idx]
            question_num = idx + 1
            
            logger.info(f"Processing question {question_num}/{len(df)}")
            
            # Extract question and answers
            question = row['Domanda']
            original_options = [
                row['Risposta A'],  # Index 0 = correct answer
                row['Risposta B'],
                row['Risposta C'],
                row['Risposta D'],
                row['Risposta E']
            ]
            
            # Shuffle options to eliminate position bias
            shuffled_options, correct_answer, shuffle_map = self._shuffle_options(original_options)
            
            # Process the question
            answer, elapsed_time = self.process_question(question, shuffled_options, correct_answer, year, run_idx + 1, question_num)
            
            # Track for consistency: 1 = correct, original letter (B-E) = wrong, ND = no answer
            if answer == correct_answer:
                consistency_value = 1
            elif answer in ['A', 'B', 'C', 'D', 'E']:
                consistency_value = shuffle_map[answer]
            else:
                consistency_value = "ND"
            answers_list.append(consistency_value)
            total_time += elapsed_time
            
            # Evaluate correctness
            if answer == correct_answer:
                correct_answers += 1
            elif answer in ['A', 'B', 'C', 'D', 'E']:
                wrong_answers += 1
            
            # Save partial state in checkpoint
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
        consistency = self._calculate_consistency_single(answers_list)
        
        # Clean partial data from checkpoint
        checkpoint.pop("partial_answers", None)
        checkpoint.pop("partial_correct", None)
        checkpoint.pop("partial_wrong", None)
        checkpoint.pop("partial_time", None)
        
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
    
    def _calculate_consistency_single(self, answers: List) -> float:
        """
        Calcola la consistency per una singola run
        Returns percentage of valid answers (not ND)
        """
        valid_answers = sum(1 for ans in answers if ans != "ND")
        return (valid_answers / len(answers)) * 100 if answers else 0
    
    def _save_final_results(self, checkpoint: Dict):
        """Salva i risultati finali in JSON e Excel"""
        
        # 1. Salva JSON
        json_output = {
            "model": self.model_name,
            "test_completed": time.strftime("%Y-%m-%d %H:%M:%S"),
            "years": checkpoint["results"]
        }
        
        json_file = f"results_mistral.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_output, f, ensure_ascii=False, indent=2)
        logger.info(f"JSON salvato: {json_file}")
        
        # 2. Salva Excel Consistency
        consistency_df = pd.DataFrame(checkpoint["consistency_data"])
        consistency_file = f"consistency_mistral.xlsx"
        consistency_df.to_excel(consistency_file, index=False)
        logger.info(f"Excel Consistency salvato: {consistency_file}")
        
        # 3. Salva Excel Metrics
        metrics_df = pd.DataFrame(checkpoint["metrics_data"])
        metrics_file = f"metrics_mistral.xlsx"
        metrics_df.to_excel(metrics_file, index=False)
        logger.info(f"Excel Metrics salvato: {metrics_file}")

def main():
    """Punto di ingresso principale"""
    # Richiedi l'API key di Mistral
    api_key = getpass.getpass("Inserisci la tua API key di Mistral: ")
    
    # Configurazione
    dataset_path = "Dataset"
    years = [2020, 2021, 2022, 2023, 2024]
    num_repetitions = 50
    
    # Inizializza il tester
    logger.info("="*80)
    logger.info("INIZIALIZZAZIONE TEST MISTRAL LARGE - MULTI YEAR EVALUATION")
    logger.info("="*80)
    logger.info(f"Anni da testare: {years}")
    logger.info(f"Ripetizioni per anno: {num_repetitions}")
    logger.info(f"Totale run: {len(years) * num_repetitions}")
    
    tester = MistralMedicalExamTester(api_key)
    
    # Esegui il test
    tester.run_multi_year_test(dataset_path, years, num_repetitions)
    
    logger.info("\nTEST COMPLETATO!")

if __name__ == "__main__":
    main()
