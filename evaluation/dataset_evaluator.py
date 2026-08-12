import sys
import os
import json
import re
import math
import asyncio
from typing import Dict, Any, List, Tuple, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.model_manager import ModelManager
from execution.query_execution import ExecutionController

class AnswerExtractor:
    """Extracts standardized answers from LLM text responses based on dataset type."""
    
    @staticmethod
    def extract_numeric(text: str) -> Optional[str]:
        """Extracts numerical answer from CoT text (e.g. GSM8K format #### 42)."""
        if not text:
            return None
        # Check for explicit GSM8K output separator ####
        if "####" in text:
            ans = text.split("####")[-1].strip()
            # Clean non-digit characters except negative sign and decimal point
            ans_clean = re.sub(r'[^\d.-]', '', ans)
            if ans_clean:
                return ans_clean
                
        # Look for numbers in latex \boxed{123}
        boxed_match = re.search(r'\\boxed\{([0-9.]+)\}', text)
        if boxed_match:
            return boxed_match.group(1).strip()
            
        # Fallback: search for final standing number in text
        numbers = re.findall(r'-?\d+(?:\.\d+)?', text)
        if numbers:
            return numbers[-1].strip()
            
        return text.strip()

    @staticmethod
    def extract_option(text: str) -> Optional[str]:
        """Extracts option letter (A, B, C, D) for multiple choice questions."""
        if not text:
            return None
            
        # Patterns like "Answer: A", "The answer is (B)", "**C**"
        patterns = [
            r'(?:correct\s+)?answer\s*(?:is|:)?\s*[\*\(]*([A-D])[\*\)]*',
            r'option\s*[\*\(]*([A-D])[\*\)]*',
            r'\b([A-D])\b'
        ]
        
        for pat in patterns:
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                return match.group(1).upper()
                
        return text.strip()[:10]

    @staticmethod
    def is_correct(predicted: str, target: str, dataset_type: str = "math") -> bool:
        """Determines if predicted output matches ground truth target."""
        if not predicted or not target:
            return False
            
        target_str = str(target).strip()
        
        if dataset_type in ["math", "arithmetic"]:
            pred_num = AnswerExtractor.extract_numeric(predicted)
            targ_num = AnswerExtractor.extract_numeric(target_str)
            if pred_num and targ_num:
                try:
                    return math.isclose(float(pred_num), float(targ_num), rel_tol=1e-3, abs_tol=1e-3)
                except ValueError:
                    return pred_num == targ_num
        elif dataset_type in ["mmlu", "multiple_choice"]:
            pred_opt = AnswerExtractor.extract_option(predicted)
            targ_opt = AnswerExtractor.extract_option(target_str)
            return pred_opt == targ_opt
            
        # Default string match / normalization
        pred_norm = re.sub(r'\s+', ' ', str(predicted).lower().strip())
        targ_norm = re.sub(r'\s+', ' ', target_str.lower().strip())
        return targ_norm in pred_norm or pred_norm == targ_norm


class DatasetLoader:
    """Loads benchmark datasets (GSM8K, MMLU, Arithmetic, Biographies, Chess)."""
    
    @staticmethod
    def get_gsm8k_samples(limit: int = 5) -> List[Dict[str, Any]]:
        """Loads sample problems from local GSM8K test file."""
        gsm_path = os.path.join("Dataset", "grade-school-math", "grade_school_math", "data", "test.jsonl")
        samples = []
        if os.path.exists(gsm_path):
            with open(gsm_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        raw_ans = data.get("answer", "")
                        target_val = raw_ans.split("####")[-1].strip() if "####" in raw_ans else raw_ans
                        samples.append({
                            "question": data.get("question"),
                            "target": target_val,
                            "dataset": "Grade School Math",
                            "type": "math"
                        })
                        if len(samples) >= limit:
                            break
        return samples

    @staticmethod
    def get_benchmark_suite() -> Dict[str, List[Dict[str, Any]]]:
        """Provides full benchmark evaluation suite matching research paper categories."""
        suite = {}
        
        # 1. Biographies
        suite["Biographies"] = [
            {
                "question": "Which university did Albert Einstein earn his PhD from in 1905?",
                "target": "University of Zurich",
                "dataset": "Biographies",
                "type": "text"
            },
            {
                "question": "Who was the primary architect of the Indian Constitution?",
                "target": "B. R. Ambedkar",
                "dataset": "Biographies",
                "type": "text"
            },
            {
                "question": "In what year was Marie Curie awarded her first Nobel Prize in Physics?",
                "target": "1903",
                "dataset": "Biographies",
                "type": "math"
            }
        ]
        
        # 2. MMLU (Multiple Choice)
        suite["MMLU"] = [
            {
                "question": "Which of the following data structures operates on a Last In, First Out (LIFO) basis?\nA) Queue\nB) Stack\nC) Array\nD) Linked List",
                "target": "B",
                "dataset": "MMLU",
                "type": "mmlu"
            },
            {
                "question": "What is the legal term for a written statement confirmed by oath or affirmation for use as evidence in court?\nA) Subpoena\nB) Affidavit\nC) Injunction\nD) Indictment",
                "target": "B",
                "dataset": "MMLU",
                "type": "mmlu"
            },
            {
                "question": "Which law of thermodynamics states that entropy of an isolated system always increases over time?\nA) Zeroth Law\nB) First Law\nC) Second Law\nD) Third Law",
                "target": "C",
                "dataset": "MMLU",
                "type": "mmlu"
            }
        ]
        
        # 3. Chess Move Validity
        suite["Chess Move Validity"] = [
            {
                "question": "In standard chess from the starting position, is the move e2 to e4 a valid chess move? (Answer Yes or No)",
                "target": "Yes",
                "dataset": "Chess Move Validity",
                "type": "text"
            },
            {
                "question": "Can a Knight move 3 squares diagonally in a single move in standard chess? (Answer Yes or No)",
                "target": "No",
                "dataset": "Chess Move Validity",
                "type": "text"
            },
            {
                "question": "Is castling allowed if the King has already moved previously in the game? (Answer Yes or No)",
                "target": "No",
                "dataset": "Chess Move Validity",
                "type": "text"
            }
        ]
        
        # 4. Arithmetic
        suite["Arithmetic"] = [
            {
                "question": "What is 17 * 14 - 39 + 112 / 4?",
                "target": "227",
                "dataset": "Arithmetic",
                "type": "arithmetic"
            },
            {
                "question": "A store owner bought an item for $50, marked it up by 40%, then offered a 20% discount on marked price, and added 10% tax on discounted price. What is the final price paid?",
                "target": "61.6",
                "dataset": "Arithmetic",
                "type": "arithmetic"
            },
            {
                "question": "What is 15% of 840 plus 64?",
                "target": "190",
                "dataset": "Arithmetic",
                "type": "arithmetic"
            }
        ]
        
        # 5. Grade School Math (GSM8K & Logic Traps)
        gsm_samples = DatasetLoader.get_gsm8k_samples(limit=3)
        if gsm_samples:
            suite["Grade School Math"] = gsm_samples
        else:
            suite["Grade School Math"] = [
                {
                    "question": "If 5 machines take 5 minutes to make 5 widgets, how many minutes does it take 100 machines to make 100 widgets?",
                    "target": "5",
                    "dataset": "Grade School Math",
                    "type": "math"
                },
                {
                    "question": "Train A leaves station at 60 mph. 2 hours later, Train B leaves same station at 90 mph. How many miles from station will Train B catch Train A?",
                    "target": "360",
                    "dataset": "Grade School Math",
                    "type": "math"
                }
            ]

        # 6. Chess Move Optimality
        suite["Chess Move Optimality"] = [
            {
                "question": "White has Queen on d1, Black King on e8 with no defending pieces. White plays Qd8#. Is this move checkmate? (Answer Yes or No)",
                "target": "Yes",
                "dataset": "Chess Move Optimality",
                "type": "text"
            },
            {
                "question": "If a player can capture the opponent's undefended Queen for free or capture a pawn, which move is optimal?",
                "target": "Queen",
                "dataset": "Chess Move Optimality",
                "type": "text"
            }
        ]
        
        return suite


class BenchmarkEvaluator:
    """Evaluates multi-agent debate strategies across benchmark datasets."""
    
    def __init__(self, manager: Optional[ModelManager] = None):
        self.manager = manager or ModelManager()
        self.controller = ExecutionController(self.manager)
        
    async def evaluate_single_agent(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        """Direct query execution using single agent (agent_1)."""
        prompt = f"Solve the following question clearly and end with your final concise answer:\n{sample['question']}"
        try:
            raw_resp = await self.manager.generate_response("agent_1", [{"role": "user", "content": prompt}])
            is_correct = AnswerExtractor.is_correct(raw_resp, sample['target'], sample['type'])
            return is_correct, raw_resp
        except Exception as e:
            print(f"[Error Single Agent] {e}")
            return False, f"Error: {e}"

    async def evaluate_single_agent_reflection(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        """Single agent self-reflection (round 1 reasoning + round 2 self-critique)."""
        prompt1 = f"Solve the following question step-by-step:\n{sample['question']}"
        try:
            resp1 = await self.manager.generate_response("agent_1", [{"role": "user", "content": prompt1}])
            prompt2 = [
                {"role": "user", "content": prompt1},
                {"role": "assistant", "content": resp1},
                {"role": "user", "content": "Review your previous steps carefully. Check for any arithmetic or logical mistakes, and state your final definitive answer."}
            ]
            resp2 = await self.manager.generate_response("agent_1", prompt2)
            is_correct = AnswerExtractor.is_correct(resp2, sample['target'], sample['type'])
            return is_correct, resp2
        except Exception as e:
            print(f"[Error Reflection] {e}")
            return False, f"Error: {e}"

    async def evaluate_multiagent_majority(self, sample: Dict[str, Any]) -> Tuple[bool, str]:
        """Confidence-weighted majority voting among 3 independent agents."""
        prompt = f"Solve the following question concisely and state your final answer:\n{sample['question']}"
        agents = ["agent_1", "agent_2", "agent_3"]
        
        async def fetch(agent):
            try:
                from schemas.response_schema import ArgumentSchema
                resp_json = await self.manager.generate_response(
                    agent, 
                    [{"role": "user", "content": prompt}],
                    response_schema=ArgumentSchema.model_json_schema()
                )
                data = json.loads(resp_json)
                return data.get("final_answer", ""), float(data.get("confidence_score", 0.8))
            except Exception:
                try:
                    raw = await self.manager.generate_response(agent, [{"role": "user", "content": prompt}])
                    return raw, 0.5
                except Exception:
                    return "", 0.0

        results = await asyncio.gather(*[fetch(a) for a in agents])
        
        # Aggregate confidence-weighted votes
        weights = {}
        conf_sums = {}
        counts = {}
        for raw_ans, conf in results:
            if not raw_ans:
                continue
            if sample['type'] in ["math", "arithmetic"]:
                extracted = AnswerExtractor.extract_numeric(raw_ans)
            elif sample['type'] in ["mmlu", "multiple_choice"]:
                extracted = AnswerExtractor.extract_option(raw_ans)
            else:
                extracted = raw_ans.strip()

            if extracted:
                weights[extracted] = weights.get(extracted, 0.0) + conf
                conf_sums[extracted] = conf_sums.get(extracted, 0.0) + conf
                counts[extracted] = counts.get(extracted, 0) + 1

        if weights:
            # Sort by total weight, then by average confidence
            best_ans = max(weights.keys(), key=lambda k: (weights[k], conf_sums[k] / counts[k]))
        else:
            best_ans = ""

        is_correct = AnswerExtractor.is_correct(best_ans, sample['target'], sample['type'])
        return is_correct, str(best_ans)

    async def evaluate_multiagent_debate(self, sample: Dict[str, Any], max_rounds: int = 2) -> Tuple[bool, str]:
        """Multi-agent iterative debate and consensus synthesis with target answer extraction."""
        try:
            debate_result = await self.controller.run_full_debate(sample['question'], max_rounds=max_rounds)
            final_consensus = debate_result.get("final_consensus", {})
            raw_conclusion = final_consensus.get("final_conclusion") or final_consensus.get("final_answer") or ""
            
            # Extract target format prior to scoring
            if sample['type'] in ["math", "arithmetic"]:
                conclusion = AnswerExtractor.extract_numeric(raw_conclusion) or raw_conclusion
            elif sample['type'] in ["mmlu", "multiple_choice"]:
                conclusion = AnswerExtractor.extract_option(raw_conclusion) or raw_conclusion
            else:
                conclusion = raw_conclusion

            is_correct = AnswerExtractor.is_correct(conclusion, sample['target'], sample['type'])
            return is_correct, str(conclusion)
        except Exception as e:
            print(f"[Error Debate] {e}")
            return False, f"Error: {e}"

    async def run_full_suite_benchmark(self, suite: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """
        Runs evaluation across all datasets and strategies.
        Returns dataset -> strategy -> (accuracy_pct, std_dev)
        """
        if suite is None:
            suite = DatasetLoader.get_benchmark_suite()
            
        results = {}
        
        for dataset_name, samples in suite.items():
            print(f"\n========================================")
            print(f"Evaluating Benchmark Dataset: {dataset_name} ({len(samples)} samples)")
            print(f"========================================")
            
            strat_accuracies = {
                "Single Agent": [],
                "Single Agent (Reflection)": [],
                "Multiagent (Majority)": [],
                "Multiagent (Debate)": []
            }
            
            for i, item in enumerate(samples):
                print(f"Sample [{i+1}/{len(samples)}]: {item['question'][:60]}...")
                
                # Single Agent
                sa_corr, _ = await self.evaluate_single_agent(item)
                strat_accuracies["Single Agent"].append(1.0 if sa_corr else 0.0)
                
                # Single Agent (Reflection)
                ref_corr, _ = await self.evaluate_single_agent_reflection(item)
                strat_accuracies["Single Agent (Reflection)"].append(1.0 if ref_corr else 0.0)
                
                # Multiagent (Majority)
                maj_corr, _ = await self.evaluate_multiagent_majority(item)
                strat_accuracies["Multiagent (Majority)"].append(1.0 if maj_corr else 0.0)
                
                # Multiagent (Debate)
                deb_corr, _ = await self.evaluate_multiagent_debate(item)
                strat_accuracies["Multiagent (Debate)"].append(1.0 if deb_corr else 0.0)

            dataset_metrics = {}
            for strat, vals in strat_accuracies.items():
                acc = (sum(vals) / len(vals)) * 100.0 if vals else 0.0
                # Compute std dev (or standard error)
                variance = sum((x * 100.0 - acc) ** 2 for x in vals) / len(vals) if len(vals) > 1 else 0.0
                std_dev = math.sqrt(variance)
                dataset_metrics[strat] = (round(acc, 1), round(std_dev, 1))
                
            results[dataset_name] = dataset_metrics
            
        return results
