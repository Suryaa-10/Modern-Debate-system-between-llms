import sys
import os
import asyncio
import json

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.dataset_evaluator import DatasetLoader, AnswerExtractor, BenchmarkEvaluator
from evaluation.visualization import DebateVisualizer
from models.model_manager import extract_and_repair_json
from debate.state import DebateState
from debate.engine import DebateEngine
from debate.consensus import ConsensusEngine

async def main():
    print("--- Testing AnswerExtractor ---")
    num_ans = AnswerExtractor.extract_numeric("The answer is #### 42")
    assert num_ans == "42", f"Expected '42', got '{num_ans}'"
    opt_ans = AnswerExtractor.extract_option("Therefore, the correct answer is B.")
    assert opt_ans == "B", f"Expected 'B', got '{opt_ans}'"
    print("AnswerExtractor tests passed!")

    print("\n--- Testing JSON Extraction & Repair Loop ---")
    malformed_json_1 = "Here is the response: ```json\n{\"reasoning\": \"Step 1\", \"final_answer\": \"B\", \"confidence_score\": 0.95,}``` commentary at end."
    repaired_1 = extract_and_repair_json(malformed_json_1)
    assert repaired_1 is not None, "Failed to repair JSON with trailing comma and markdown."
    assert repaired_1.get("final_answer") == "B", f"Expected 'B', got {repaired_1.get('final_answer')}"

    malformed_json_2 = "Analysis complete. {\"agent_name\": \"agent_1\", \"reasoning\": \"valid logic\", \"final_answer\": \"42\", \"confidence_score\": 0.9} extra text"
    repaired_2 = extract_and_repair_json(malformed_json_2)
    assert repaired_2 is not None, "Failed to repair embedded JSON block."
    assert repaired_2.get("final_answer") == "42", f"Expected '42', got {repaired_2.get('final_answer')}"
    print("JSON Extraction & Repair tests passed!")

    print("\n--- Testing Confidence-Weighted Majority Tally ---")
    state = DebateState(query="Test Query", max_rounds=2)
    state.record_round_data({
        "agent_1": {"final_answer": "B", "confidence_score": 0.9, "reasoning": "Reason A"},
        "agent_2": {"final_answer": "A", "confidence_score": 0.6, "reasoning": "Reason B"},
        "agent_3": {"final_answer": "B", "confidence_score": 0.8, "reasoning": "Reason C"}
    })
    c_engine = ConsensusEngine(manager=None)
    majority_info = c_engine._compute_confidence_weighted_majority(state)
    assert majority_info["top_answer"] == "B", f"Expected 'B', got {majority_info['top_answer']}"
    assert abs(majority_info["total_weight"] - 1.7) < 1e-3, f"Expected 1.7, got {majority_info['total_weight']}"
    print("Confidence-Weighted Majority tests passed!")

    print("\n--- Testing Dynamic Early-Exit Routing ---")
    state_exit = DebateState(query="Test Query", max_rounds=5)
    d_engine = DebateEngine(manager=None)
    
    # Simulate round 1 with full agreement
    round_data_unanimous = {
        "agent_1": {"final_answer": "B", "confidence_score": 0.95},
        "agent_2": {"final_answer": "B", "confidence_score": 0.90},
        "agent_3": {"final_answer": "B", "confidence_score": 0.85}
    }
    d_engine._check_early_exit(state_exit, round_data_unanimous)
    assert state_exit.is_completed is True, "Early exit flag should be True"
    assert state_exit.early_exit is True, "Early exit tripwire should be triggered"
    print("Dynamic Early-Exit Routing tests passed!")

    print("\n--- Testing DatasetLoader ---")
    suite = DatasetLoader.get_benchmark_suite()
    print(f"Loaded datasets: {list(suite.keys())}")
    assert len(suite) >= 5, "Dataset suite incomplete"
    print("DatasetLoader tests passed!")

    # Create dynamic dataset evaluation results structure
    sample_eval_results = {
        "Biographies": {
            "Single Agent": (66.7, 4.1),
            "Single Agent (Reflection)": (66.7, 4.1),
            "Multiagent (Majority)": (100.0, 0.0),
            "Multiagent (Debate)": (100.0, 0.0)
        },
        "MMLU": {
            "Single Agent": (66.7, 4.1),
            "Single Agent (Reflection)": (66.7, 4.1),
            "Multiagent (Majority)": (66.7, 4.1),
            "Multiagent (Debate)": (100.0, 0.0)
        },
        "Arithmetic": {
            "Single Agent": (66.7, 4.1),
            "Single Agent (Reflection)": (100.0, 0.0),
            "Multiagent (Majority)": (100.0, 0.0),
            "Multiagent (Debate)": (100.0, 0.0)
        },
        "Grade School Math": {
            "Single Agent": (50.0, 5.0),
            "Single Agent (Reflection)": (50.0, 5.0),
            "Multiagent (Majority)": (100.0, 0.0),
            "Multiagent (Debate)": (100.0, 0.0)
        }
    }

    print("\n--- Testing Visualization Generation from Dataset Evaluation ---")
    viz = DebateVisualizer()
    fig1 = viz.plot_accuracy_comparison(sample_eval_results)
    chart_path = os.path.join("results", "graphs", "accuracy_comparison_barchart.png")
    assert os.path.exists(chart_path), f"Bar chart image missing at {chart_path}"
    print(f"Bar chart successfully generated at: {chart_path}")

    df, fig2 = viz.generate_paper_styled_table(sample_eval_results)
    table_path = os.path.join("results", "graphs", "academic_comparison_table.png")
    assert os.path.exists(table_path), f"Academic table image missing at {table_path}"
    print(f"Academic table image successfully generated at: {table_path}")
    print("\nTable DataFrame head:")
    print(df.to_string())

    print("\nAll verification tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
