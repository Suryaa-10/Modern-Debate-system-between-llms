import sys
import os
import asyncio

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.dataset_evaluator import DatasetLoader, AnswerExtractor, BenchmarkEvaluator
from evaluation.visualization import DebateVisualizer

async def main():
    print("--- Testing AnswerExtractor ---")
    num_ans = AnswerExtractor.extract_numeric("The answer is #### 42")
    assert num_ans == "42", f"Expected '42', got '{num_ans}'"
    opt_ans = AnswerExtractor.extract_option("Therefore, the correct answer is B.")
    assert opt_ans == "B", f"Expected 'B', got '{opt_ans}'"
    print("AnswerExtractor tests passed!")

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
