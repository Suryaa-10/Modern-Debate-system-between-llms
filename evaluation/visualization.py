import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional

class DebateVisualizer:
    """Provides visual formatting for dataset debate evaluations (Dual Bar Chart & Academic Paper Table)."""
    
    def __init__(self, output_dir: str = "results/graphs"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def display_metrics(self, metrics: Dict[str, Any]):
        """Prints terminal-based evaluation metrics summary."""
        print("\n=== DEBATE EVALUATION METRICS ===")
        print(f"Final Answer:        {metrics.get('final_answer', 'N/A')}")
        print(f"Confidence Score:    {metrics.get('confidence_score', 0):.2f}")
        print(f"Total Rounds:        {metrics.get('total_rounds', 0)}")
        
        print("\n--- Agent Contributions ---")
        for agent, count in metrics.get('agent_participation', {}).items():
            print(f"{agent}: {count} rounds participated")
        print("===================================\n")

    def plot_accuracy_comparison(
        self, 
        benchmark_results: Dict[str, Dict[str, Tuple[float, float]]],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Generates the Grouped Bar Chart comparing Single Agent vs Multi-Agent Debate accuracy
        based strictly on the user's dataset evaluation results.
        
        Args:
            benchmark_results: Dict mapping dataset name -> strategy -> (accuracy_pct, std_dev)
            save_path: Optional path to save output image
        """
        if not benchmark_results:
            raise ValueError("benchmark_results dictionary cannot be empty. Run dataset evaluation first.")

        data = {}
        for ds, strats in benchmark_results.items():
            sa_acc = strats.get("Single Agent", (0.0, 0.0))[0]
            # Match either "Multiagent (Debate)" or "Multi-Agent Debate"
            deb_acc = strats.get("Multiagent (Debate)", strats.get("Multi-Agent Debate", (0.0, 0.0)))[0]
            data[ds] = {"Single Agent": sa_acc, "Multi-Agent Debate": deb_acc}

        datasets = list(data.keys())
        single_agent_accs = [data[ds]["Single Agent"] for ds in datasets]
        debate_accs = [data[ds]["Multi-Agent Debate"] for ds in datasets]

        x = np.arange(len(datasets))
        width = 0.35

        fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 2), 5), dpi=300)
        
        # Dual colors: Blue #4285F4 & Red #EA4335
        rects1 = ax.bar(x - width/2, single_agent_accs, width, label='Single Agent', color='#4285F4', edgecolor='none')
        rects2 = ax.bar(x + width/2, debate_accs, width, label='Multi-Agent Debate', color='#EA4335', edgecolor='none')

        # Y-axis & Spine styling
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='medium', labelpad=10)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=11, fontweight='medium', rotation=0 if len(datasets) <= 6 else 15)
        
        # Background grids & Spines
        ax.grid(axis='y', linestyle='-', alpha=0.3, color='#CCCCCC')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#000000')
        ax.spines['bottom'].set_color('#000000')
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        # Top Legend
        ax.legend(
            loc='upper center', 
            bbox_to_anchor=(0.5, 1.18), 
            ncols=2, 
            frameon=False, 
            fontsize=11, 
            handlelength=1.2, 
            handleheight=1.2
        )

        # Data value annotations directly above bars
        for rect in rects1:
            height = rect.get_height()
            ax.annotate(
                f'{int(round(height))}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 5),  
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#4285F4'
            )

        for rect in rects2:
            height = rect.get_height()
            ax.annotate(
                f'{int(round(height))}',
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 5),  
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#EA4335'
            )

        plt.tight_layout()

        out_file = save_path or os.path.join(self.output_dir, "accuracy_comparison_barchart.png")
        plt.savefig(out_file, bbox_inches='tight', dpi=300)
        print(f"Bar chart successfully saved to {out_file}")
        return fig

    def generate_paper_styled_table(
        self,
        benchmark_results: Dict[str, Dict[str, Tuple[float, float]]],
        save_path: Optional[str] = None
    ) -> Tuple[pd.DataFrame, plt.Figure]:
        """
        Generates the Academic LaTeX/Publication styled table derived strictly from
        the user's dataset evaluation results:
        - Columns: Model | [Dataset Name] (%) ↑
        - Rows: Single Agent, Single Agent (Reflection), Multiagent (Majority), Multiagent (Debate)
        - Cell Values: mean ± std, with top performer per dataset bolded.
        """
        if not benchmark_results:
            raise ValueError("benchmark_results dictionary cannot be empty. Run dataset evaluation first.")

        strategies = [
            "Single Agent",
            "Single Agent (Reflection)",
            "Multiagent (Majority)",
            "Multiagent (Debate)"
        ]

        raw_data = benchmark_results
        datasets = list(raw_data.keys())
        
        # Construct Pandas DataFrame
        table_rows = []
        for strat in strategies:
            row = {"Model": strat}
            for ds in datasets:
                mean, std = raw_data.get(ds, {}).get(strat, (0.0, 0.0))
                row[f"{ds} (%) ↑"] = f"{mean:.1f} ± {std:.1f}"
            table_rows.append(row)

        df = pd.DataFrame(table_rows)

        # Plot academic LaTeX table
        fig, ax = plt.subplots(figsize=(max(10, len(datasets) * 2.2), 2.6), dpi=300)
        ax.axis('off')

        col_labels = ["Model"] + [f"{ds} (%) ↑" for ds in datasets]
        cell_text = []

        # Find best performing strategy index per dataset for bold formatting
        best_indices = {}
        for ds in datasets:
            means = [raw_data.get(ds, {}).get(strat, (0.0, 0.0))[0] for strat in strategies]
            best_indices[ds] = np.argmax(means)

        for row_idx, strat in enumerate(strategies):
            r_text = [strat]
            for ds in datasets:
                mean, std = raw_data.get(ds, {}).get(strat, (0.0, 0.0))
                val_str = f"{mean:.1f} ± {std:.1f}"
                if row_idx == best_indices[ds]:
                    val_str = rf"$\bf{{{mean:.1f} \pm {std:.1f}}}$"
                r_text.append(val_str)
            cell_text.append(r_text)

        tbl = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc='center',
            cellLoc='center'
        )

        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1.3, 1.9)

        # Publication booktabs styling
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor('none')
            cell.set_facecolor('white')
            if r == 0:
                cell.get_text().set_fontweight('bold')
                cell.get_text().set_fontsize(11)
                cell.visible_edges = 'TB'
                cell.set_edgecolor('black')
                cell.set_linewidth(1.2)
            elif r == len(strategies):
                cell.visible_edges = 'B'
                cell.set_edgecolor('black')
                cell.set_linewidth(1.2)
            
            if c == 0:
                cell.get_text().set_ha('left')

        plt.tight_layout()

        out_file = save_path or os.path.join(self.output_dir, "academic_comparison_table.png")
        plt.savefig(out_file, bbox_inches='tight', dpi=300)
        print(f"Academic table image successfully saved to {out_file}")

        return df, fig
