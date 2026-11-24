import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os, sys, ipdb

if __name__ == "__main__":

    MODEL = 'gpt4o'
    # MODEL = 'gpt4_1mini'
    MODEL_MAP = {
        'gpt4o': 'GPT-4o',
        'gpt4_1mini': 'GPT-4.1-mini',
    }
    METRIC = 'median_percentile'
    PATH = f'./results/results_comp_vs_algo_{MODEL}.csv'
    results_df = pd.read_csv(PATH)
    print(results_df.columns)
    consistency_dct = {}
    median_metric_dct = {}
    median_percentile_dct = {}
    algo_map = {
        'lats_hierarchical_rd_0_feedback': 'Hierarchical MCTS',
        'lats_shaped_rewards_rd_0_feedback': 'MCTS-Shaped',
        'lats_outcome_rewards_rd_0_feedback': 'MCTS-Outcome',
        'lats_reflection': 'LATS',
        'react': 'ReAct',
    }
    for col in results_df.columns:
        if('consistency' in col):
            algo, met = col.split(',')
            algo = algo.strip()[2:-1]
            met = met.strip()[1:-2]
            consistency_dct[algo_map[algo]] = col
        if('median_metric' in col):
            algo, met = col.split(',')
            algo = algo.strip()[2:-1]
            met = met.strip()[1:-2]
            median_metric_dct[algo_map[algo]] = col
        if('median_percentile' in col):
            algo, met = col.split(',')
            algo = algo.strip()[2:-1]
            met = met.strip()[1:-2]
            median_percentile_dct[algo_map[algo]] = col   

    if(METRIC == 'consistency'):
        dct = consistency_dct
    elif(METRIC == 'median_metric'):
        dct = median_metric_dct
    elif(METRIC == 'median_percentile'):
        dct = median_percentile_dct
    else:
        raise ValueError(f"Invalid metric: {METRIC}")

    # Reshape data for plotting - melt the algorithm columns into long format
    plot_data = []
    for algo in ['ReAct', 'LATS', 'MCTS-Outcome', 'MCTS-Shaped', 'Hierarchical MCTS']:
        # values = results_df[col].dropna()  # Remove NaN values
        col = dct[algo]
        values = results_df[col]
        for val in values:
            plot_data.append({'algorithm': algo, 'score': val})
    
    plot_df = pd.DataFrame(plot_data)
    print(f"\nPlot data shape: {plot_df.shape}")
    print(f"Algorithms: {plot_df['algorithm'].unique()}")
    print(f"Score range: {plot_df['score'].min():.3f} to {plot_df['score'].max():.3f}")
    
    # plt.figure(figsize=(12, 8))
    # sns.violinplot(data=plot_df, x="algorithm", y="score", inner="quartile", cut=0)
    # sns.stripplot(data=plot_df, x="algorithm", y="score", color="black", size=4, jitter=True, alpha=0.7)
    # plt.title(f"Distribution of {METRIC.replace('_', ' ').title()} Across Competitions ({MODEL.upper()})")
    # plt.ylabel(f"{METRIC.replace('_', ' ').title()}")
    # plt.xlabel("Algorithm")
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(12, 8))
    
    # Prepare data for boxplot
    algorithm_order = ['ReAct', 'LATS', 'MCTS-Outcome', 'MCTS-Shaped', 'Hierarchical MCTS']
    box_data = []
    labels = []
    colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink']
    
    for algo in algorithm_order:
        col = dct[algo]
        data = results_df[col].dropna()
        box_data.append(data)
        # labels.append(f"{algo}\n(n={len(data)})")
        labels.append(f"{algo}")
    
    # Create boxplot with template style
    bp = plt.boxplot(box_data, tick_labels=labels, patch_artist=True, widths=0.25)
    
    # Set tick label font size for boxplot
    plt.xticks(fontsize=26)

    # Set colors and transparency
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    
    # Add statistical annotations
    for i, data in enumerate(box_data):
        if len(data) > 0:
            mean_val = data.mean()
            median_val = data.median()
            std_val = data.std()
            plt.text(
                i + 1,
                plt.gca().get_ylim()[1] * 0.9,
                f"μ={mean_val:.3f}\nM={median_val:.3f}\nσ={std_val:.3f}",
                ha="center",
                va="top",
                fontsize=14,
                bbox=dict(boxstyle="round,pad=0.6", facecolor="white", alpha=0.8),
            )
    
    # Add strip plot overlay
    for i, data in enumerate(box_data):
        y_values = data.values
        x_values = np.random.normal(i + 1, 0.04, size=len(y_values))  # jitter
        plt.scatter(x_values, y_values, color="black", s=16, alpha=0.7)
    
    # Add median trend line
    median_values = [data.median() for data in box_data]
    
    plt.plot(range(1, len(algorithm_order) + 1), median_values, 
             color='red', marker='o', linewidth=2.5, markersize=7, 
             label='Median Trend', alpha=0.5)
    
    plt.grid(True, alpha=0.3)
    plt.title(f"Distribution of {METRIC.replace('_', ' ').title()} Across Competitions ({MODEL_MAP[MODEL]})", fontsize=22, fontweight="bold")
    plt.ylabel(f"{METRIC.replace('_', ' ').title()}", fontsize=18)
    plt.xlabel("Algorithm", fontsize=18)
    plt.tick_params(axis="x", rotation=45, labelsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=14, loc='best')
    plt.tight_layout()
    plt.savefig(f'distribution_{METRIC}_{MODEL}.png', dpi=300, bbox_inches='tight')
    plt.show()