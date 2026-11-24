"""
Evaluate Kaggle Benchmark Results - Simple version that just processes the JSON
"""

import json
import pandas as pd
import numpy as np
import os
from competetion_eval_type import COMPETITION_EVAL_TYPE

COMPETITION_NAME_MAP = {
    'Spaceship Titanic': 'spaceship_titanic',
    'Calorie Expenditure Prediction': 'calorie_expenditure_prediction',
    'California Housing Regression': 'california_housing_regression',
    'BPM_Prediction': 'bpm_prediction',
    'Bank Deposit Classification': 'bank_deposit_classification',
    'Bank Churn Classification': 'bank_churn_classification',
}

def compute_percentile(score, leaderboard_scores, lower_is_better):
    if lower_is_better:
        return (leaderboard_scores >= score).mean() * 100.0
    else:
        return (leaderboard_scores <= score).mean() * 100.0

# Load JSON
with open('kaggle_benchmark_results.json', 'r') as f:
    data = json.load(f)

print(f"\n{'='*80}")
print("KAGGLE BENCHMARK EVALUATION")
print(f"{'='*80}\n")

results = []

for competition, algorithms in data.items():
    comp_name = COMPETITION_NAME_MAP[competition]
    
    print(f"\nCOMPETITION: {competition}")
    
    # Load leaderboard
    leaderboard_df = pd.read_csv(f'leaderboards/{comp_name}.csv')
    leaderboard_scores = leaderboard_df['Score'].dropna().to_numpy()
    
    eval_type = COMPETITION_EVAL_TYPE[comp_name]
    lower_is_better = eval_type in ['RMSE', 'RMSLE']
    
    print(f"Metric: {eval_type}, Leaderboard: {len(leaderboard_scores)} scores")
    
    for algorithm, scores in algorithms.items():
        if not scores:
            # Empty results - all 10 runs failed
            print(f"  {algorithm}: 0/10 valid | median=0.00% mean=0.00% min=0.00% max=0.00%")
            results.append({
                'competition': competition,
                'algorithm': algorithm,
                'num_valid': 0,
                'consistency': 0.0,
                'median_percentile': 0.0,
                'mean_percentile': 0.0,
                'min_percentile': 0.0,
                'max_percentile': 0.0,
            })
            continue
        
        # Calculate percentiles
        percentiles = [compute_percentile(s, leaderboard_scores, lower_is_better) for s in scores]
        
        # Pad to 10
        num_valid = len(percentiles)
        percentiles.extend([0.0] * (10 - num_valid))
        
        # Stats
        median = np.median(percentiles)
        mean = np.mean(percentiles)
        min_p = np.min(percentiles)
        max_p = np.max(percentiles)
        
        print(f"  {algorithm}: {num_valid}/10 valid | median={median:.2f}% mean={mean:.2f}% min={min_p:.2f}% max={max_p:.2f}%")
        
        results.append({
            'competition': competition,
            'algorithm': algorithm,
            'num_valid': num_valid,
            'consistency': num_valid / 10.0,
            'median_percentile': median,
            'mean_percentile': mean,
            'min_percentile': min_p,
            'max_percentile': max_p,
        })

# Save
df = pd.DataFrame(results)

os.makedirs('eval_results_new', exist_ok=True)
from datetime import datetime
ts = datetime.now().strftime('%Y%m%d_%H%M%S')

long_file = f'eval_results_new/kaggle_benchmark_{ts}.csv'
df.to_csv(long_file, index=False)

print(f"\n{'='*80}")
print("SAVED:")
print(f"  {long_file}")
print(f"{'='*80}\n")

print("BY ALGORITHM:")
print(df.groupby('algorithm')[['consistency', 'median_percentile', 'mean_percentile']].mean().round(2))
