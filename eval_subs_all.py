from competetion_eval_type import COMPETITION_EVAL_TYPE
from competetion_type import COMPETITION_TYPE
import numpy as np
import pandas as pd
import os, sys, ipdb
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score


def rmsle(preds, labels):
    return np.sqrt(np.mean((np.log(preds + 1) - np.log(labels + 1)) ** 2))

def rmse(preds, labels):
    return np.sqrt(np.mean((preds - labels) ** 2))

def accuracy(preds, labels):
    return (preds == labels).mean()
 
def gini_normalized(preds, labels):

    def gini(actual, pred, cmpcol = 0, sortcol = 1):
        assert( len(actual) == len(pred) )
        all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=float)
        all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]
        totalLosses = all[:,0].sum()
        giniSum = all[:,0].cumsum().sum() / totalLosses

        giniSum -= (len(actual) + 1) / 2.
        return giniSum / len(actual)

    return gini(labels, preds) / gini(labels, labels)

def auc_roc(preds, labels):
    return roc_auc_score(labels, preds)

def matthews_corr_coef(preds, labels):
    return matthews_corrcoef(labels, preds)

def macro_f1(preds, labels):
    return f1_score(labels, preds, average='macro')

if __name__ == "__main__":

    eval_func_map = {
        'accuracy': accuracy,
        'RMSE': rmse,
        'RMSLE': rmsle,
        'normalized-gini-coefficient': gini_normalized,
        'auc-roc': auc_roc,
        'matthews-corrcoef': matthews_corr_coef,
        'macro-f1': macro_f1,
    }
    target_col_map = {
        'spaceship_titanic': 'Transported', 
        'santander_value_prediction_challenge': 'target', 
        'nyc_taxi_fare_prediction': 'fare_amount', 
        'nyc_taxi_trip_duration': 'trip_duration',
        'bpm_prediction': 'BeatsPerMinute',
        'calorie_expenditure_prediction': 'Calories',
        'california_housing_regression': 'MedHouseVal', 
        'used_car_prices_regression': 'price', 
        'porto_seguro_safe_driver_prediction': 'target', 
        'costa_rican_household_poverty_level_prediction': 'Target', 
        'forest_cover_type_prediction': 'Cover_Type', 
        'santander_customer_transaction_prediction': 'target', 
        'poisonous_mushroom_prediction': 'class', 
        'bank_deposit_classification': 'y', 
        'bank_churn_classification': 'Exited', 
    }

    COMPETITIONS = list(COMPETITION_EVAL_TYPE.keys())

    DIR_PREFIX = '/sensei-fs-3/users/ychittepu/exps_new/submissions'
    OUTPUT_DIR = './eval_results'
    # ALGO = 'lats_reflection'
    ALGO = 'lats_shaped_rewards_rd_0_feedback'
    ALGOS = ['lats_hierarchical_rd_0_feedback', 'lats_shaped_rewards_rd_0_feedback', 'lats_outcome_rewards_rd_0_feedback', 'lats_reflection', 'react']
    # ['lats_shaped_rewards_rd_0_feedback', 'lats_outcome_rewards_rd_0_feedback', 'lats_reflection]
    
    # Specify target files per algorithm (if any). If empty dict, uses default logic
    # TARGET_FILES = {}  # e.g., {'react': ['file1.csv', 'file2.csv'], 'lats_reflection': ['manual_1.csv']}
    TARGET_FILES = {
        'react': ['trajectory_4_47a197f8-edb6-4c43-a9ce-ae5f8ef45736.csv'],
        'lats_hierarchical_rd_0_feedback': ['lats_hierarchical_rollout_0_parallel_feedback_trajectory_3_23cab1e3-fd64-47d4-8c12-9b3c0f352f09_1763677973.csv'],
        'lats_outcome_rewards_rd_0_feedback': ['lats_verifiable_rewards_outcome_feedback_trajectory_4_efc47741-d530-4f33-8cb0-3220ec48fc99.csv'],
        'lats_shaped_rewards_rd_0_feedback': ['lats_verifiable_rewards_shaped_feedback_trajectory_4_de404b82-c236-4a70-8082-7361514daab9.csv'],
        'lats_reflection': ['lats_trajectory_4_794a43f3-2334-4f75-9ee5-69d5e5abf93c.csv'],
    }
    # Example:
    # TARGET_FILES = {
    #     'react': ['trajectory_0_agent.csv', 'trajectory_1_agent.csv'],
    #     'lats_reflection': ['lats_trajectory_manual_best.csv'],
    #     'lats_shaped_rewards_rd_0_feedback': ['shaped_manual_top.csv']
    # }
    
    print(f"📊 EVALUATION CONFIGURATION:")
    print(f"   Input Dir: {DIR_PREFIX}")
    print(f"   Output Dir: {OUTPUT_DIR}")
    print(f"   Algorithms: {ALGOS}")
    print(f"   Competitions: {len(COMPETITIONS)} total")
    if TARGET_FILES:
        print(f"   Target Files Per Algorithm:")
        for algo, files in TARGET_FILES.items():
            print(f"     {algo}: {files}")
        print(f"   (Algorithms not listed will use default filter)")
    else:
        print(f"   File Filter: Default (react=all csvs, others=manual only)")
    print()

    lst = []
    results_rows = []
    # competitions = ['bank_deposit_classification']
    competitions = COMPETITIONS
    for comp in competitions:
        test_df = pd.read_csv(f'./data/{comp}/test_with_labels.csv')
        target_col = target_col_map[comp]
        test_df_labels = test_df[target_col]

        leaderboard_df = pd.read_csv(f'./leaderboards/{comp}.csv')
        if 'Score' not in leaderboard_df.columns:
            raise ValueError(f"Leaderboard file for '{comp}' must contain a 'Score' column.")
        leaderboard_scores = leaderboard_df['Score'].dropna().to_numpy()
        lower_is_better = COMPETITION_EVAL_TYPE[comp] in ['RMSE', 'RMSLE']

        for algo in ALGOS:
            path = os.path.join(DIR_PREFIX, comp, algo)
            if not os.path.isdir(path):
                print(f"Competition: {comp}, Algo: {algo} path missing, skipping")
                continue
            dataframes = {}
            corrupted_df = 0
            for file in os.listdir(path):
                # If specific target files are provided for this algorithm, only consider those
                if TARGET_FILES and algo in TARGET_FILES:
                    file_condition = file in TARGET_FILES[algo]
                else:
                    # Default logic: react = all csvs, others = manual only
                    file_condition = file.endswith('.csv') and (algo == 'react' or 'manual' in file.lower())
                
                if file_condition:
                    try:
                        df = pd.read_csv(os.path.join(path, file))
                        dataframes[file] = df
                    except Exception as e:
                        corrupted_df += 1
                        pass

            print(f"Competition: {comp}, Algo: {algo}, Length of loaded dataframes: {len(dataframes)}")

            metrics = {}
            percentiles = {}
            size_mismatch = 0
            invalid_cols = 0
            for df_name, df in dataframes.items():
                if(df.shape[0] != test_df_labels.shape[0]):
                    size_mismatch += 1
                    continue
                if(COMPETITION_TYPE[comp] in ['binary-acc', 'multiclass', 'regression']):
                    cols = list(df.columns)
                    if len(cols) > 2:
                        invalid_cols += 1
                        print(f"Expected at most 2 columns (id + prediction) in '{df_name}' for '{comp}', found {len(cols)}: {cols}")
                        continue
                    if len(cols) == 1:
                        col_name = cols[0]
                    else:
                        c1, c2 = cols[0], cols[1]
                        def classify_col(series: pd.Series):
                            s = series.dropna()
                            if s.size == 0:
                                return ('empty', False)
                            if pd.api.types.is_float_dtype(s) or pd.api.types.is_integer_dtype(s):
                                return ('numeric', True)
                            if pd.api.types.is_bool_dtype(s):
                                return ('bool', False)
                            if pd.api.types.is_object_dtype(s):
                                lowered = s.astype(str).str.strip().str.lower()
                                allowed = {'true', 'false', 'yes', 'no', 'y', 'n'}
                                uniq = set(lowered.unique().tolist())
                                if len(uniq) > 0 and uniq.issubset(allowed):
                                    return ('bool-like', False)
                            return ('other', False)

                        t1, _ = classify_col(df[c1])
                        t2, _ = classify_col(df[c2])

                        # Prefer numeric over bool/bool-like
                        if t1 == 'numeric' and t2 != 'numeric':
                            col_name = c1
                        elif t2 == 'numeric' and t1 != 'numeric':
                            col_name = c2
                        elif t1 == 'numeric' and t2 == 'numeric':
                            # Heuristic: avoid choosing an ID-like column by preferring the one
                            # with fewer unique values than rows (predictions typically repeat)
                            n = df.shape[0]
                            uniq1, uniq2 = df[c1].nunique(dropna=True), df[c2].nunique(dropna=True)
                            if uniq1 == n and uniq2 < n:
                                col_name = c2
                            elif uniq2 == n and uniq1 < n:
                                col_name = c1
                            else:
                                # fall back to preferring float over int
                                if pd.api.types.is_float_dtype(df[c1]) and not pd.api.types.is_float_dtype(df[c2]):
                                    col_name = c1
                                elif pd.api.types.is_float_dtype(df[c2]) and not pd.api.types.is_float_dtype(df[c1]):
                                    col_name = c2
                                else:
                                    # if still ambiguous, choose second column (common Kaggle format: id, prediction)
                                    col_name = c2
                        elif t1 in ('bool', 'bool-like') and t2 not in ('bool', 'bool-like'):
                            col_name = c1
                        elif t2 in ('bool', 'bool-like') and t1 not in ('bool', 'bool-like'):
                            col_name = c2
                        elif t1 in ('bool', 'bool-like') and t2 in ('bool', 'bool-like'):
                            invalid_cols += 1
                            raise ValueError(f"Ambiguous: two boolean-like columns {cols} in '{df_name}' for '{comp}'.")
                        else:
                            invalid_cols += 1
                            raise ValueError(f"No suitable prediction column among {cols} in '{df_name}' for '{comp}'.")

                    # Check for invalid values in predictions (NaN, inf, -inf)
                    predictions = df[col_name]
                    nan_count = predictions.isna().sum()
                    inf_count = np.isinf(predictions).sum()
                    
                    if nan_count > 0 or inf_count > 0:
                        invalid_cols += 1
                        print(f"Invalid submission '{df_name}' for '{comp}': contains {nan_count} NaN values and {inf_count} infinite values")
                        continue
                    
                    metric_val = eval_func_map[COMPETITION_EVAL_TYPE[comp]](predictions, test_df_labels)
                    metrics[df_name] = metric_val
                    if lower_is_better:
                        pct = (leaderboard_scores >= metric_val).mean() * 100.0
                    else:
                        pct = (leaderboard_scores <= metric_val).mean() * 100.0
                    percentiles[df_name] = pct
                else:
                    # For competitions expecting probability outputs (e.g., 'binary-probs'),
                    # find columns that are float and lie within [0, 1].
                    float_cols_in_01 = []
                    # print("DF COLS: ", df.columns)
                    for col in df.columns:
                        if pd.api.types.is_float_dtype(df[col]):
                            s = df[col].dropna()
                            if s.size == 0:
                                continue
                            # Require that at least one value lies strictly between 0 and 1 (exclusive)
                            # to avoid selecting pure 0/1 label columns.
                            if ((s > 0.0 + 1e-9) & (s < 1.0 - 1e-9)).any():
                                float_cols_in_01.append(col)

                    if len(float_cols_in_01) == 0:
                        invalid_cols += 1
                        continue

                    if len(float_cols_in_01) >= 2:
                        if 'prob_class_1.0' in float_cols_in_01:
                            col_name = 'prob_class_1.0'
                        elif 'prob_class_1' in float_cols_in_01:
                            col_name = 'prob_class_1'
                        else:
                            invalid_cols += 1
                            print(
                                f"Multiple probability columns found {float_cols_in_01} but neither 'prob_class_1.0' nor 'prob_class_1' present in '{df_name}' for competition '{comp}'."
                            )
                            continue
                    else:
                        col_name = float_cols_in_01[0]

                    # Check for invalid values in predictions (NaN, inf, -inf)
                    predictions = df[col_name]
                    nan_count = predictions.isna().sum()
                    inf_count = np.isinf(predictions).sum()
                    
                    if nan_count > 0 or inf_count > 0:
                        invalid_cols += 1
                        print(f"Invalid submission '{df_name}' for '{comp}': contains {nan_count} NaN values and {inf_count} infinite values")
                        continue
                    
                    metric_val = eval_func_map[COMPETITION_EVAL_TYPE[comp]](predictions, test_df_labels)
                    metrics[df_name] = metric_val
                    if lower_is_better:
                        pct = (leaderboard_scores >= metric_val).mean() * 100.0
                    else:
                        pct = (leaderboard_scores <= metric_val).mean() * 100.0
                    percentiles[df_name] = pct
            # Penalize invalid runs (assume 10 total runs): fill missing with worst-case values
            total_runs = 10
            valid_metric_vals = list(metrics.values())
            valid_percentiles = list(percentiles.values())
            num_invalid = max(0, total_runs - len(valid_metric_vals))
            if num_invalid > 0:
                if lower_is_better:
                    # Worst is large; use leaderboard max as concrete worst
                    worst_metric = 1e8
                else:
                    # Worst is small; use leaderboard min as concrete worst
                    worst_metric = 0.0
                valid_metric_vals.extend([worst_metric] * num_invalid)
                valid_percentiles.extend([0.0] * num_invalid)

            median_metric = np.median(valid_metric_vals) if len(valid_metric_vals) else float('nan')
            median_percentile = np.median(valid_percentiles) if len(valid_percentiles) else float('nan')
            print(f"Competition: {comp}, Algo: {algo}, Consistency: {len(metrics)/10}, Median metric: {median_metric:.4f}, Median percentile: {median_percentile:.2f}, Corrupted df: {corrupted_df}, Size mismatch: {size_mismatch}, Invalid cols: {invalid_cols}")
            results_rows.append({
                'comp': comp,
                'algo': algo,
                'scoring_function': COMPETITION_EVAL_TYPE[comp],
                'lower_is_better': lower_is_better,
                'consistency': len(metrics)/10,
                'median_percentile': median_percentile,
                'median_metric': median_metric,
            })
    results_df = pd.DataFrame(results_rows)
    print("\nResults summary:\n", results_df)
    # Build a wide table with MultiIndex columns: top-level = algo, sub-level = [consistency, median_metric, median_percentile]
    if len(results_df) > 0:
        meta_df = results_df[['comp', 'scoring_function', 'lower_is_better']].drop_duplicates('comp').set_index('comp')
        pivot = results_df.pivot_table(
            index='comp',
            columns='algo',
            values=['consistency', 'median_metric', 'median_percentile'],
            aggfunc='first'
        )
        # Swap so that top level is algo and second level is metric name
        pivot = pivot.swaplevel(0, 1, axis=1).sort_index(axis=1, level=0)
        wide_results_df = pd.concat([meta_df, pivot], axis=1).reset_index()
        print("\nWide results (algo as columns):\n", wide_results_df)
        # Save both long and wide results with unique naming
        import datetime
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        long_filename = f'results_long_{timestamp}.csv'
        wide_filename = f'results_comp_vs_algo_{timestamp}.csv'
        
        results_df.to_csv(os.path.join(OUTPUT_DIR, long_filename), index=False)
        wide_results_df.to_csv(os.path.join(OUTPUT_DIR, wide_filename), index=False)
        
        print(f"\n💾 Results saved to:")
        print(f"   - {os.path.join(OUTPUT_DIR, long_filename)}")
        print(f"   - {os.path.join(OUTPUT_DIR, wide_filename)}")
        

