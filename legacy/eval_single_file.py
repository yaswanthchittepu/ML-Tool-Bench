#!/usr/bin/env python3
"""
Single File Performance Evaluator
Evaluates the performance and median percentile of a single submission file
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from competetion_eval_type import COMPETITION_EVAL_TYPE
from competetion_type import COMPETITION_TYPE
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


def classify_col(series: pd.Series):
    """Classify a pandas series to determine if it's numeric, boolean, etc."""
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


def evaluate_single_file(file_path, competition_name, data_dir='./data', leaderboard_dir='./leaderboards'):
    """
    Evaluate a single submission file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV submission file
    competition_name : str
        Name of the competition (must be in COMPETITION_EVAL_TYPE)
    data_dir : str
        Directory containing competition data
    leaderboard_dir : str
        Directory containing leaderboard files
        
    Returns:
    --------
    dict : Evaluation results
    """
    
    # Validate inputs
    if competition_name not in COMPETITION_EVAL_TYPE:
        raise ValueError(f"Unknown competition: {competition_name}. Available: {list(COMPETITION_EVAL_TYPE.keys())}")
    
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    
    # Target column mapping
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
    
    # Evaluation function mapping
    eval_func_map = {
        'accuracy': accuracy,
        'RMSE': rmse,
        'RMSLE': rmsle,
        'normalized-gini-coefficient': gini_normalized,
        'auc-roc': auc_roc,
        'matthews-corrcoef': matthews_corr_coef,
        'macro-f1': macro_f1,
    }
    
    # Load test data with labels
    test_data_path = os.path.join(data_dir, competition_name, 'test_with_labels.csv')
    if not os.path.exists(test_data_path):
        raise ValueError(f"Test data not found: {test_data_path}")
    
    test_df = pd.read_csv(test_data_path)
    target_col = target_col_map[competition_name]
    test_labels = test_df[target_col]
    
    # Load leaderboard for percentile calculation
    leaderboard_path = os.path.join(leaderboard_dir, f'{competition_name}.csv')
    if not os.path.exists(leaderboard_path):
        raise ValueError(f"Leaderboard not found: {leaderboard_path}")
    
    leaderboard_df = pd.read_csv(leaderboard_path)
    if 'Score' not in leaderboard_df.columns:
        raise ValueError(f"Leaderboard file must contain a 'Score' column.")
    leaderboard_scores = leaderboard_df['Score'].dropna().to_numpy()
    
    # Load submission file
    try:
        submission_df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"Error loading submission file: {e}")
    
    # Validate submission format
    if submission_df.shape[0] != test_labels.shape[0]:
        raise ValueError(f"Size mismatch: submission has {submission_df.shape[0]} rows, expected {test_labels.shape[0]}")
    
    # Determine prediction column
    eval_type = COMPETITION_EVAL_TYPE[competition_name]
    comp_type = COMPETITION_TYPE[competition_name]
    lower_is_better = eval_type in ['RMSE', 'RMSLE']
    
    if comp_type in ['binary-acc', 'multiclass', 'regression']:
        cols = list(submission_df.columns)
        if len(cols) > 2:
            raise ValueError(f"Expected at most 2 columns (id + prediction), found {len(cols)}: {cols}")
        
        if len(cols) == 1:
            col_name = cols[0]
        else:
            c1, c2 = cols[0], cols[1]
            t1, _ = classify_col(submission_df[c1])
            t2, _ = classify_col(submission_df[c2])
            
            # Prefer numeric over bool/bool-like
            if t1 == 'numeric' and t2 != 'numeric':
                col_name = c1
            elif t2 == 'numeric' and t1 != 'numeric':
                col_name = c2
            elif t1 == 'numeric' and t2 == 'numeric':
                # Avoid ID-like columns by preferring the one with fewer unique values
                n = submission_df.shape[0]
                uniq1, uniq2 = submission_df[c1].nunique(dropna=True), submission_df[c2].nunique(dropna=True)
                if uniq1 == n and uniq2 < n:
                    col_name = c2
                elif uniq2 == n and uniq1 < n:
                    col_name = c1
                else:
                    # Prefer float over int
                    if pd.api.types.is_float_dtype(submission_df[c1]) and not pd.api.types.is_float_dtype(submission_df[c2]):
                        col_name = c1
                    elif pd.api.types.is_float_dtype(submission_df[c2]) and not pd.api.types.is_float_dtype(submission_df[c1]):
                        col_name = c2
                    else:
                        # Default to second column (common Kaggle format: id, prediction)
                        col_name = c2
            elif t1 in ('bool', 'bool-like') and t2 not in ('bool', 'bool-like'):
                col_name = c1
            elif t2 in ('bool', 'bool-like') and t1 not in ('bool', 'bool-like'):
                col_name = c2
            else:
                raise ValueError(f"No suitable prediction column among {cols}")
    
    else:
        # For probability-based competitions
        float_cols_in_01 = []
        for col in submission_df.columns:
            if pd.api.types.is_float_dtype(submission_df[col]):
                s = submission_df[col].dropna()
                if s.size == 0:
                    continue
                if ((s > 0.0 + 1e-9) & (s < 1.0 - 1e-9)).any():
                    float_cols_in_01.append(col)
        
        if len(float_cols_in_01) == 0:
            raise ValueError("No suitable probability columns found")
        
        if len(float_cols_in_01) >= 2:
            if 'prob_class_1.0' in float_cols_in_01:
                col_name = 'prob_class_1.0'
            elif 'prob_class_1' in float_cols_in_01:
                col_name = 'prob_class_1'
            else:
                raise ValueError(f"Multiple probability columns found {float_cols_in_01} but neither 'prob_class_1.0' nor 'prob_class_1' present")
        else:
            col_name = float_cols_in_01[0]
    
    # Check for invalid values
    predictions = submission_df[col_name]
    nan_count = predictions.isna().sum()
    inf_count = np.isinf(predictions).sum()
    
    if nan_count > 0 or inf_count > 0:
        raise ValueError(f"Invalid submission: contains {nan_count} NaN values and {inf_count} infinite values")
    
    # Calculate metric
    metric_val = eval_func_map[eval_type](predictions, test_labels)
    
    # Calculate percentile
    if lower_is_better:
        percentile = (leaderboard_scores >= metric_val).mean() * 100.0
    else:
        percentile = (leaderboard_scores <= metric_val).mean() * 100.0
    
    # Return results
    results = {
        'file_path': file_path,
        'competition': competition_name,
        'evaluation_metric': eval_type,
        'lower_is_better': lower_is_better,
        'metric_value': float(metric_val),
        'percentile': float(percentile),
        'prediction_column': col_name,
        'num_predictions': len(predictions),
        'leaderboard_scores_count': len(leaderboard_scores)
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate a single submission file')
    parser.add_argument('file_path', help='Path to the CSV submission file')
    parser.add_argument('competition', help='Competition name')
    parser.add_argument('--data_dir', default='./data', help='Data directory (default: ./data)')
    parser.add_argument('--leaderboard_dir', default='./leaderboards', help='Leaderboard directory (default: ./leaderboards)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    try:
        results = evaluate_single_file(args.file_path, args.competition, args.data_dir, args.leaderboard_dir)
        
        if args.verbose:
            print("🎯 SINGLE FILE EVALUATION RESULTS")
            print("=" * 45)
            print(f"📁 File: {results['file_path']}")
            print(f"🏆 Competition: {results['competition']}")
            print(f"📊 Metric: {results['evaluation_metric']}")
            print(f"📈 Value: {results['metric_value']:.6f}")
            print(f"🥇 Percentile: {results['percentile']:.2f}%")
            print(f"🔢 Predictions: {results['num_predictions']}")
            print(f"📋 Column Used: {results['prediction_column']}")
            print(f"🎯 Better when {'lower' if results['lower_is_better'] else 'higher'}")
        else:
            # Compact output
            print(f"Metric: {results['metric_value']:.6f}, Percentile: {results['percentile']:.2f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

