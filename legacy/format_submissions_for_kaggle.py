"""
Format Agent Submissions for Kaggle

This script converts agent-generated CSV files to proper Kaggle submission format.

Proper Kaggle format:
- ID column from original test data
- For 'binary-acc': predicted label (0/1, True/False)
- For 'binary-probs': predicted probability (float in [0,1])
- For 'regression': predicted value (float)
- For 'multiclass': predicted class label

Usage:
    # Single file
    python format_submissions_for_kaggle.py --input submissions/spaceship_titanic/react_code/react_code_agent_123.csv --competition spaceship_titanic --output formatted_submission.csv
    
    # Batch process all CSVs in a directory
    python format_submissions_for_kaggle.py --input_dir submissions/spaceship_titanic/react_code/ --competition spaceship_titanic --output_dir formatted_submissions/
    
    # Auto-detect competition from path
    python format_submissions_for_kaggle.py --input submissions/spaceship_titanic/react_code/react_code_agent_123.csv --auto
    
    # Process ALL competitions and algorithms (like eval_subs_all.py)
    python format_submissions_for_kaggle.py --batch_all --dir_prefix submissions --output_dir formatted_submissions --algorithms react react_code lats_reflection
    
    # Process specific competitions only
    python format_submissions_for_kaggle.py --batch_all --competitions_list spaceship_titanic bank_churn_classification --algorithms react react_code
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from competetion_type import COMPETITION_TYPE


# Map competition to ID column name
ID_COLUMN_MAP = {
    'spaceship_titanic': 'PassengerId',
    # 'santander_value_prediction_challenge': 'ID',
    # 'nyc_taxi_fare_prediction': 'key',
    # 'nyc_taxi_trip_duration': 'id',
    'bpm_prediction': 'id',
    'calorie_expenditure_prediction': 'id',
    'california_housing_regression': 'id',
    # 'used_car_prices_regression': 'id',
    # 'porto_seguro_safe_driver_prediction': 'id',
    # 'costa_rican_household_poverty_level_prediction': 'Id',
    # 'forest_cover_type_prediction': 'Id',
    # 'santander_customer_transaction_prediction': 'ID_code',
    # 'poisonous_mushroom_prediction': 'id',
    'bank_deposit_classification': 'id',
    'bank_churn_classification': 'id',
}

# Map competition to target column name for Kaggle submission
TARGET_COLUMN_MAP = {
    'spaceship_titanic': 'Transported',
    # 'santander_value_prediction_challenge': 'target',
    # 'nyc_taxi_fare_prediction': 'fare_amount',
    # 'nyc_taxi_trip_duration': 'trip_duration',
    'bpm_prediction': 'BeatsPerMinute',
    'calorie_expenditure_prediction': 'Calories',
    'california_housing_regression': 'MedHouseVal',
    # 'used_car_prices_regression': 'price',
    # 'porto_seguro_safe_driver_prediction': 'target',
    # 'costa_rican_household_poverty_level_prediction': 'Target',
    # 'forest_cover_type_prediction': 'Cover_Type',
    # 'santander_customer_transaction_prediction': 'target',
    # 'poisonous_mushroom_prediction': 'class',
    'bank_deposit_classification': 'y',
    'bank_churn_classification': 'Exited',
}


def identify_prediction_column(df, competition_type):
    """
    Identify the prediction column in the dataframe based on competition type.
    Reuses logic from eval_subs_all.py
    """
    
    if competition_type in ['binary-acc', 'multiclass', 'regression']:
        cols = list(df.columns)
        
        if len(cols) == 1:
            return cols[0]
        
        if len(cols) == 2:
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
                return c1
            elif t2 == 'numeric' and t1 != 'numeric':
                return c2
            elif t1 == 'numeric' and t2 == 'numeric':
                # Prefer the column with fewer unique values (likely predictions, not IDs)
                n = df.shape[0]
                uniq1, uniq2 = df[c1].nunique(dropna=True), df[c2].nunique(dropna=True)
                if uniq1 == n and uniq2 < n:
                    return c2
                elif uniq2 == n and uniq1 < n:
                    return c1
                else:
                    # Prefer float over int
                    if pd.api.types.is_float_dtype(df[c1]) and not pd.api.types.is_float_dtype(df[c2]):
                        return c1
                    elif pd.api.types.is_float_dtype(df[c2]) and not pd.api.types.is_float_dtype(df[c1]):
                        return c2
                    else:
                        # Choose second column (common Kaggle format: id, prediction)
                        return c2
            elif t1 in ('bool', 'bool-like'):
                return c1
            elif t2 in ('bool', 'bool-like'):
                return c2
            else:
                raise ValueError(f"Could not identify prediction column from {cols}")
        
        raise ValueError(f"Expected at most 2 columns for {competition_type}, found {len(cols)}: {cols}")
    
    elif competition_type == 'binary-probs':
        # Find float columns in [0, 1] range
        float_cols_in_01 = []
        for col in df.columns:
            if pd.api.types.is_float_dtype(df[col]):
                s = df[col].dropna()
                if s.size == 0:
                    continue
                # Require at least one value strictly between 0 and 1
                if ((s > 0.0 + 1e-9) & (s < 1.0 - 1e-9)).any():
                    float_cols_in_01.append(col)
        
        if len(float_cols_in_01) == 0:
            raise ValueError("No probability column found (float in [0,1] range)")
        
        if len(float_cols_in_01) == 1:
            return float_cols_in_01[0]
        
        # If multiple, prefer specific names
        if 'prob_class_1.0' in float_cols_in_01:
            return 'prob_class_1.0'
        elif 'prob_class_1' in float_cols_in_01:
            return 'prob_class_1'
        else:
            raise ValueError(f"Multiple probability columns found: {float_cols_in_01}")
    
    else:
        raise ValueError(f"Unknown competition type: {competition_type}")


def format_submission(input_csv, competition, output_csv=None, test_data_path=None):
    """
    Format a single submission CSV to proper Kaggle format.
    
    Args:
        input_csv: Path to agent-generated CSV
        competition: Competition name
        output_csv: Path to save formatted CSV (optional, defaults to input_kaggle.csv)
        test_data_path: Path to test.csv (optional, defaults to data/{competition}/test.csv)
    
    Returns:
        Path to formatted CSV file
    """
    
    if competition not in COMPETITION_TYPE:
        raise ValueError(f"Unknown competition: {competition}. Available: {list(COMPETITION_TYPE.keys())}")
    
    # Load test data to get IDs
    if test_data_path is None:
        test_data_path = f"data/{competition}/test.csv"
    
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")
    
    test_df = pd.read_csv(test_data_path)
    id_col = ID_COLUMN_MAP.get(competition, 'id')
    
    if id_col not in test_df.columns:
        raise ValueError(f"ID column '{id_col}' not found in test data. Columns: {test_df.columns.tolist()}")
    
    test_ids = test_df[id_col]
    
    # Load agent submission
    agent_df = pd.read_csv(input_csv)
    
    # Check size match
    if len(agent_df) != len(test_ids):
        raise ValueError(f"Size mismatch: agent CSV has {len(agent_df)} rows, test has {len(test_ids)} rows")
    
    # Identify prediction column
    comp_type = COMPETITION_TYPE[competition]
    pred_col = identify_prediction_column(agent_df, comp_type)
    
    print(f"  ✓ Identified prediction column: '{pred_col}'")
    
    predictions = agent_df[pred_col]
    
    # Validate predictions
    if predictions.isna().any():
        raise ValueError(f"Predictions contain {predictions.isna().sum()} NaN values")
    if np.isinf(predictions).any():
        raise ValueError(f"Predictions contain {np.isinf(predictions).sum()} infinite values")
    
    # Format based on competition type
    target_col = TARGET_COLUMN_MAP.get(competition, 'target')
    
    if comp_type == 'binary-acc':
        # Convert to proper format (True/False or 0/1 depending on original test data)
        # Check a sample submission if available, otherwise use boolean
        sample_path = f"data/{competition}/sample_submission.csv"
        if os.path.exists(sample_path):
            sample_df = pd.read_csv(sample_path)
            if target_col in sample_df.columns:
                # Match the dtype of sample submission
                if pd.api.types.is_bool_dtype(sample_df[target_col]):
                    predictions = predictions.astype(bool)
                else:
                    predictions = predictions.astype(int)
        else:
            # Default to boolean for binary-acc
            predictions = predictions.astype(bool)
    
    elif comp_type == 'binary-probs':
        # Ensure float and in [0, 1]
        predictions = predictions.astype(float)
        if (predictions < 0).any() or (predictions > 1).any():
            print(f"  ⚠ Warning: probabilities outside [0,1] range, clipping...")
            predictions = predictions.clip(0, 1)
    
    elif comp_type == 'regression':
        # Ensure numeric
        predictions = predictions.astype(float)
    
    elif comp_type == 'multiclass':
        # Keep as-is (should be class labels)
        pass
    
    # Create formatted submission
    formatted_df = pd.DataFrame({
        id_col: test_ids,
        target_col: predictions
    })
    
    # Determine output path
    if output_csv is None:
        input_path = Path(input_csv)
        output_csv = input_path.parent / f"{input_path.stem}_kaggle.csv"
    
    # Save
    formatted_df.to_csv(output_csv, index=False)
    print(f"  ✓ Saved formatted submission to: {output_csv}")
    print(f"    Format: {id_col} ({len(test_ids)} rows) + {target_col} ({comp_type})")
    
    return output_csv


def batch_format_directory(input_dir, competition, output_dir=None, pattern="*.csv"):
    """
    Format all CSV files in a directory.
    
    Args:
        input_dir: Directory containing agent CSVs
        competition: Competition name
        output_dir: Directory to save formatted CSVs (defaults to input_dir/kaggle_formatted/)
        pattern: File pattern to match (default: *.csv)
    
    Returns:
        List of formatted CSV paths
    """
    input_path = Path(input_dir)
    
    if output_dir is None:
        output_dir = input_path / "kaggle_formatted"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    csv_files = list(input_path.glob(pattern))
    
    # Filter out already formatted files
    csv_files = [f for f in csv_files if '_kaggle.csv' not in f.name and 'kaggle_formatted' not in str(f)]
    
    print(f"\n{'='*80}")
    print(f"Batch formatting {len(csv_files)} files from {input_dir}")
    print(f"Competition: {competition}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    formatted_files = []
    errors = []
    
    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")
        try:
            output_csv = output_path / f"{csv_file.stem}_kaggle.csv"
            formatted_path = format_submission(csv_file, competition, output_csv)
            formatted_files.append(formatted_path)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            errors.append((csv_file, str(e)))
    
    print(f"\n{'='*80}")
    print(f"BATCH FORMATTING COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully formatted: {len(formatted_files)}/{len(csv_files)}")
    
    if errors:
        print(f"\nErrors encountered ({len(errors)}):")
        for file, error in errors:
            print(f"  - {file.name}: {error}")
    
    return formatted_files


def batch_format_all_experiments(
    dir_prefix,
    output_dir,
    competitions,
    algorithms,
    target_files=None
):
    """
    Format all submissions across competitions and algorithms.
    Mirrors the structure of eval_subs_all.py
    
    Args:
        dir_prefix: Root directory containing submissions (e.g., 'submissions')
        output_dir: Root output directory for formatted files
        competitions: List of competition names
        algorithms: List of algorithm names
        target_files: Optional dict mapping algo -> list of specific files to process
    
    Returns:
        Summary statistics
    """
    print(f"\n{'='*80}")
    print(f"BATCH FORMATTING ALL EXPERIMENTS")
    print(f"{'='*80}")
    print(f"Input directory: {dir_prefix}")
    print(f"Output directory: {output_dir}")
    print(f"Competitions: {len(competitions)}")
    print(f"Algorithms: {algorithms}")
    if target_files:
        print(f"Target files specified for: {list(target_files.keys())}")
    print(f"{'='*80}\n")
    
    stats = {
        'total_processed': 0,
        'total_success': 0,
        'total_errors': 0,
        'by_competition': {},
        'by_algorithm': {}
    }
    
    for comp in competitions:
        print(f"\n{'='*80}")
        print(f"COMPETITION: {comp}")
        print(f"{'='*80}")
        
        if comp not in COMPETITION_TYPE:
            print(f"⚠ Unknown competition type, skipping")
            continue
        
        stats['by_competition'][comp] = {'success': 0, 'errors': 0}
        
        for algo in algorithms:
            input_path = os.path.join(dir_prefix, comp, algo)
            
            if not os.path.isdir(input_path):
                print(f"  Algorithm: {algo} - path missing, skipping")
                continue
            
            print(f"\n  Algorithm: {algo}")
            print(f"  Path: {input_path}")
            
            if algo not in stats['by_algorithm']:
                stats['by_algorithm'][algo] = {'success': 0, 'errors': 0}
            
            # Create output directory for this competition/algo
            algo_output_dir = os.path.join(output_dir, comp, algo)
            os.makedirs(algo_output_dir, exist_ok=True)
            
            # Get list of files to process
            files_to_process = []
            for file in os.listdir(input_path):
                # Apply same filtering logic as eval_subs_all.py
                if target_files and algo in target_files:
                    file_condition = file in target_files[algo]
                else:
                    # Default logic: react/react_code = all csvs, others = manual only
                    file_condition = file.endswith('.csv') and (
                        'react' in algo or 'manual' in file.lower()
                    )
                
                if file_condition:
                    files_to_process.append(file)
            
            print(f"    Files found: {len(files_to_process)}")
            
            # Process each file
            success_count = 0
            error_count = 0
            
            for file in files_to_process:
                input_csv = os.path.join(input_path, file)
                output_csv = os.path.join(algo_output_dir, f"{Path(file).stem}_kaggle.csv")
                
                try:
                    format_submission(input_csv, comp, output_csv)
                    success_count += 1
                    stats['total_success'] += 1
                    stats['by_competition'][comp]['success'] += 1
                    stats['by_algorithm'][algo]['success'] += 1
                except Exception as e:
                    print(f"    ✗ Error formatting {file}: {e}")
                    error_count += 1
                    stats['total_errors'] += 1
                    stats['by_competition'][comp]['errors'] += 1
                    stats['by_algorithm'][algo]['errors'] += 1
                
                stats['total_processed'] += 1
            
            print(f"    ✓ Success: {success_count}, ✗ Errors: {error_count}")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"FORMATTING SUMMARY")
    print(f"{'='*80}")
    print(f"Total files processed: {stats['total_processed']}")
    print(f"Successfully formatted: {stats['total_success']}")
    print(f"Errors: {stats['total_errors']}")
    
    print(f"\nBy Competition:")
    for comp, counts in stats['by_competition'].items():
        print(f"  {comp:50s} ✓ {counts['success']:3d}  ✗ {counts['errors']:3d}")
    
    print(f"\nBy Algorithm:")
    for algo, counts in stats['by_algorithm'].items():
        print(f"  {algo:50s} ✓ {counts['success']:3d}  ✗ {counts['errors']:3d}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Format agent submissions for Kaggle')
    parser.add_argument('--input', '-i', type=str, help='Input CSV file')
    parser.add_argument('--input_dir', '-d', type=str, help='Input directory (batch mode)')
    parser.add_argument('--competition', '-c', type=str, help='Competition name')
    parser.add_argument('--output', '-o', type=str, help='Output CSV file (single file mode)')
    parser.add_argument('--output_dir', type=str, help='Output directory (batch mode)')
    parser.add_argument('--auto', action='store_true', help='Auto-detect competition from path')
    parser.add_argument('--pattern', type=str, default='*.csv', help='File pattern for batch mode (default: *.csv)')
    parser.add_argument('--list', action='store_true', help='List available competitions')
    
    # Batch processing all experiments
    parser.add_argument('--batch_all', action='store_true', help='Process all competitions and algorithms')
    parser.add_argument('--dir_prefix', type=str, help='Full path to submissions directory (e.g., /path/to/exps_new/submissions)')
    parser.add_argument('--algorithms', '-a', type=str, nargs='+', 
                       help='List of algorithms to process (e.g., react react_code lats_reflection)')
    parser.add_argument('--competitions_list', type=str, nargs='+',
                       help='List of competitions to process (default: all)')
    
    args = parser.parse_args()
    
    if args.list:
        print("\nAvailable competitions:")
        for comp, comp_type in sorted(COMPETITION_TYPE.items()):
            print(f"  - {comp:50s} ({comp_type})")
        return
    
    # Batch process all experiments (like eval_subs_all.py)
    if args.batch_all:
        # Default algorithms
        default_algos = ['react', 'react_code', 'lats_reflection', 
                        'lats_hierarchical_rd_0_feedback', 
                        'lats_shaped_rewards_rd_0_feedback',
                        'lats_outcome_rewards_rd_0_feedback']
        
        algorithms = args.algorithms if args.algorithms else default_algos
        competitions = args.competitions_list if args.competitions_list else list(ID_COLUMN_MAP.keys())
        output_dir = args.output_dir if args.output_dir else 'formatted_submissions'
        
        stats = batch_format_all_experiments(
            dir_prefix=args.dir_prefix,
            output_dir=output_dir,
            competitions=competitions,
            algorithms=algorithms,
            target_files=None  # Can be extended to support TARGET_FILES like eval_subs_all.py
        )
        
        print(f"\n✓ Batch formatting complete!")
        return
    
    # Determine competition
    if args.auto:
        if args.input:
            path = args.input
        elif args.input_dir:
            path = args.input_dir
        else:
            print("Error: --auto requires --input or --input_dir")
            sys.exit(1)
        
        # Try to extract competition name from path
        competition = None
        for comp in COMPETITION_TYPE.keys():
            if comp in path:
                competition = comp
                break
        
        if not competition:
            print(f"Error: Could not auto-detect competition from path: {path}")
            print(f"Available competitions: {list(ID_COLUMN_MAP.keys())}")
            sys.exit(1)
        
        print(f"Auto-detected competition: {competition}")
    else:
        competition = args.competition
        if not competition:
            print("Error: --competition required (or use --auto)")
            sys.exit(1)
    
    # Single file or batch mode
    if args.input:
        print(f"\nFormatting single file: {args.input}")
        print(f"Competition: {competition}\n")
        try:
            formatted_path = format_submission(args.input, competition, args.output)
            print(f"\n✓ Success!")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            sys.exit(1)
    
    elif args.input_dir:
        try:
            formatted_files = batch_format_directory(
                args.input_dir,
                competition,
                args.output_dir,
                args.pattern
            )
            print(f"\n✓ Batch formatting complete! {len(formatted_files)} files formatted.")
        except Exception as e:
            print(f"\n✗ Error: {e}")
            sys.exit(1)
    
    else:
        print("Error: Either --input or --input_dir required")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

