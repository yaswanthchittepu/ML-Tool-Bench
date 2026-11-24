#!/usr/bin/env python3
"""
Experiment Cost Calculator
Extracts and summarizes total costs from all experiment pickle files
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd


def extract_cost_from_react(traj_data):
    """Extract cost from ReAct trajectory"""
    total_cost = 0.0
    total_tokens = 0
    
    # ReAct trajectories have a summary step at the end with total_cost
    if 'trajectory' in traj_data:
        for step in traj_data['trajectory']:
            if step.get('step_type') == 'execution_summary':
                # Found the summary step - use its totals
                total_cost = step.get('total_cost', 0.0)
                total_tokens = step.get('total_tokens', 0)
                return total_cost, total_tokens
    
    # Fallback: sum individual step costs if no summary found
    if 'trajectory' in traj_data:
        for step in traj_data['trajectory']:
            if 'cost' in step:
                total_cost += step['cost']
            if 'tokens_used' in step:
                total_tokens += step['tokens_used']
    
    return total_cost, total_tokens


def extract_cost_from_lats(traj_data):
    """Extract cost from LATS variants (reflection, shaped, outcome)"""
    total_cost = 0.0
    total_tokens = 0
    
    if 'search_statistics' in traj_data:
        stats = traj_data['search_statistics']
        total_cost = stats.get('total_cost', 0.0)
        total_tokens = stats.get('total_tokens', 0)
    
    return total_cost, total_tokens


def extract_cost_from_hmcts(traj_data):
    """Extract cost from HMCTS/Hierarchical trajectories"""
    total_cost = 0.0
    total_tokens = 0
    
    # Check if it has search_statistics (some hierarchical might)
    if 'search_statistics' in traj_data:
        stats = traj_data['search_statistics']
        total_cost = stats.get('total_cost', 0.0)
        total_tokens = stats.get('total_tokens', 0)
    
    # If no search_statistics, try to extract from stage trajectories
    elif 'stage_trajectories' in traj_data:
        for stage_name, stage_data in traj_data['stage_trajectories'].items():
            # Look for cost info in stage data
            if isinstance(stage_data, dict):
                if 'cost' in stage_data:
                    total_cost += stage_data['cost']
                if 'tokens' in stage_data:
                    total_tokens += stage_data['tokens']
                
                # Check if steps have cost info
                if 'steps' in stage_data:
                    for step in stage_data['steps']:
                        if isinstance(step, dict):
                            if 'cost' in step:
                                total_cost += step['cost']
                            if 'tokens_used' in step:
                                total_tokens += step['tokens_used']
    
    return total_cost, total_tokens


def analyze_experiment_costs(base_dir):
    """
    Analyze costs from all experiment pickle files
    
    Parameters:
    -----------
    base_dir : str
        Path to the base experiments directory
        
    Returns:
    --------
    dict : Cost analysis results
    """
    
    if not os.path.exists(base_dir):
        raise ValueError(f"Directory not found: {base_dir}")
    
    print("🔍 ANALYZING EXPERIMENT COSTS")
    print("=" * 50)
    print(f"📂 Base Directory: {base_dir}")
    
    costs_map = defaultdict(dict)
    results = {
        'experiment_details': [],
        'algorithm_summaries': defaultdict(lambda: {'total_cost': 0.0, 'experiment_count': 0, 'trajectory_count': 0}),
        'experiment_summaries': {},
        'total_cost': 0.0,
        'total_experiments': 0,
        'total_algorithms': 0
    }
    
    # Algorithm-specific cost extractors
    cost_extractors = {
        'react': extract_cost_from_react,
        'lats_reflection': extract_cost_from_lats,
        'lats_shaped': extract_cost_from_lats,
        'lats_outcome': extract_cost_from_lats,
        'hmcts': extract_cost_from_hmcts,
    }
    
    # Get all experiments
    exps = [exp for exp in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, exp))]
    exps.sort()  # Sort for consistent ordering
    
    print(f"📊 Found {len(exps)} experiments")
    print()
    
    for exp in exps:
        exp_path = os.path.join(base_dir, exp)
        print(f"🧪 Processing experiment: {exp}")
        
        exp_total_cost = 0.0
        exp_algo_count = 0
        
        # Get all algorithms in this experiment
        algos = [algo for algo in os.listdir(exp_path) if os.path.isdir(os.path.join(exp_path, algo))]
        algos.sort()  # Sort for consistent ordering
        
        for algo in algos:
            algo_path = os.path.join(exp_path, algo)
            
            try:
                # Get all JSON files in the algorithm directory
                json_files = [f for f in os.listdir(algo_path) if f.endswith('.json')]
                
                if len(json_files) == 0:
                    print(f"  ❌ No JSON files found in {algo_path}")
                    continue
                
                # Find appropriate cost extractor
                extractor = cost_extractors.get(algo, extract_cost_from_lats)  # Default to LATS format
                
                # Extract costs from all JSON files
                costs = []
                processed_files = []
                
                for json_file in json_files:
                    json_path = os.path.join(algo_path, json_file)
                    
                    try:
                        with open(json_path, 'r') as f:
                            traj_data = json.load(f)
                        
                        # Extract cost using appropriate extractor
                        cost, tokens = extractor(traj_data)
                        
                        if cost > 0:  # Only include trajectories with valid cost data
                            costs.append(cost)
                            processed_files.append(json_file)
                    
                    except Exception as e:
                        print(f"  ⚠️  Error processing {json_file}: {e}")
                        continue
                
                if len(costs) == 0:
                    print(f"  ❌ No valid cost data found for {algo}")
                    continue
                
                # Validate expected number of trajectories (flexible - warn but don't fail)
                expected_trajectories = 10
                if len(costs) != expected_trajectories:
                    print(f"  ⚠️  Expected ~{expected_trajectories} trajectories for {algo}, found {len(costs)}")
                
                # Calculate total cost for this algorithm
                algo_total_cost = sum(costs)
                costs_map[exp][algo] = algo_total_cost
                
                # Store detailed information
                exp_detail = {
                    'experiment': exp,
                    'algorithm': algo,
                    'trajectory_count': len(costs),
                    'total_cost': algo_total_cost,
                    'average_cost': algo_total_cost / len(costs) if costs else 0.0,
                    'min_cost': min(costs) if costs else 0.0,
                    'max_cost': max(costs) if costs else 0.0,
                    'json_files': len(json_files),
                    'processed_files': len(processed_files)
                }
                results['experiment_details'].append(exp_detail)
                
                # Update algorithm summaries
                results['algorithm_summaries'][algo]['total_cost'] += algo_total_cost
                results['algorithm_summaries'][algo]['experiment_count'] += 1
                results['algorithm_summaries'][algo]['trajectory_count'] += len(costs)
                
                # Add to experiment totals
                exp_total_cost += algo_total_cost
                exp_algo_count += 1
                
                print(f"  ✅ {algo}: ${algo_total_cost:.4f} ({len(costs)} trajectories from {len(processed_files)}/{len(json_files)} files, avg: ${algo_total_cost/len(costs):.4f})")
                    
            except Exception as e:
                print(f"  ❌ Error processing {algo}: {e}")
                continue
        
        # Store experiment summary
        if exp_algo_count > 0:
            results['experiment_summaries'][exp] = {
                'algorithm_count': exp_algo_count,
                'total_cost': exp_total_cost,
                'average_cost_per_algorithm': exp_total_cost / exp_algo_count
            }
            
            results['total_cost'] += exp_total_cost
            results['total_experiments'] += 1
            
            print(f"  📊 {exp} Summary: {exp_algo_count} algorithms, ${exp_total_cost:.4f} total")
        
        print()
    
    # Calculate final statistics
    results['total_algorithms'] = len(results['algorithm_summaries'])
    
    # Convert algorithm summaries to regular dict for easier handling
    results['algorithm_summaries'] = dict(results['algorithm_summaries'])
    
    # Calculate average cost per algorithm across all experiments
    for algo, stats in results['algorithm_summaries'].items():
        if stats['experiment_count'] > 0:
            stats['average_cost_per_experiment'] = stats['total_cost'] / stats['experiment_count']
            stats['average_cost_per_trajectory'] = stats['total_cost'] / stats['trajectory_count'] if stats['trajectory_count'] > 0 else 0.0
    
    return results, costs_map


def print_experiment_cost_summary(results, costs_map):
    """Print a formatted cost summary for experiments"""
    
    print("=" * 60)
    print("💰 EXPERIMENT COST ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Overall statistics
    print(f"\n🌟 OVERALL STATISTICS:")
    print(f"   Total Experiments: {results['total_experiments']}")
    print(f"   Total Algorithms: {results['total_algorithms']}")
    print(f"   Total Cost: ${results['total_cost']:.4f}")
    if results['total_experiments'] > 0:
        print(f"   Average Cost per Experiment: ${results['total_cost']/results['total_experiments']:.4f}")
    
    # Algorithm ranking by total cost across all experiments
    print(f"\n🏆 ALGORITHM COST RANKING (Total Across All Experiments):")
    algo_costs = [(algo, stats['total_cost']) for algo, stats in results['algorithm_summaries'].items()]
    algo_costs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (algo, total_cost) in enumerate(algo_costs, 1):
        stats = results['algorithm_summaries'][algo]
        exp_count = stats['experiment_count']
        avg_per_exp = stats['average_cost_per_experiment']
        avg_per_traj = stats['average_cost_per_trajectory']
        print(f"{i:2d}. {algo:<40} ${total_cost:>8.4f} ({exp_count} exps, avg: ${avg_per_exp:.4f}/exp, ${avg_per_traj:.4f}/traj)")
    
    # Experiment ranking by total cost
    print(f"\n📊 EXPERIMENT COST RANKING:")
    exp_costs = [(exp, stats['total_cost']) for exp, stats in results['experiment_summaries'].items()]
    exp_costs.sort(key=lambda x: x[1], reverse=True)
    
    for i, (exp, total_cost) in enumerate(exp_costs, 1):
        stats = results['experiment_summaries'][exp]
        algo_count = stats['algorithm_count']
        avg_per_algo = stats['average_cost_per_algorithm']
        print(f"{i:2d}. {exp:<40} ${total_cost:>8.4f} ({algo_count} algos, avg: ${avg_per_algo:.4f}/algo)")
    
    # Cost matrix (experiments vs algorithms)
    print(f"\n📋 COST MATRIX (Experiments vs Algorithms):")
    print("=" * 60)
    
    # Get all algorithms and experiments
    all_algorithms = sorted(results['algorithm_summaries'].keys())
    all_experiments = sorted(results['experiment_summaries'].keys())
    
    # Print header
    print(f"{'Experiment':<30}", end="")
    for algo in all_algorithms:
        print(f"{algo[:15]:>15}", end="")
    print(f"{'Total':>15}")
    print("-" * (30 + len(all_algorithms) * 15 + 15))
    
    # Print each experiment row
    for exp in all_experiments:
        print(f"{exp:<30}", end="")
        exp_total = 0.0
        for algo in all_algorithms:
            cost = costs_map[exp].get(algo, 0.0)
            exp_total += cost
            if cost > 0:
                print(f"{cost:>15.4f}", end="")
            else:
                print(f"{'—':>15}", end="")
        print(f"{exp_total:>15.4f}")
    
    # Print totals row
    print("-" * (30 + len(all_algorithms) * 15 + 15))
    print(f"{'TOTAL':<30}", end="")
    grand_total = 0.0
    for algo in all_algorithms:
        algo_total = results['algorithm_summaries'][algo]['total_cost']
        grand_total += algo_total
        print(f"{algo_total:>15.4f}", end="")
    print(f"{grand_total:>15.4f}")


def generate_experiment_cost_report(results, costs_map, output_dir="./eval_results"):
    """Generate detailed experiment cost reports"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create detailed DataFrame
    df_details = pd.DataFrame(results['experiment_details'])
    
    # Create algorithm summary DataFrame
    algo_summary_data = []
    for algo, stats in results['algorithm_summaries'].items():
        algo_summary_data.append({
            'algorithm': algo,
            'experiment_count': stats['experiment_count'],
            'trajectory_count': stats['trajectory_count'],
            'total_cost': stats['total_cost'],
            'average_cost_per_experiment': stats['average_cost_per_experiment'],
            'average_cost_per_trajectory': stats['average_cost_per_trajectory']
        })
    df_algo_summary = pd.DataFrame(algo_summary_data)
    
    # Create experiment summary DataFrame
    exp_summary_data = []
    for exp, stats in results['experiment_summaries'].items():
        exp_summary_data.append({
            'experiment': exp,
            'algorithm_count': stats['algorithm_count'],
            'total_cost': stats['total_cost'],
            'average_cost_per_algorithm': stats['average_cost_per_algorithm']
        })
    df_exp_summary = pd.DataFrame(exp_summary_data)
    
    # Create cost matrix DataFrame
    all_algorithms = sorted(results['algorithm_summaries'].keys())
    all_experiments = sorted(results['experiment_summaries'].keys())
    
    cost_matrix_data = []
    for exp in all_experiments:
        row = {'experiment': exp}
        for algo in all_algorithms:
            row[algo] = costs_map[exp].get(algo, 0.0)
        row['total'] = sum(costs_map[exp].values())
        cost_matrix_data.append(row)
    df_cost_matrix = pd.DataFrame(cost_matrix_data)
    
    # Save to CSV files
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    details_file = os.path.join(output_dir, f'experiment_costs_details_{timestamp}.csv')
    algo_summary_file = os.path.join(output_dir, f'experiment_costs_algo_summary_{timestamp}.csv')
    exp_summary_file = os.path.join(output_dir, f'experiment_costs_exp_summary_{timestamp}.csv')
    matrix_file = os.path.join(output_dir, f'experiment_costs_matrix_{timestamp}.csv')
    
    df_details.to_csv(details_file, index=False)
    df_algo_summary.to_csv(algo_summary_file, index=False)
    df_exp_summary.to_csv(exp_summary_file, index=False)
    df_cost_matrix.to_csv(matrix_file, index=False)
    
    print(f"\n💾 Reports saved:")
    print(f"   Details: {details_file}")
    print(f"   Algorithm Summary: {algo_summary_file}")
    print(f"   Experiment Summary: {exp_summary_file}")
    print(f"   Cost Matrix: {matrix_file}")
    
    return df_details, df_algo_summary, df_exp_summary, df_cost_matrix


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate costs from experiment JSON files')
    parser.add_argument('--base_dir', 
                       default='/sensei-fs-3/users/ychittepu/rebuttals/gpt-4p1-mini/logs',
                       help='Path to base experiments directory')
    parser.add_argument('--output_dir', 
                       default='./eval_results', 
                       help='Output directory for reports')
    parser.add_argument('--no_csv', action='store_true', help='Skip CSV output generation')
    
    args = parser.parse_args()
    
    try:
        # Analyze costs
        results, costs_map = analyze_experiment_costs(args.base_dir)
        
        # Print summary
        print_experiment_cost_summary(results, costs_map)
        
        # Generate reports
        if not args.no_csv:
            generate_experiment_cost_report(results, costs_map, args.output_dir)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
