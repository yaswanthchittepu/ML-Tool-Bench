#!/usr/bin/env python3
"""
Trajectory Cost Calculator
Extracts and summarizes total costs from all trajectory JSON files
"""

import json
import os
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


def analyze_trajectory_costs(trajectories_dir):
    """
    Analyze costs from all trajectory files
    
    Parameters:
    -----------
    trajectories_dir : str
        Path to the trajectories_dump/traj_dump directory
        
    Returns:
    --------
    dict : Cost analysis results
    """
    
    base_dir = Path(trajectories_dir)
    if not base_dir.exists():
        raise ValueError(f"Directory not found: {trajectories_dir}")
    
    # Algorithm-specific cost extractors
    cost_extractors = {
        'react': extract_cost_from_react,
        'lats_reflection': extract_cost_from_lats,
        'lats_shaped': extract_cost_from_lats,
        'lats_outcome': extract_cost_from_lats,
        'hmcts': extract_cost_from_hmcts,
    }
    
    results = {
        'algorithm_costs': defaultdict(list),
        'trajectory_details': [],
        'algorithm_summaries': {},
        'total_cost': 0.0,
        'total_tokens': 0
    }
    
    print("🔍 ANALYZING TRAJECTORY COSTS")
    print("=" * 40)
    
    # Process each algorithm directory
    for algo_dir in base_dir.iterdir():
        if not algo_dir.is_dir():
            continue
            
        algo_name = algo_dir.name
        print(f"\n📂 Processing {algo_name}...")
        
        # Find appropriate cost extractor
        extractor = cost_extractors.get(algo_name, extract_cost_from_lats)  # Default to LATS format
        
        algo_total_cost = 0.0
        algo_total_tokens = 0
        trajectory_count = 0
        
        # Process each JSON file in the algorithm directory
        for json_file in algo_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    traj_data = json.load(f)
                
                # Extract cost using appropriate extractor
                cost, tokens = extractor(traj_data)
                
                # Store individual trajectory details
                traj_detail = {
                    'algorithm': algo_name,
                    'filename': json_file.name,
                    'cost': cost,
                    'tokens': tokens
                }
                results['trajectory_details'].append(traj_detail)
                results['algorithm_costs'][algo_name].append(cost)
                
                # Add to algorithm totals
                algo_total_cost += cost
                algo_total_tokens += tokens
                trajectory_count += 1
                
                print(f"  📄 {json_file.name}: ${cost:.4f} ({tokens:,} tokens)")
                
            except Exception as e:
                print(f"  ❌ Error processing {json_file.name}: {e}")
                continue
        
        # Calculate algorithm summary
        if trajectory_count > 0:
            avg_cost = algo_total_cost / trajectory_count
            avg_tokens = algo_total_tokens / trajectory_count
            
            results['algorithm_summaries'][algo_name] = {
                'trajectory_count': trajectory_count,
                'total_cost': algo_total_cost,
                'total_tokens': algo_total_tokens,
                'average_cost': avg_cost,
                'average_tokens': avg_tokens
            }
            
            print(f"  📊 {algo_name} Summary:")
            print(f"     Trajectories: {trajectory_count}")
            print(f"     Total Cost: ${algo_total_cost:.4f}")
            print(f"     Average Cost: ${avg_cost:.4f}")
            print(f"     Total Tokens: {algo_total_tokens:,}")
            print(f"     Average Tokens: {avg_tokens:,.0f}")
        
        # Add to overall totals
        results['total_cost'] += algo_total_cost
        results['total_tokens'] += algo_total_tokens
    
    return results


def generate_cost_report(results, output_dir="./eval_results"):
    """Generate detailed cost reports"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create detailed trajectory DataFrame
    df_details = pd.DataFrame(results['trajectory_details'])
    
    # Create algorithm summary DataFrame
    summary_data = []
    for algo, stats in results['algorithm_summaries'].items():
        summary_data.append({
            'algorithm': algo,
            'trajectory_count': stats['trajectory_count'],
            'total_cost': stats['total_cost'],
            'average_cost': stats['average_cost'],
            'total_tokens': stats['total_tokens'],
            'average_tokens': stats['average_tokens']
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Save to CSV files
    import datetime
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    details_file = os.path.join(output_dir, f'trajectory_costs_details_{timestamp}.csv')
    summary_file = os.path.join(output_dir, f'trajectory_costs_summary_{timestamp}.csv')
    
    df_details.to_csv(details_file, index=False)
    df_summary.to_csv(summary_file, index=False)
    
    print(f"\n💾 Reports saved:")
    print(f"   Details: {details_file}")
    print(f"   Summary: {summary_file}")
    
    return df_details, df_summary


def print_cost_summary(results):
    """Print a formatted cost summary"""
    
    print("\n" + "=" * 50)
    print("💰 TRAJECTORY COST SUMMARY")
    print("=" * 50)
    
    # Algorithm ranking by total cost
    algo_costs = [(algo, stats['total_cost']) for algo, stats in results['algorithm_summaries'].items()]
    algo_costs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n🏆 COST RANKING (Total):")
    for i, (algo, cost) in enumerate(algo_costs, 1):
        count = results['algorithm_summaries'][algo]['trajectory_count']
        avg_cost = results['algorithm_summaries'][algo]['average_cost']
        print(f"{i:2d}. {algo:<35} ${cost:>8.4f} ({count} trajectories, avg: ${avg_cost:.4f})")
    
    # Algorithm ranking by average cost
    algo_avg_costs = [(algo, stats['average_cost']) for algo, stats in results['algorithm_summaries'].items()]
    algo_avg_costs.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n📊 EFFICIENCY RANKING (Average Cost):")
    for i, (algo, avg_cost) in enumerate(algo_avg_costs, 1):
        count = results['algorithm_summaries'][algo]['trajectory_count']
        total_cost = results['algorithm_summaries'][algo]['total_cost']
        print(f"{i:2d}. {algo:<35} ${avg_cost:>8.4f} per trajectory ({count} total)")
    
    # Overall summary
    print(f"\n🌟 OVERALL SUMMARY:")
    print(f"   Total Cost: ${results['total_cost']:.4f}")
    print(f"   Total Tokens: {results['total_tokens']:,}")
    print(f"   Total Trajectories: {len(results['trajectory_details'])}")
    print(f"   Average Cost per Trajectory: ${results['total_cost']/len(results['trajectory_details']):.4f}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate costs from trajectory JSON files')
    parser.add_argument('--trajectories_dir', 
                       default='/Users/ychittepu/Documents/Codes/DS-Tools-v2/trajectories_dump/traj_dump',
                       help='Path to trajectories directory')
    parser.add_argument('--output_dir', 
                       default='./eval_results', 
                       help='Output directory for reports')
    parser.add_argument('--no_csv', action='store_true', help='Skip CSV output generation')
    
    args = parser.parse_args()
    
    try:
        # Analyze costs
        results = analyze_trajectory_costs(args.trajectories_dir)
        
        # Print summary
        print_cost_summary(results)
        
        # Generate reports
        if not args.no_csv:
            generate_cost_report(results, args.output_dir)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
