#!/usr/bin/env python3
"""
Extract and analyze reflection scores from LATS reflection trajectories
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def extract_reflection_scores():
    """Extract all reflection scores from LATS reflection trajectories"""
    
    reflection_dir = Path("/Users/ychittepu/Documents/Codes/DS-Tools-v2/trajectories_dump/traj_dump/lats_reflection")
    
    trajectory_scores = {}
    
    for json_file in reflection_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            traj = json.load(f)
            
        filename = json_file.name
        agent_id = traj.get('agent_id', filename)
        solution_found = traj.get('search_statistics', {}).get('solution_found', False)
        final_value = traj.get('solution_node_info', {}).get('value', 0)
        
        # Extract all reflection scores from trajectory
        reflection_scores = []
        if 'trajectory_steps' in traj:
            for step in traj['trajectory_steps']:
                if step.get('step_type') == 'reflection' and 'extracted_score' in step:
                    reflection_scores.append(step['extracted_score'])
        
        trajectory_scores[agent_id] = {
            'filename': filename,
            'solution_found': solution_found,
            'final_value': final_value,
            'reflection_scores': reflection_scores,
            'min_score': min(reflection_scores) if reflection_scores else 0,
            'max_score': max(reflection_scores) if reflection_scores else 0,
            'avg_score': sum(reflection_scores) / len(reflection_scores) if reflection_scores else 0,
            'num_reflections': len(reflection_scores)
        }
    
    return trajectory_scores

def analyze_score_distributions(trajectory_scores):
    """Analyze and display score distributions"""
    
    print("🎯 LATS REFLECTION SCORE DISTRIBUTIONS")
    print("=" * 50)
    
    for agent_id, data in trajectory_scores.items():
        status = "✅ SUCCESS" if data['solution_found'] else "❌ FAILED"
        print(f"\n📊 {agent_id} ({status})")
        print(f"   Final Value: {data['final_value']}")
        print(f"   Number of Reflections: {data['num_reflections']}")
        print(f"   Score Range: {data['min_score']:.1f} - {data['max_score']:.1f}")
        print(f"   Average Score: {data['avg_score']:.1f}")
        print(f"   All Scores: {data['reflection_scores']}")
    
    # Overall statistics
    print(f"\n📈 OVERALL SCORE ANALYSIS:")
    
    all_scores = []
    successful_scores = []
    failed_scores = []
    
    for agent_id, data in trajectory_scores.items():
        all_scores.extend(data['reflection_scores'])
        if data['solution_found']:
            successful_scores.extend(data['reflection_scores'])
        else:
            failed_scores.extend(data['reflection_scores'])
    
    if all_scores:
        print(f"   All Reflection Scores: {sorted(all_scores)}")
        print(f"   Overall Range: {min(all_scores):.1f} - {max(all_scores):.1f}")
        print(f"   Overall Average: {sum(all_scores) / len(all_scores):.1f}")
    
    if successful_scores:
        print(f"   ✅ Success Scores: {sorted(successful_scores)}")
        print(f"   Success Average: {sum(successful_scores) / len(successful_scores):.1f}")
    
    if failed_scores:
        print(f"   ❌ Failed Scores: {sorted(failed_scores)}")
        print(f"   Failed Average: {sum(failed_scores) / len(failed_scores):.1f}")
    
    # Score frequency analysis
    print(f"\n🔢 SCORE FREQUENCY DISTRIBUTION:")
    from collections import Counter
    score_counts = Counter(all_scores)
    for score in sorted(score_counts.keys()):
        print(f"   Score {score:.1f}: {score_counts[score]} occurrences")
    
    return successful_scores, failed_scores

def create_histograms(successful_scores, failed_scores):
    """Create histograms for successful and failed trajectory scores"""
    
    plt.style.use('default')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('LATS Reflection Score Distributions', fontsize=16, fontweight='bold')
    
    # Histogram bins from 1 to 8
    bins = np.arange(0.5, 8.6, 1.0)
    
    # Successful trajectory histogram
    ax1.hist(successful_scores, bins=bins, alpha=0.7, color='green', edgecolor='black')
    ax1.set_title('✅ Successful Trajectory\nScore Distribution', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Reflection Score')
    ax1.set_ylabel('Frequency')
    ax1.set_xticks(range(1, 9))
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    success_avg = np.mean(successful_scores)
    success_std = np.std(successful_scores)
    ax1.text(0.05, 0.95, f'Avg: {success_avg:.1f}\nStd: {success_std:.1f}\nN: {len(successful_scores)}', 
             transform=ax1.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Failed trajectory histogram
    ax2.hist(failed_scores, bins=bins, alpha=0.7, color='red', edgecolor='black')
    ax2.set_title('❌ Failed Trajectory\nScore Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Reflection Score')
    ax2.set_ylabel('Frequency')
    ax2.set_xticks(range(1, 9))
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    failed_avg = np.mean(failed_scores)
    failed_std = np.std(failed_scores)
    ax2.text(0.05, 0.95, f'Avg: {failed_avg:.1f}\nStd: {failed_std:.1f}\nN: {len(failed_scores)}', 
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Make y-axis scales the same for comparison
    max_freq = max(max(np.histogram(successful_scores, bins=bins)[0]), 
                   max(np.histogram(failed_scores, bins=bins)[0]))
    ax1.set_ylim(0, max_freq + 1)
    ax2.set_ylim(0, max_freq + 1)
    
    plt.tight_layout()
    plt.savefig('/Users/ychittepu/Documents/Codes/DS-Tools-v2/lats_reflection_score_histograms.png', 
                dpi=300, bbox_inches='tight')
    print("\n📊 Histograms saved: lats_reflection_score_histograms.png")
    plt.show()
    
    # Additional comparison plot
    fig2, ax3 = plt.subplots(1, 1, figsize=(12, 6))
    
    # Overlapping histograms
    ax3.hist(successful_scores, bins=bins, alpha=0.6, color='green', label='✅ Success', edgecolor='black')
    ax3.hist(failed_scores, bins=bins, alpha=0.6, color='red', label='❌ Failed', edgecolor='black')
    
    ax3.set_title('LATS Reflection Scores: Success vs Failed Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Reflection Score')
    ax3.set_ylabel('Frequency')
    ax3.set_xticks(range(1, 9))
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add vertical lines for averages
    ax3.axvline(success_avg, color='green', linestyle='--', linewidth=2, label=f'Success Avg: {success_avg:.1f}')
    ax3.axvline(failed_avg, color='red', linestyle='--', linewidth=2, label=f'Failed Avg: {failed_avg:.1f}')
    
    # Update legend
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('/Users/ychittepu/Documents/Codes/DS-Tools-v2/lats_reflection_score_comparison.png', 
                dpi=300, bbox_inches='tight')
    print("📊 Comparison plot saved: lats_reflection_score_comparison.png")
    plt.show()

def main():
    trajectory_scores = extract_reflection_scores()
    successful_scores, failed_scores = analyze_score_distributions(trajectory_scores)
    create_histograms(successful_scores, failed_scores)

if __name__ == "__main__":
    main()
