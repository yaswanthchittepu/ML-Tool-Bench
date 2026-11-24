#!/usr/bin/env python3
"""
Trajectory Comparison Analysis Script
Compares trajectories from different algorithms: ReAct, LATS variants, and Hierarchical MCTS
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class TrajectoryAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.algorithms = {}
        self.load_all_trajectories()
    
    def load_all_trajectories(self):
        """Load trajectories from all algorithm folders"""
        for algo_dir in self.base_dir.iterdir():
            if algo_dir.is_dir():
                algo_name = algo_dir.name
                self.algorithms[algo_name] = []
                
                for json_file in algo_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r') as f:
                            trajectory = json.load(f)
                            trajectory['filename'] = json_file.name
                            self.algorithms[algo_name].append(trajectory)
                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")
    
    def extract_key_metrics(self):
        """Extract key metrics for comparison"""
        metrics = []
        
        for algo_name, trajectories in self.algorithms.items():
            for traj in trajectories:
                metric = {
                    'algorithm': algo_name,
                    'filename': traj.get('filename', 'unknown')
                }
                
                # Extract tool call counts
                tool_calls_info = self.count_tool_calls(traj, algo_name)
                metric.update(tool_calls_info)
                
                # Common metrics
                if 'search_statistics' in traj:
                    stats = traj['search_statistics']
                    metric.update({
                        'total_time': stats.get('total_search_time', 0),
                        'total_tokens': stats.get('total_tokens', 0),
                        'total_cost': stats.get('total_cost', 0),
                        'tree_height': stats.get('tree_height', 0),
                        'solution_found': stats.get('solution_found', False),
                        'model_used': stats.get('model_used', 'unknown')
                    })
                
                # ReAct specific
                if 'total_steps' in traj:
                    metric.update({
                        'total_steps': traj.get('total_steps', 0),
                        'total_time': (pd.to_datetime(traj.get('end_time', '')) - 
                                     pd.to_datetime(traj.get('start_time', ''))).total_seconds() if 'end_time' in traj else 0
                    })
                
                # LATS reward specific
                if 'trajectory_rewards' in traj:
                    metric['trajectory_rewards'] = traj['trajectory_rewards']
                    metric['reward_type'] = traj.get('search_statistics', {}).get('reward_type', 'unknown')
                
                # Hierarchical specific
                if 'num_stages' in traj:
                    metric.update({
                        'num_stages': traj['num_stages'],
                        'stages_completed': traj['stages_completed'],
                        'completion_rate': traj['stages_completed'] / traj['num_stages']
                    })
                
                # Solution quality
                if 'solution_node_info' in traj:
                    sol_info = traj['solution_node_info']
                    metric.update({
                        'solution_depth': sol_info.get('depth', 0),
                        'solution_value': sol_info.get('value', 0),
                        'solution_visits': sol_info.get('visits', 0)
                    })
                
                metrics.append(metric)
        
        return pd.DataFrame(metrics)
    
    def count_tool_calls(self, traj, algo_name):
        """Count tool calls for different algorithm types"""
        tool_info = {
            'total_tool_calls': 0,
            'tool_calls_per_subtask': {}
        }
        
        if algo_name == 'react':
            # ReAct: Count tool execution initiations
            if 'trajectory' in traj:
                tool_info['total_tool_calls'] = len([
                    step for step in traj['trajectory'] 
                    if step.get('step_type') == 'tool_execution_initiation'
                ])
        
        elif algo_name in ['lats_reflection', 'lats_shaped', 'lats_outcome']:
            # LATS variants: Count tool selections
            if 'trajectory_steps' in traj:
                tool_info['total_tool_calls'] = len([
                    step for step in traj['trajectory_steps'] 
                    if step.get('step_type') == 'tool_selection'
                ])
        
        elif algo_name == 'hmcts':
            # HMCTS: Count tool calls per subtask and total
            total_tools = 0
            subtask_tools = {}
            
            if 'stage_trajectories' in traj:
                for stage_name, stage_data in traj['stage_trajectories'].items():
                    stage_tool_count = 0
                    if 'steps' in stage_data:
                        stage_tool_count = len([
                            step for step in stage_data['steps'] 
                            if step.get('step_type') == 'tool_selection'
                        ])
                    
                    subtask_tools[stage_name] = stage_tool_count
                    total_tools += stage_tool_count
            
            tool_info['total_tool_calls'] = total_tools
            tool_info['tool_calls_per_subtask'] = subtask_tools
        
        return tool_info
    
    def analyze_step_patterns(self):
        """Analyze step patterns across algorithms"""
        step_patterns = defaultdict(lambda: defaultdict(int))
        
        for algo_name, trajectories in self.algorithms.items():
            for traj in trajectories:
                # Count step types
                if 'step_types' in traj:  # ReAct
                    for step_type, count in traj['step_types'].items():
                        step_patterns[algo_name][step_type] += count
                
                elif 'trajectory_steps' in traj:  # LATS variants
                    for step in traj['trajectory_steps']:
                        step_type = step.get('step_type', 'unknown')
                        step_patterns[algo_name][step_type] += 1
                
                elif 'stage_trajectories' in traj:  # Hierarchical
                    for stage_name, stage_data in traj['stage_trajectories'].items():
                        if 'steps' in stage_data:
                            for step in stage_data['steps']:
                                step_type = step.get('step_type', 'unknown')
                                step_patterns[algo_name][step_type] += 1
        
        return dict(step_patterns)
    
    def compare_efficiency_metrics(self):
        """Compare efficiency across algorithms"""
        df = self.extract_key_metrics()
        
        if df.empty:
            print("No metrics data available for comparison")
            return
            
        print("=== EFFICIENCY COMPARISON ===")
        
        # Success rates
        print("\n📊 Success Rates:")
        success_rates = df.groupby('algorithm')['solution_found'].agg(['count', 'sum', 'mean']).round(3)
        success_rates.columns = ['total_runs', 'successes', 'success_rate']
        print(success_rates)
        
        # Tool call analysis
        print("\n🔧 Tool Call Analysis:")
        if 'total_tool_calls' in df.columns:
            tool_stats = df.groupby('algorithm')['total_tool_calls'].agg(['mean', 'median', 'std']).round(1)
            print(tool_stats)
        
        # Time and cost efficiency
        print("\n⏱️ Time & Cost Efficiency:")
        efficiency_cols = ['total_time', 'total_tokens', 'total_cost']
        for col in efficiency_cols:
            if col in df.columns:
                print(f"\n{col.upper()}:")
                stats = df.groupby('algorithm')[col].agg(['mean', 'median', 'std']).round(3)
                print(stats)
        
        # Tree characteristics
        print("\n🌳 Tree Search Characteristics:")
        tree_cols = ['tree_height', 'solution_depth']
        for col in tree_cols:
            if col in df.columns:
                print(f"\n{col.upper()}:")
                stats = df.groupby('algorithm')[col].agg(['mean', 'median', 'std']).round(3)
                print(stats)
        
        # Reward analysis
        if 'trajectory_rewards' in df.columns:
            print("\n🎯 Reward Analysis:")
            reward_stats = df.groupby(['algorithm', 'reward_type'])['trajectory_rewards'].agg(['mean', 'median', 'std']).round(3)
            print(reward_stats)
            
        return df
    
    def analyze_tool_usage_patterns(self):
        """Detailed analysis of tool usage patterns"""
        print("\n=== DETAILED TOOL USAGE ANALYSIS ===")
        
        df = self.extract_key_metrics()
        
        for algo_name, trajectories in self.algorithms.items():
            print(f"\n🔧 {algo_name.upper()} - Tool Usage Breakdown:")
            
            for traj in trajectories:
                filename = traj.get('filename', 'unknown')
                tool_calls_info = self.count_tool_calls(traj, algo_name)
                
                print(f"  📁 {filename}")
                print(f"     Total Tool Calls: {tool_calls_info['total_tool_calls']}")
                
                # Special handling for HMCTS subtask breakdown
                if algo_name == 'hmcts' and tool_calls_info['tool_calls_per_subtask']:
                    print(f"     📊 Subtask Breakdown:")
                    for subtask, count in tool_calls_info['tool_calls_per_subtask'].items():
                        print(f"       • {subtask}: {count} tool calls")
                    
                    # Calculate efficiency metrics
                    if 'num_stages' in traj and 'stages_completed' in traj:
                        total_stages = traj['num_stages']
                        completed_stages = traj['stages_completed']
                        avg_tools_per_stage = tool_calls_info['total_tool_calls'] / completed_stages if completed_stages > 0 else 0
                        print(f"     📈 Efficiency: {avg_tools_per_stage:.1f} tools/completed_stage")
                        print(f"     ✅ Completion: {completed_stages}/{total_stages} stages ({100*completed_stages/total_stages:.1f}%)")
                
                # Success correlation
                success = False
                if 'search_statistics' in traj:
                    success = traj['search_statistics'].get('solution_found', False)
                elif 'solution_found' in traj:
                    success = traj['solution_found']
                
                success_indicator = "✅ SUCCESS" if success else "❌ FAILED"
                print(f"     Result: {success_indicator}")
                
                print()  # Empty line for readability
    
    def analyze_solution_paths(self):
        """Analyze solution path characteristics"""
        print("\n=== SOLUTION PATH ANALYSIS ===")
        
        for algo_name, trajectories in self.algorithms.items():
            successful_trajectories = [t for t in trajectories if t.get('search_statistics', {}).get('solution_found', False) or t.get('solution_found', False)]
            
            if successful_trajectories:
                print(f"\n🎯 {algo_name.upper()} - Successful Solutions:")
                for traj in successful_trajectories:
                    filename = traj.get('filename', 'unknown')
                    
                    # Get solution characteristics
                    if 'solution_node_info' in traj:
                        info = traj['solution_node_info']
                        print(f"  📁 {filename}")
                        print(f"     Depth: {info.get('depth', 'N/A')}, Value: {info.get('value', 'N/A')}")
                        if 'total_reward' in info:
                            print(f"     Total Reward: {info['total_reward']:.3f}")
                    
                    # Hierarchical specific
                    if 'stages_completed' in traj:
                        print(f"     Stages: {traj['stages_completed']}/{traj['num_stages']}")
            else:
                print(f"\n❌ {algo_name.upper()} - No successful solutions found")
    
    def visualize_comparisons(self, df):
        """Create visualization comparing algorithms"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16)
        
        # Success rate comparison
        if 'solution_found' in df.columns:
            success_data = df.groupby('algorithm')['solution_found'].mean()
            axes[0,0].bar(success_data.index, success_data.values, color='green', alpha=0.7)
            axes[0,0].set_title('Success Rate by Algorithm')
            axes[0,0].set_ylabel('Success Rate')
            axes[0,0].tick_params(axis='x', rotation=45)
            axes[0,0].set_ylim(0, 1)
        
        # Tool calls comparison
        if 'total_tool_calls' in df.columns:
            df.boxplot(column='total_tool_calls', by='algorithm', ax=axes[0,1])
            axes[0,1].set_title('Tool Calls Distribution by Algorithm')
            axes[0,1].set_ylabel('Total Tool Calls')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # Cost comparison
        if 'total_cost' in df.columns:
            df.boxplot(column='total_cost', by='algorithm', ax=axes[0,2])
            axes[0,2].set_title('Cost Distribution by Algorithm')
            axes[0,2].set_ylabel('Total Cost ($)')
            axes[0,2].tick_params(axis='x', rotation=45)
        
        # Time comparison
        if 'total_time' in df.columns:
            df.boxplot(column='total_time', by='algorithm', ax=axes[1,0])
            axes[1,0].set_title('Time Distribution by Algorithm')
            axes[1,0].set_ylabel('Total Time (seconds)')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Tree height comparison  
        if 'tree_height' in df.columns:
            df.boxplot(column='tree_height', by='algorithm', ax=axes[1,1])
            axes[1,1].set_title('Search Tree Height by Algorithm')
            axes[1,1].set_ylabel('Tree Height')
            axes[1,1].tick_params(axis='x', rotation=45)
        
        # Cost vs Tool Calls efficiency
        if 'total_cost' in df.columns and 'total_tool_calls' in df.columns:
            for algo in df['algorithm'].unique():
                algo_data = df[df['algorithm'] == algo]
                axes[1,2].scatter(algo_data['total_tool_calls'], algo_data['total_cost'], 
                                label=algo, alpha=0.7, s=60)
            axes[1,2].set_xlabel('Total Tool Calls')
            axes[1,2].set_ylabel('Total Cost ($)')
            axes[1,2].set_title('Cost vs Tool Call Efficiency')
            axes[1,2].legend()
        
        plt.tight_layout()
        plt.savefig('/Users/ychittepu/Documents/Codes/DS-Tools-v2/algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_detailed_report(self):
        """Generate comprehensive comparison report"""
        print("🚀 TRAJECTORY COMPARISON REPORT")
        print("="*50)
        
        # Basic algorithm info
        print(f"\n📂 Algorithms Found: {len(self.algorithms)}")
        for algo, trajs in self.algorithms.items():
            print(f"   • {algo}: {len(trajs)} trajectories")
        
        # Step pattern analysis
        print("\n🔄 Step Pattern Analysis:")
        step_patterns = self.analyze_step_patterns()
        for algo, patterns in step_patterns.items():
            print(f"\n  {algo.upper()}:")
            for step_type, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
                print(f"    - {step_type}: {count}")
        
        # Efficiency comparison
        df = self.compare_efficiency_metrics()
        
        # Tool usage analysis
        self.analyze_tool_usage_patterns()
        
        # Solution path analysis
        self.analyze_solution_paths()
        
        # Generate visualizations
        if df is not None and not df.empty:
            print(f"\n📊 Generating visualizations...")
            self.visualize_comparisons(df)
            print(f"   Saved: algorithm_comparison.png")
        
        return df

def main():
    # Analyze trajectories
    traj_dir = "/Users/ychittepu/Documents/Codes/DS-Tools-v2/trajectories_dump/traj_dump"
    analyzer = TrajectoryAnalyzer(traj_dir)
    
    # Generate comprehensive report
    results_df = analyzer.generate_detailed_report()
    
    # Save detailed results
    if results_df is not None and not results_df.empty:
        results_df.to_csv('/Users/ychittepu/Documents/Codes/DS-Tools-v2/trajectory_comparison_results.csv', index=False)
        print(f"\n💾 Detailed results saved: trajectory_comparison_results.csv")

if __name__ == "__main__":
    main()
