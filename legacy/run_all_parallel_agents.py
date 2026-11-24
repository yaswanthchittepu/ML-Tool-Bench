#!/usr/bin/env python3
"""
Driver Script for Running All Parallelized Agent Scripts

This script automatically runs all parallelized agent scripts sequentially with 
configurable parameters, eliminating the need for manual execution.

Available Scripts:
1. react_ds_agent_parallel.py - Basic ReAct Data Science Agent
2. hierarchical_react_ds_agent_parallel.py - Hierarchical ReAct Agent
3. LATS/lats_ds_reflection_parallel.py - LATS with Reflection
4. LATS/lats_ds_verifiable_rewards_parallel.py - LATS with Verifiable Rewards
5. LATS/lats_hierarchical_parallel.py - Hierarchical LATS

Usage Examples:
    # Run all scripts with default settings
    python run_all_parallel_agents.py --competition spaceship_titanic
    
    # Run specific scripts only
    python run_all_parallel_agents.py --competition customer_churn_prediction --scripts react hierarchical_react
    
    # Run with custom parameters
    python run_all_parallel_agents.py -c bank_marketing_campaign -t 16 -w 8 --use_feedback --verbose
    
    # Run LATS scripts only
    python run_all_parallel_agents.py --competition spaceship_titanic --scripts lats_reflection lats_rewards lats_hierarchical
"""

import subprocess
import sys
import os
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

class ParallelAgentRunner:
    """Driver for running multiple parallelized agent scripts sequentially."""
    
    def __init__(self):
        self.scripts = {
            'react': {
                'path': 'react_ds_agent_parallel.py',
                'name': 'ReAct Data Science Agent',
                'supports_feedback': False,
                'supports_outcome_rewards': False,
                'supports_rollout': False
            },
            'hierarchical_react': {
                'path': 'hierarchical_react_ds_agent_parallel.py', 
                'name': 'Hierarchical ReAct Agent',
                'supports_feedback': False,
                'supports_outcome_rewards': False,
                'supports_rollout': False
            },
            'lats_reflection': {
                'path': 'LATS/lats_ds_reflection_parallel.py',
                'name': 'LATS with Reflection',
                'supports_feedback': True,
                'supports_outcome_rewards': False,
                'supports_rollout': False
            },
            'lats_outcome_rewards': {
                'path': 'LATS/lats_ds_verifiable_rewards_rollout_parallel.py',
                'name': 'LATS with Outcome Rewards',
                'supports_feedback': True,
                'supports_outcome_rewards': True,
                'supports_rollout': True
            },
            'lats_shaped_rewards': {
                'path': 'LATS/lats_ds_verifiable_rewards_rollout_parallel.py',
                'name': 'LATS with Shaped Rewards',
                'supports_feedback': True,
                'supports_outcome_rewards': True,
                'supports_rollout': True
            },
            'lats_hierarchical': {
                'path': 'LATS/lats_hierarchical_rollout_parallel.py',
                'name': 'Hierarchical LATS',
                'supports_feedback': True,
                'supports_outcome_rewards': False,
                'supports_rollout': True
            }
        }
        
        self.results = []
        
    def build_command(self, script_key: str, args: argparse.Namespace) -> List[str]:
        """Build command arguments for a specific script."""
        script_info = self.scripts[script_key]
        cmd = ['python', script_info['path']]
        
        # Common arguments
        if args.competition:
            cmd.extend(['--competition', args.competition])
        if args.trajectories:
            cmd.extend(['--trajectories', str(args.trajectories)])
        if args.max_workers:
            cmd.extend(['--max_workers', str(args.max_workers)])
            
        # Feedback arguments (only for scripts that support it)
        if script_info['supports_feedback']:
            if args.use_feedback:
                cmd.append('--use_feedback')
            elif args.no_feedback:
                cmd.append('--no_feedback')
        
        # Reward arguments (only for verifiable rewards script)
        if script_info['supports_outcome_rewards']:
            if args.outcome_reward:
                cmd.append('--outcome_reward')
            elif args.stage_reward:
                cmd.append('--stage_reward')
                
        # Depth argument (for LATS scripts)
        if 'lats' in script_key and args.depth:
            cmd.extend(['--depth', str(args.depth)])
            
        # Verbose/quiet flags
        if args.verbose:
            cmd.append('--verbose')
        elif args.quiet:
            cmd.append('--quiet')
            
        return cmd
    
    def run_script(self, script_key: str, args: argparse.Namespace) -> Dict[str, Any]:
        """Run a single script and return results."""
        script_info = self.scripts[script_key]
        
        print(f"\n{'='*80}")
        print(f"RUNNING: {script_info['name']}")
        print(f"Script: {script_info['path']}")
        print(f"{'='*80}")
        
        if not Path(script_info['path']).exists():
            error_msg = f"Script not found: {script_info['path']}"
            print(f"❌ ERROR: {error_msg}")
            return {
                'script': script_key,
                'name': script_info['name'],
                'success': False,
                'error': error_msg,
                'execution_time': 0,
                'command': None
            }
        
        cmd = self.build_command(script_key, args)
        print(f"Command: {' '.join(cmd)}")
        print("")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=os.getcwd(),
                capture_output=not args.show_output,
                text=True,
                timeout= None
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            success = result.returncode == 0
            
            if success:
                print(f"✅ {script_info['name']} completed successfully in {execution_time:.2f}s")
            else:
                print(f"❌ {script_info['name']} failed with return code {result.returncode}")
                if not args.show_output and result.stderr:
                    print(f"Error output: {result.stderr[:500]}...")
            
            return {
                'script': script_key,
                'name': script_info['name'],
                'success': success,
                'error': result.stderr if not success else None,
                'execution_time': execution_time,
                'command': ' '.join(cmd),
                'return_code': result.returncode,
                'stdout': result.stdout if not args.show_output else None
            }
            
        except subprocess.TimeoutExpired:
            end_time = time.time()
            error_msg = f"Script timed out after {args.timeout}s"
            print(f"⏰ {script_info['name']} timed out")
            return {
                'script': script_key,
                'name': script_info['name'],
                'success': False,
                'error': error_msg,
                'execution_time': end_time - start_time,
                'command': ' '.join(cmd)
            }
            
        except Exception as e:
            end_time = time.time()
            error_msg = f"Unexpected error: {str(e)}"
            print(f"💥 {script_info['name']} failed with exception: {e}")
            return {
                'script': script_key,
                'name': script_info['name'],
                'success': False,
                'error': error_msg,
                'execution_time': end_time - start_time,
                'command': ' '.join(cmd)
            }
    
    def run_all(self, script_keys: List[str], args: argparse.Namespace) -> None:
        """Run all specified scripts sequentially."""
        total_start_time = time.time()
        
        print(f"🚀 Starting execution of {len(script_keys)} parallelized agent scripts")
        print(f"Scripts to run: {', '.join([self.scripts[key]['name'] for key in script_keys])}")
        
        for i, script_key in enumerate(script_keys, 1):
            print(f"\n📍 Progress: {i}/{len(script_keys)}")
            
            result = self.run_script(script_key, args)
            self.results.append(result)
            
            if not result['success'] and args.stop_on_failure:
                print(f"\n🛑 Stopping execution due to failure in {result['name']}")
                break
            else:
                print(f"\n\n\n\n\n\n############################### {script_key} completed successfully ###############################\n\n\n\n\n\n")
        
        total_end_time = time.time()
        total_execution_time = total_end_time - total_start_time
        
        self.print_summary(total_execution_time)
    
    def print_summary(self, total_time: float) -> None:
        """Print execution summary."""
        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]
        
        print(f"\n{'='*80}")
        print("EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total scripts run: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        print(f"Total execution time: {total_time:.2f}s")
        
        if successful:
            avg_time = sum(r['execution_time'] for r in successful) / len(successful)
            print(f"Average successful execution time: {avg_time:.2f}s")
            
            print(f"\n✅ Successful Scripts:")
            for result in successful:
                print(f"  • {result['name']}: {result['execution_time']:.2f}s")
        
        if failed:
            print(f"\n❌ Failed Scripts:")
            for result in failed:
                print(f"  • {result['name']}: {result['error']}")
                print(f"    Command: {result['command']}")
        
        print(f"\n{'='*80}")
        
        if len(successful) == len(self.results):
            print("🎉 All scripts completed successfully!")
        elif len(successful) > 0:
            print(f"⚠️  {len(successful)}/{len(self.results)} scripts completed successfully")
        else:
            print("💥 All scripts failed")
        
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(
        description='Run all parallelized agent scripts sequentially',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --competition spaceship_titanic
  %(prog)s -c customer_churn_prediction --scripts react hierarchical_react
  %(prog)s --competition bank_marketing_campaign -t 16 -w 8 --use_feedback
  %(prog)s -c spaceship_titanic --scripts lats_reflection lats_rewards --verbose
        """
    )
    
    # Required arguments
    parser.add_argument('--competition', '-c', type=str, required=True,
                       help='Competition name to run (required)')
    
    # Script selection
    parser.add_argument('--scripts', nargs='+', 
                       choices=['react', 'hierarchical_react', 'lats_reflection', 'lats_rewards', 'lats_hierarchical', 'all'],
                       default=['all'],
                       help='Which scripts to run (default: all)')
    
    # Common execution parameters
    parser.add_argument('--trajectories', '-t', type=int,
                       help='Number of parallel trajectories for each script')
    parser.add_argument('--max_workers', '-w', type=int,
                       help='Maximum number of worker processes')
    parser.add_argument('--depth', '-d', type=int,
                       help='Maximum depth for LATS scripts')
    
    # Feedback options
    parser.add_argument('--use_feedback', action='store_true',
                       help='Enable feedback for supported scripts')
    parser.add_argument('--no_feedback', action='store_true',
                       help='Disable feedback for supported scripts')
    
    # Reward options (for verifiable rewards script)
    parser.add_argument('--outcome_reward', action='store_true',
                       help='Use outcome-based rewards (lats_rewards only)')
    parser.add_argument('--stage_reward', action='store_true',
                       help='Use stage-based rewards (lats_rewards only)')
    
    # Output control
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output for all scripts')
    parser.add_argument('--quiet', action='store_true',
                       help='Disable verbose output for all scripts')
    parser.add_argument('--show_output', action='store_true',
                       help='Show real-time output from scripts (instead of capturing)')
    
    # Execution control
    parser.add_argument('--stop_on_failure', action='store_true',
                       help='Stop execution if any script fails')

    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_feedback and args.no_feedback:
        parser.error("Cannot specify both --use_feedback and --no_feedback")
    
    if args.outcome_reward and args.stage_reward:
        parser.error("Cannot specify both --outcome_reward and --stage_reward")
    
    # Determine which scripts to run
    runner = ParallelAgentRunner()
    
    if 'all' in args.scripts:
        script_keys = list(runner.scripts.keys())
    else:
        script_keys = args.scripts
    
    # Show configuration
    print(f"Driver Script Configuration:")
    print(f"  Competition: {args.competition}")
    print(f"  Scripts: {', '.join([runner.scripts[key]['name'] for key in script_keys])}")
    print(f"  Trajectories: {args.trajectories or 'default'}")
    print(f"  Max Workers: {args.max_workers or 'default'}")
    if args.depth:
        print(f"  Depth: {args.depth}")
    if args.use_feedback:
        print(f"  Feedback: enabled")
    elif args.no_feedback:
        print(f"  Feedback: disabled")
    if args.outcome_reward:
        print(f"  Rewards: outcome-based")
    elif args.stage_reward:
        print(f"  Rewards: stage-based")
    print(f"  Stop on failure: {args.stop_on_failure}")
    print("")
    
    # Run all scripts
    try:
        runner.run_all(script_keys, args)
    except KeyboardInterrupt:
        print(f"\n🛑 Execution interrupted by user")
        runner.print_summary(0)
        sys.exit(1)


if __name__ == "__main__":
    main()
