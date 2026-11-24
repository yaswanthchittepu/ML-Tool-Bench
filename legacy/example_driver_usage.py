#!/usr/bin/env python3
"""
Example Usage Patterns for the Parallel Agent Driver Script

This file demonstrates common usage patterns for run_all_parallel_agents.py.
You can copy these commands and modify them for your needs.
"""

import subprocess
import os

def run_command(description, command):
    """Run a command with description."""
    print(f"\n{'='*60}")
    print(f"EXAMPLE: {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print("\nTo run this example, execute:")
    print(f"python {command}")
    print("")

def main():
    """Show example usage patterns."""
    
    print("PARALLEL AGENT DRIVER - EXAMPLE USAGE PATTERNS")
    print("=" * 60)
    
    # Basic examples
    run_command(
        "Run all scripts with default settings",
        "run_all_parallel_agents.py --competition spaceship_titanic"
    )
    
    run_command(
        "Run with custom trajectory and worker counts",
        "run_all_parallel_agents.py --competition customer_churn_prediction --trajectories 16 --max_workers 8"
    )
    
    # Script selection examples
    run_command(
        "Run only ReAct agents",
        "run_all_parallel_agents.py --competition bank_marketing_campaign --scripts react hierarchical_react"
    )
    
    run_command(
        "Run only LATS scripts",
        "run_all_parallel_agents.py --competition spaceship_titanic --scripts lats_reflection lats_rewards lats_hierarchical"
    )
    
    run_command(
        "Run specific LATS scripts with feedback",
        "run_all_parallel_agents.py -c customer_churn_prediction --scripts lats_rewards lats_hierarchical --use_feedback"
    )
    
    # Advanced configuration examples
    run_command(
        "High-performance run with outcome rewards",
        "run_all_parallel_agents.py --competition bank_marketing_campaign -t 32 -w 16 --outcome_reward --verbose"
    )
    
    run_command(
        "Quick test run with reduced parameters",
        "run_all_parallel_agents.py --competition spaceship_titanic -t 4 -w 2 --depth 10 --quiet"
    )
    
    run_command(
        "Comprehensive run with all features",
        "run_all_parallel_agents.py -c customer_churn_prediction -t 24 -w 12 --depth 25 --use_feedback --outcome_reward --verbose"
    )
    
    # Error handling examples  
    run_command(
        "Stop on first failure",
        "run_all_parallel_agents.py --competition spaceship_titanic --stop_on_failure"
    )
    
    run_command(
        "Show real-time output from scripts",
        "run_all_parallel_agents.py --competition bank_marketing_campaign --show_output"
    )
    
    run_command(
        "Custom timeout",
        "run_all_parallel_agents.py --competition spaceship_titanic --timeout 7200"
    )
    
    # Competition-specific examples
    print("\n" + "="*60)
    print("COMPETITION-SPECIFIC EXAMPLES")
    print("="*60)
    
    competitions = [
        "spaceship_titanic",
        "customer_churn_prediction", 
        "bank_marketing_campaign"
    ]
    
    for comp in competitions:
        run_command(
            f"Optimized run for {comp}",
            f"run_all_parallel_agents.py --competition {comp} --trajectories 16 --max_workers 8 --use_feedback --verbose"
        )
    
    # Batch processing examples
    print("\n" + "="*60)
    print("BATCH PROCESSING IDEAS")
    print("="*60)
    
    batch_examples = [
        "# Run all competitions sequentially",
        "for comp in spaceship_titanic customer_churn_prediction bank_marketing_campaign; do",
        "    python run_all_parallel_agents.py --competition $comp --trajectories 12 --max_workers 6",
        "    sleep 30  # Wait between competitions",
        "done",
        "",
        "# Compare feedback vs no-feedback",
        "python run_all_parallel_agents.py --competition spaceship_titanic --use_feedback",
        "python run_all_parallel_agents.py --competition spaceship_titanic --no_feedback",
        "",
        "# Compare reward systems",
        "python run_all_parallel_agents.py --competition customer_churn_prediction --scripts lats_rewards --outcome_reward",
        "python run_all_parallel_agents.py --competition customer_churn_prediction --scripts lats_rewards --stage_reward",
    ]
    
    for line in batch_examples:
        print(line)
    
    print("\n" + "="*60)
    print("HELPFUL TIPS")
    print("="*60)
    
    tips = [
        "1. Start with smaller trajectory counts (4-8) to test your setup",
        "2. Use --quiet for production runs to reduce log noise", 
        "3. Use --show_output for debugging script issues",
        "4. Set appropriate --timeout based on your competition complexity",
        "5. Use --stop_on_failure for critical runs where you need all scripts to succeed",
        "6. Monitor system resources when using high --max_workers values",
        "7. The driver creates separate result directories for each script type",
        "8. Check the summary output to identify which scripts succeeded/failed",
        "9. You can interrupt execution with Ctrl+C and still get a summary",
        "10. Use --scripts to focus on specific agent types for comparison studies"
    ]
    
    for tip in tips:
        print(f"  {tip}")
    
    print(f"\n{'='*60}")
    print("For help with all options, run:")
    print("python run_all_parallel_agents.py --help")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
