#!/usr/bin/env python3
"""
Evo-MCTS Run Template Script
Based on test_flexible.sh for programmatic execution

This template script provides a programmatic way to run the Evo-MCTS framework
with flexible configuration. It's designed for easy deployment and portability
across different computing environments.

Usage:
    python run_template.py
    
    Or with custom parameters:
    python run_template.py --model gpt-4 --api-key your-key --temperature 0.8
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def load_env_file(env_file=".env"):
    """Load environment variables from .env file if it exists."""
    env_path = Path(env_file)
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"✅ Loaded environment variables from {env_file}")
    else:
        print(f"⚠️  Environment file {env_file} not found")

def get_config_value(key, default=None, args=None):
    """Get configuration value from args, environment, or default."""
    if args and hasattr(args, key.lower()) and getattr(args, key.lower()):
        return getattr(args, key.lower())
    return os.environ.get(key, default)

def print_configuration(config):
    """Print current configuration with masked sensitive information."""
    print("\n" + "="*50)
    print("🚀 Evo-MCTS Configuration")
    print("="*50)
    
    for key, value in config.items():
        if 'api_key' in key.lower() or 'key' in key.lower():
            # Mask API keys for security
            masked_value = value[:10] + "..." if value and len(value) > 10 else "***"
            print(f"  {key}: {masked_value}")
        else:
            print(f"  {key}: {value}")
    
    print("="*50 + "\n")

def run_evo_mcts(config):
    """Run the Evo-MCTS main program with the given configuration."""
    
    # Set environment variables for the subprocess
    env = os.environ.copy()
    for key, value in config.items():
        if value is not None:
            env[key.upper()] = str(value)
    
    # Construct the command
    cmd = [
        "python", "main.py",
        "problem=gw_mlgwsc1",
        "debug_mode=False",
        "use_seed=True", 
        "timeout=2400",
        "+label=TEST",
        f"llm_client.model={config['model']}",
        f"llm_client.temperature={config['temperature']}",
        f"llm_client.api_key={config['api_key']}",
        f"llm_client.base_url={config['base_url']}"
    ]
    
    print("🔄 Starting Evo-MCTS execution...")
    print(f"Command: {' '.join(cmd[:6])} [... with API configuration]")
    
    try:
        # Run the command
        result = subprocess.run(cmd, env=env, check=True, 
                              capture_output=False, text=True)
        print("\n✅ Evo-MCTS execution completed successfully!")
        return result.returncode
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Evo-MCTS execution failed with return code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

def main():
    """Main function with argument parsing and execution logic."""
    
    parser = argparse.ArgumentParser(
        description="Evo-MCTS Run Template - Programmatic execution with flexible configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_template.py
  python run_template.py --model gpt-4 --api-key sk-your-key
  python run_template.py --temperature 0.5 --cpu-usage 75
  
Environment Variables:
  MODEL, API_KEY, BASE_URL, TEMPERATURE, ML_CHALLENGE_PATH, 
  DATA_DIR, NUMEXPR_MAX_THREADS, CPU_USAGE_PERCENT,
  DEEPSEEK_MODEL, DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL
        """
    )
    
    # LLM Configuration
    parser.add_argument("--model", type=str, help="LLM model to use")
    parser.add_argument("--api-key", type=str, help="API key for LLM service")
    parser.add_argument("--base-url", type=str, help="Base URL for LLM API")
    parser.add_argument("--temperature", type=float, help="Generation temperature")
    
    # Path Configuration
    parser.add_argument("--ml-challenge-path", type=str, help="ML challenge data path")
    parser.add_argument("--data-dir", type=str, help="Data directory path")
    
    # Performance Configuration
    parser.add_argument("--numexpr-threads", type=int, help="Number of NumExpr threads")
    parser.add_argument("--cpu-usage", type=int, help="CPU usage percentage")
    
    # DeepSeek Configuration
    parser.add_argument("--deepseek-model", type=str, help="DeepSeek model name")
    parser.add_argument("--deepseek-api-key", type=str, help="DeepSeek API key")
    parser.add_argument("--deepseek-base-url", type=str, help="DeepSeek base URL")
    
    # Utility options
    parser.add_argument("--env-file", type=str, default=".env", 
                       help="Environment file to load (default: .env)")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show configuration without running")
    
    args = parser.parse_args()
    
    # Load environment file
    load_env_file(args.env_file)
    
    # Build configuration with precedence: args > env > defaults
    config = {
        'model': get_config_value('MODEL', 'gpt-3.5-turbo', args),
        'api_key': get_config_value('API_KEY', None, args),
        'base_url': get_config_value('BASE_URL', 'https://api.openai.com/v1', args),
        'temperature': get_config_value('TEMPERATURE', '1.0', args),
        'ml_challenge_path': get_config_value('ML_CHALLENGE_PATH', '/tmp', args),
        'data_dir': get_config_value('DATA_DIR', '/tmp', args),
        'numexpr_max_threads': get_config_value('NUMEXPR_MAX_THREADS', '96', args),
        'cpu_usage_percent': get_config_value('CPU_USAGE_PERCENT', '50', args),
        'deepseek_model': get_config_value('DEEPSEEK_MODEL', 'deepseek-chat', args),
        'deepseek_api_key': get_config_value('DEEPSEEK_API_KEY', None, args),
        'deepseek_base_url': get_config_value('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1', args),
    }
    
    # Validate required configuration
    if not config['api_key']:
        print("❌ Error: API key is required but not provided.")
        print("   Set API_KEY environment variable or use --api-key argument.")
        sys.exit(1)
    
    # Print configuration
    print_configuration(config)
    
    # Dry run mode
    if args.dry_run:
        print("🔍 Dry run mode - configuration validated, exiting without execution.")
        sys.exit(0)
    
    # Check if main.py exists
    if not Path("main.py").exists():
        print("❌ Error: main.py not found. Please run this script from the Evo-MCTS root directory.")
        sys.exit(1)
    
    # Run Evo-MCTS
    exit_code = run_evo_mcts(config)
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
