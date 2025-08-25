import os
import sys
import time
import subprocess
import logging
import h5py
import numpy as np
import glob
import shutil
import concurrent.futures
import multiprocessing
import traceback
import atexit
import signal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from gen_inst import get_injection_ids, SET_NUMBER, DURATION, RANDOM_SEED, START_TIME, DATA_DIR, FAR_MIN, FAR_MAX
from plot_auc import calculate_auc

# Configure NUMEXPR_MAX_THREADS from environment variable or default
NUMEXPR_MAX_THREADS = os.getenv('NUMEXPR_MAX_THREADS', '96')
os.environ["NUMEXPR_MAX_THREADS"] = NUMEXPR_MAX_THREADS
logging.info(f"Using NUMEXPR_MAX_THREADS: {NUMEXPR_MAX_THREADS}")

# Define max_workers as a global variable using percentage of available CPUs
CPU_USAGE_PERCENT = int(os.getenv('CPU_USAGE_PERCENT', '50'))  # Use environment variable or default 50%
logging.info(f"Using CPU_USAGE_PERCENT: {CPU_USAGE_PERCENT}%")
MAX_WORKERS = max(1, int(multiprocessing.cpu_count() * CPU_USAGE_PERCENT / 100))

# Define the results directory path - will be set in main based on runid
RESULTS_DIR = None

def clean_results_directory(RESULTS_DIR):
    """Clean up the results directory completely"""
    try:
        if RESULTS_DIR and os.path.exists(RESULTS_DIR):
            logging.info(f"Cleaning up results directory: {RESULTS_DIR}")
            # Remove the entire directory and its contents
            shutil.rmtree(RESULTS_DIR)
            logging.info("Results directory cleanup completed")
    except Exception as e:
        logging.error(f"Error during results directory cleanup: {str(e)}")

# Register the cleanup function to run at exit
atexit.register(clean_results_directory, RESULTS_DIR)

# Define a function to process a single injection
def process_injection(args):
    ix, inj_id, total_injections, RESULTS_DIR, eval_script_path, gpt_runid = args
    try:
        logging.info(f"Processing injection {ix+1}/{total_injections} (ID: {inj_id})")
        
        # Execute the eval_inj.py script with the current injection ID
        cmd = [sys.executable, eval_script_path, str(ix), str(inj_id), RESULTS_DIR, gpt_runid]
        
        # Run the subprocess
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Check if the execution was successful
        if result.returncode == 0:
            logging.info(f"Successfully processed injection {inj_id} ({ix+1}/{total_injections})")
            return True
        else:
            error_msg = f"Error processing injection {inj_id} ({ix+1}/{total_injections}): {result.stderr}"
            logging.error(error_msg)
            # Kill the entire process to terminate all parallel execution
            logging.critical("Critical error encountered. Terminating all processes.")
            os.kill(os.getpid(), signal.SIGTERM)
            sys.exit(1)  # This line won't execute after os.kill, but added as a fallback

    except subprocess.TimeoutExpired:
        logging.error(f"Timeout processing injection {inj_id} ({ix+1}/{total_injections}) after 600 seconds")
        return False
    except Exception as e:
        logging.error(f"Failed to process injection {inj_id} ({ix+1}/{total_injections}): {str(e)}")
        return False

def aggregate_results(direction, RESULTS_DIR):
    """
    Aggregate all individual injection HDF5 files into a single file for evaluation.
    
    Args:
        direction: 'f' for foreground or 'b' for background
    """
    logging.info(f"Aggregating {direction} results...")
    
    # Find all injection files for the given direction
    pattern = os.path.join(RESULTS_DIR, f"inj_*_{direction}.hdf5")
    inj_files = glob.glob(pattern)
    
    if not inj_files:
        logging.error(f"No {direction} injection files found matching pattern: {pattern}")
        return None
    
    # Initialize lists to store data
    all_peak_times = []
    all_peak_heights = []
    all_peak_deltat = []
    
    # Read data from each file
    total_files = len(inj_files)
    for i, file_path in enumerate(inj_files):
        try:
            logging.info(f"Reading {direction} file {i+1}/{total_files}: {file_path}")
            with h5py.File(file_path, 'r') as f:
                all_peak_times.append(f['peak_times'][:])
                all_peak_heights.append(f['peak_heights'][:])
                all_peak_deltat.append(f['peak_deltat'][:])
        except Exception as e:
            logging.error(f"Error reading {file_path}: {str(e)}")
    
    # Concatenate all arrays
    if all_peak_times:
        peak_times = np.concatenate(all_peak_times)
        peak_heights = np.concatenate(all_peak_heights)
        peak_deltat = np.concatenate(all_peak_deltat)
        
        # Create output file
        output_file = os.path.join(RESULTS_DIR, f"aggregated_{direction}.hdf5")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('time', data=peak_times)
            f.create_dataset('stat', data=peak_heights)
            f.create_dataset('var', data=peak_deltat)
        
        logging.info(f"Aggregated {direction} data saved to {output_file}")
        return output_file
    else:
        logging.error(f"No valid data found for {direction} direction")
        return None

def run_evaluation(fg_file, bg_file, eval_script_path, RESULTS_DIR):
    """Run the evaluation script with the aggregated files"""
    # Construct data file paths
    file_template = f'set{SET_NUMBER}_{DURATION}dur_seed{RANDOM_SEED}_start{START_TIME}.hdf'
    foreground_path = os.path.join(DATA_DIR, f'foreground_{file_template}')
    injection_path = os.path.join(DATA_DIR, f'injections_{file_template}')
    output_path = os.path.join(RESULTS_DIR, 'sensitivity_results.hdf')
    
    
    # Run the evaluation script
    cmd = [
        eval_script_path,
        foreground_path,
        fg_file,
        bg_file,
        injection_path,
        output_path
    ]
    
    logging.info(f"Running evaluation with command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logging.info(f"Evaluation completed successfully. Results saved to {output_path}")
            return output_path
        else:
            logging.error(f"Evaluation failed: {result.stderr}")
            return None
    except Exception as e:
        logging.error(f"Error running evaluation: {str(e)}")
        return None

def process_sensitivity_results(results_file):
    """Process the sensitivity results from the HDF file"""
    try:
        if not os.path.exists(results_file):
            logging.error(f"Results file not found: {results_file}")
            return None
            
        logging.info(f"Processing sensitivity results from {results_file}")
        results = {}
        
        with h5py.File(results_file, 'r') as f:
            # Extract all available datasets
            for key in f.keys():
                results[key] = f[key][()]
                
        return results
    except Exception as e:
        logging.error(f"Error processing sensitivity results: {str(e)}")
        traceback.print_exc()
        return None

def cleanup_files(RESULTS_DIR):
    """Clean up intermediate HDF5 files"""
    pattern = os.path.join(RESULTS_DIR, "inj_*_*.hdf5")
    files = glob.glob(pattern)
    
    if files:
        logging.info(f"Cleaning up {len(files)} intermediate files...")
        for file_path in files:
            try:
                os.remove(file_path)
            except Exception as e:
                logging.warning(f"Failed to remove {file_path}: {str(e)}")
    else:
        logging.info("No intermediate files to clean up")

# Create a function to plot sensitivity vs FAR
def plot_sensitivity_vs_far(sensitivity_results, output_dir=None, far_min=None, far_max=None):
    """
    Plot sensitivity vs FAR and save the figure to the specified directory.
    
    Args:
        sensitivity_results: Dictionary containing sensitivity analysis results
        output_dir: Directory to save the plot (default: current directory)
        far_min: Minimum FAR value to display
        far_max: Maximum FAR value to display
    """
    try:
        import matplotlib.pyplot as plt
        import os
        import numpy as np
        import h5py
        
        # Extract data from sensitivity results
        fars = sensitivity_results['far']
        sensitivities = sensitivity_results['sensitive-distance']
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot our algorithm results
        far_scaling_factor = 31557600/12  # Same scaling factor used in calculate_auc
        plt.plot(fars * far_scaling_factor, sensitivities, 'b-o', linewidth=2, 
                    label=f'Our Algorithm (AUC={sensitivity_results["auc"]:.1f})')
        
        # Load PyCBC and cWB results for comparison
        benchmark_results = {}
        #SET_NUMBER = 1  # Assuming set 1 based on file paths in context
        
        for res_name in ['PyCBC', 'cWB']:
            try:
                import socket
                hostname = socket.gethostname()
                if hostname == 'login05': # Dongfang
                    result_path = f'/work1/hewang/HW/ml-mock-data-challenge-1/results/{res_name}/ds{SET_NUMBER}/eval.hdf'
                elif hostname == '1f7472014c87': # A800
                    result_path = f'/home/nvme0n1/data/ml-mock-data-challenge-1/results/{res_name}/ds{SET_NUMBER}/eval.hdf'
                elif hostname == 'Gravitation-wave': # A6000
                    result_path = f'/home/main/gwtoolkit_project/gwtoolkit/benchmark/ml-mock-data-challenge-1/results/{res_name}/ds{SET_NUMBER}/eval.hdf'
                else:
                    logging.error(f"Unknown hostname: {hostname}")
                    exit(1)
                with h5py.File(result_path, 'r') as f:
                    benchmark_results[res_name] = {
                        'far': f['far'][:],
                        'sensitive-distance': f['sensitive-distance'][:]
                    }
                    
                    # Calculate AUC for the benchmark
                    far_values = benchmark_results[res_name]['far']
                    sens_values = benchmark_results[res_name]['sensitive-distance']
                    
                    # Filter values within the specified FAR range
                    if far_min is not None and far_max is not None:
                        mask = (far_values >= far_min/far_scaling_factor) & (far_values <= far_max/far_scaling_factor)
                        far_values = far_values[mask]
                        sens_values = sens_values[mask]
                    
                    # Calculate AUC using trapezoidal rule
                    if len(far_values) > 1:
                        log_far_values = np.log10(far_values)
                        auc = np.trapz(sens_values, log_far_values)
                        benchmark_results[res_name]['auc'] = auc
                    else:
                        benchmark_results[res_name]['auc'] = 0
                        
                    # Plot the benchmark results
                    plt.plot(benchmark_results[res_name]['far'] * far_scaling_factor,
                                benchmark_results[res_name]['sensitive-distance'],
                                '--' if res_name == 'PyCBC' else '-.',
                                linewidth=1.5,
                                label=f'{res_name} (AUC={-benchmark_results[res_name]["auc"]:.1f})')
            except Exception as e:
                logging.warning(f"Could not load {res_name} results: {str(e)}")
        
        # Add vertical lines for far_min and far_max if provided
        if far_min is not None:
            plt.axvline(x=far_min, color='k', linestyle=':', alpha=0.7, 
                        label=f'FAR_MIN = {far_min:.2e} Hz')
        
        if far_max is not None:
            plt.axvline(x=far_max, color='k', linestyle='--', alpha=0.7,
                        label=f'FAR_MAX = {far_max:.2e} Hz')
        
        plt.xscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel('False Alarm Rate (Hz)', fontsize=12)
        plt.ylabel('Sensitive Distance (Mpc)', fontsize=12)
        plt.title('Sensitivity vs False Alarm Rate', fontsize=14)
        plt.legend(loc='best')
        
        # Save the figure
        if output_dir is None:
            output_dir = '.'
        
        output_path = os.path.join(output_dir, f'{runid}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Sensitivity vs FAR plot saved to {output_path}")
        plt.close()
        
        np.save(os.path.join(output_dir, f'{runid}_png.npy'), sensitivity_results)
        return output_path
    except Exception as e:
        logging.error(f"Error creating sensitivity vs FAR plot: {str(e)}")
        return None
    
def alpha_decay(depth, max_depth, min_alpha=0.2, type='linear'):
    import numpy as np

    # 确保alpha随着depth增加而严格单调递减
    # 从1.0开始递减到一个最小值(这里设为0.2)
    min_alpha = 0.2

    if type == 'linear':
        # 线性递减方案
        alpha_linear = 1.0 - (depth - 1) / (max_depth - 1) * (1.0 - min_alpha)
        return alpha_linear

    elif type == 'exponential':
        # 指数递减方案 - 保证从1.0递减到min_alpha
        decay_rate = -np.log(min_alpha) / (max_depth - 1)
        alpha_exp = np.exp(-decay_rate * (depth - 1))
        return alpha_exp  

    elif type == 'logarithmic':
        # 对数递减方案 - 保证从1.0递减到min_alpha
        # 使用对数函数的特性实现递减速率逐渐变慢
        log_factor = np.log(1 + (max_depth - 1)) / (1 - min_alpha)
        alpha_log = 1.0 - np.log(1 + (depth - 1)) / log_factor
        return alpha_log

if __name__ == "__main__":

    problem_size = (sys.argv[1]) # FIXME: window_size
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    stdout_dir = sys.argv[4]
    runid = sys.argv[5]
    log_info = sys.argv[6]
    gpt_runid = sys.argv[7]

    # Set the results directory with runid
    RESULTS_DIR = os.path.join(DATA_DIR, f'res_{runid}')
    logging.info(f'{log_info}')

    # # Parse iteration, depth and max_depth from log_info
    # import re
    # # Extract iteration, depth and max_depth using regex
    # if log_info:
    #     pattern = r'Iteration: (\d+), Depth: (\d+)/(\d+)'
    #     match = re.search(pattern, log_info)
    #     if match:
    #         iteration = int(match.group(1))
    #         depth = int(match.group(2))
    #         max_depth = int(match.group(3))
    # alpha = alpha_decay(depth, max_depth, type='linear')

    try:
        # Create results directory if it doesn't exist
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        injection_ids, durations, detectors = get_injection_ids()

        # # Randomly sample alpha percentage of injection_ids
        # num_samples = max(1, int(len(injection_ids) * alpha))
        # logging.info(f"Sampling {num_samples} injections ({alpha:.2%} of total {len(injection_ids)})")
        
        # Use random.sample to get a random subset without replacement
        import random
        # random.seed(iteration)  # For reproducibility
        # Identify and exclude the problematic injection ID 1238645908
        problematic_ids = ['1238645908'] # ds4 1 week
        # problematic_ids = ['1241758890'] # ds4 1 month
        # # ['1242692672', '1244106430']
        injection_ids = [inj_id for inj_id in injection_ids if inj_id not in problematic_ids]
        logging.info(f"Excluded problematic injection IDs {problematic_ids}")
        # print(injection_ids)
        # Exclude the fifth element from injection_ids
        # if len(injection_ids) >= 5: # DEBUG: injection_ids[5] 总是跑不出来而timeout
            # injection_ids = injection_ids[:4] + injection_ids[5:]
        # injection_ids = random.sample(injection_ids, len(injection_ids)-1) # 无放回抽样
        # injection_ids = injection_ids[4:6] # FIXME: remove this
        # injection_ids = injection_ids[:4] + injection_ids[5:]

        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        eval_script_path = os.path.join(current_dir, 'eval_inj.py')
        
        # Check if the script exists
        if not os.path.exists(eval_script_path):
            logging.error(f"Evaluation script not found at: {eval_script_path}")
            sys.exit(1)
        
        logging.info(f"Found {len(injection_ids)} injections to process")
        
        # Use ProcessPoolExecutor for parallel processing
        logging.info(f"Starting parallel processing with {MAX_WORKERS} workers ({CPU_USAGE_PERCENT}% of CPUs)")
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Create a list of (ix, inj_id) tuples for all injections
            injection_args = [(ix, inj_id, len(injection_ids), RESULTS_DIR, eval_script_path, gpt_runid) for ix, inj_id in enumerate(injection_ids)]
            
            # Process injections in parallel and show progress
            total = len(injection_args)
            completed = 0
            futures = []
            
            # Dictionary to track the status of each injection
            injection_status = {inj_id: "pending" for ix, inj_id, _, _, _, _ in injection_args}
            # Map futures to their corresponding injection IDs
            future_to_inj_id = {}
            
            # Track start time for each injection
            injection_start_times = {}
            
            for args in injection_args:
                future = executor.submit(process_injection, args)
                futures.append(future)
                inj_id = args[1]
                future_to_inj_id[future] = inj_id  # Store the injection ID
                injection_start_times[inj_id] = time.time()  # Record start time
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    completed += 1
                    inj_id = future_to_inj_id[future]
                    injection_status[inj_id] = "completed"
                    
                    # Calculate processing time for this injection
                    processing_time = time.time() - injection_start_times[inj_id]
                    
                    logging.info(f"Successfully processed injection {inj_id} (duration: {durations[inj_id]} seconds) in {processing_time:.2f} seconds")
                    logging.info(f"Completed {completed}/{total} injections")
                    
                    # Log the remaining pending injections
                    pending_injections = [inj_id for inj_id, status in injection_status.items() if status == "pending"]
                    logging.info(f"Remaining pending injections: {pending_injections}")
                    
                    future.result()  # Get the result to propagate any exceptions
                except Exception as e:
                    logging.error(f"Exception in worker: {str(e)}")
                    # Continue processing other injections instead of stopping

        # Aggregate results for foreground and background
        fg_file = aggregate_results('f', RESULTS_DIR)
        bg_file = aggregate_results('b', RESULTS_DIR)
        
        # Get evaluation script path
        eval_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval.sh')

        # Run evaluation if both files were created successfully
        if fg_file and bg_file:
            results_file = run_evaluation(fg_file, bg_file, eval_script_path, RESULTS_DIR)

            if results_file:
                # Process sensitivity results
                sensitivity_results = process_sensitivity_results(results_file)
                
                if sensitivity_results:
                    # Calculate AUC and plot benchmark results using imported functions
                    calculate_auc(sensitivity_results, metric_name='sensitive-distance', 
                                 far_scaling_factor=31557600/12, far_min=FAR_MIN, far_max=FAR_MAX)

                    logging.info("Results processing and visualization completed successfully")
                else:
                    logging.error("Failed to process sensitivity results")
                
                # Clean up intermediate files after successful evaluation
                cleanup_files(RESULTS_DIR)

                # Plot sensitivity vs FAR if sensitivity_results exists
                if 'sensitivity_results' in locals() and sensitivity_results:
                    # Use the current directory as the default output directory
                    # stdout_dir = os.path.dirname(os.path.abspath(__file__))
                    plot_sensitivity_vs_far(sensitivity_results, stdout_dir, FAR_MIN, FAR_MAX)
            else:
                logging.error("Failed to process sensitivity results")

        else:
            logging.error("Failed to aggregate results, evaluation skipped")
            
    except Exception as e:
        logging.error(f"Critical error [{e}] in main execution: {traceback.format_exc()}")
    finally:
        # Always clean up the results directory at the end
        clean_results_directory(RESULTS_DIR)

    print("[*] Average:")
    print(sensitivity_results['auc'])
