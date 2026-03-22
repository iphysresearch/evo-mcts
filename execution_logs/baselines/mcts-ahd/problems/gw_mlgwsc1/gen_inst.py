# Import required libraries
import os
import h5py
from tqdm import tqdm
import numpy as np
import gwpy
import pandas as pd
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
import logging

# Suppress LAL redirection warning
import warnings
warnings.filterwarnings("ignore", "Wswiglal-redir-stdio")
import lal


# Define simulation parameters
# DATA_DIR = '/home/main/gwtoolkit_project/gwtoolkit/benchmark/ml-mock-data-challenge-1/data'
#DATA_DIR = '/home/main/LVK_strain_data/data'  # For 1 month data  # A6000# 
import socket
hostname = socket.gethostname()
if hostname == 'Gravitation-wave':
    DATA_DIR = '/home/main/LVK_strain_data/data'    # A6000
elif hostname == '1f7472014c87':
    DATA_DIR = '/home/nvme0n1/data'    # A800
elif hostname == 'login03':                      
    DATA_DIR = '/work1/hewang/HW/data'    # Dongfang 登录节点
else:
    DATA_DIR = '/work1/hewang/HW/data'    # Dongfang 其他节点
    # logging.error(f"Unknown hostname: {hostname}")
    # exit(1)
SET_NUMBER = 4          # Set number for the simulation
# DURATION = 32000       # Duration in seconds
# DURATION = 86400       # Duration in seconds (1 day)
# DURATION = 86400*3       # Duration in seconds (3 day)
DURATION = 604800       # Duration in seconds (1 week)
# DURATION = 2629746       # Duration in seconds (1 month)
RANDOM_SEED = 40       # Random seed for reproducibility 
START_TIME = 0         # Start time of the simulation
# FAR_MIN = 1
FAR_MIN = 4
# FAR_MIN = 1e2
FAR_MAX = 1000

# Construct file paths for data files
file_template = f'set{SET_NUMBER}_{DURATION}dur_seed{RANDOM_SEED}_start{START_TIME}.hdf'
foreground_path = os.path.join(DATA_DIR, f'foreground_{file_template}')
background_path = os.path.join(DATA_DIR, f'background_{file_template}')
injection_path = os.path.join(DATA_DIR, f'injections_{file_template}')


def get_injection_ids(debug=False):
    max_retries = 10
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            with h5py.File(foreground_path, 'r') as fp:
                if debug:
                    print("File structure:")
                    print("├── Keys:", fp.keys())
                detectors = list(fp.keys())
                if debug:
                    print(f"├── Detectors: {detectors}")
                # Get all injection IDs from first detector
                injection_ids = list(fp[detectors[0]].keys())
                
                # Calculate duration for each injection and sort by duration (longest to shortest)
                injection_durations = []
                for inj_id in injection_ids:
                    strain_data = fp[detectors[0]][inj_id]
                    duration = strain_data.shape[0] * strain_data.attrs['delta_t']
                    injection_durations.append((inj_id, duration))
                
                # Sort by duration in descending order
                injection_durations.sort(key=lambda x: x[1], reverse=True)
                injection_ids = [inj_id for inj_id, _ in injection_durations]
                durations = {inj_id: duration for inj_id, duration in injection_durations}
                
                if debug:
                    print(f"├── Number of Injection IDs: {len(injection_ids)}")
                    # Print duration for each injection
                    for inj_id, duration in injection_durations:
                        print(f"├── Injection {inj_id}, Duration: {duration} seconds")
                
                return injection_ids, durations, detectors
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                logging.warning(f"File access error (attempt {attempt+1}/{max_retries}): {e}. Waiting {retry_delay} seconds before retry...")
                import time
                time.sleep(retry_delay)
            else:
                logging.error(f"Failed to access file after {max_retries} attempts: {e}")
                raise
    
def generate_datasets(ix, inj_id):
    injection_ids, _, detectors = get_injection_ids()

    # Load foreground data
    logging.debug(f"Injection {inj_id} ({ix+1}/{len(injection_ids)})")
    fdata = {}
    max_retries = 10
    retry_delay = 2  # seconds
    
    # Load foreground data with retry mechanism
    for attempt in range(max_retries):
        try:
            with h5py.File(foreground_path, 'r') as fp:
                logging.debug(f"└── Foreground data")
                for det in detectors:
                    # Load data for each injection ID
                    strain_data = fp[det][inj_id]
                    logging.debug(f"\t└── Detector {det} shape: {strain_data.shape}")
                    fdata[det] = TimeSeries(data=strain_data[()],
                                        t0=strain_data.attrs['start_time'],
                                        dt=strain_data.attrs['delta_t'],
                                        name=det,
                                        unit='strain')
            break  # Successfully loaded data, exit retry loop
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                logging.warning(f"Foreground file access error for injection {inj_id} (attempt {attempt+1}/{max_retries}): {e}. Waiting {retry_delay} seconds before retry...")
                import time
                time.sleep(retry_delay)
            else:
                logging.error(f"Failed to access foreground file for injection {inj_id} after {max_retries} attempts: {e}")
                raise

    # Load background data with retry mechanism
    bdata = {}
    for attempt in range(max_retries):
        try:
            with h5py.File(background_path, 'r') as fp:
                logging.debug(f"└── Background data")
                for det in detectors:
                    # Load data for each injection ID
                    strain_data = fp[det][inj_id]
                    logging.debug(f"\t└── Detector {det} shape: {strain_data.shape}")
                    bdata[det] = TimeSeries(data=strain_data[()],
                                        t0=strain_data.attrs['start_time'],
                                        dt=strain_data.attrs['delta_t'],
                                        name=det,
                                        unit='strain')
            break  # Successfully loaded data, exit retry loop
        except (IOError, OSError) as e:
            if attempt < max_retries - 1:
                logging.warning(f"Background file access error for injection {inj_id} (attempt {attempt+1}/{max_retries}): {e}. Waiting {retry_delay} seconds before retry...")
                import time
                time.sleep(retry_delay)
            else:
                logging.error(f"Failed to access background file for injection {inj_id} after {max_retries} attempts: {e}")
                raise

    return [fdata['H1'].value, fdata['L1'].value, fdata['L1'].times.value], [bdata['H1'].value, bdata['L1'].value, bdata['L1'].times.value]

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.debug(123)
    get_injection_ids()
    # generate_datasets()
    logging.debug(456)