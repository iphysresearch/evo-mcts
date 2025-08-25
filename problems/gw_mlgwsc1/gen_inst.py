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
# DATA_DIR configuration - use environment variable or hostname-based default
import socket
hostname = socket.gethostname()

# First try to get DATA_DIR from environment variable
DATA_DIR = os.getenv('DATA_DIR')

if DATA_DIR is None:
    # Fallback to hostname-based configuration
    if hostname == 'login05':
        DATA_DIR = '/work1/hewang/HW/data'    # Dongfang
    elif hostname == '1f7472014c87':
        DATA_DIR = '/home/nvme0n1/data'    # A800
    elif hostname == 'Gravitation-wave':
        DATA_DIR = '/home/main/LVK_strain_data/data'    # A6000
    else:
        # Generic fallback - use environment variable or default
        DATA_DIR = '/home/nvme0n1/data'  # Default fallback
        logging.warning(f"Unknown hostname: {hostname}, using default DATA_DIR: {DATA_DIR}")

# Validate that DATA_DIR exists
if not os.path.exists(DATA_DIR):
    logging.error(f"DATA_DIR not found: {DATA_DIR}")
    logging.error("Please set DATA_DIR environment variable to the correct path")
    logging.error("Example: export DATA_DIR=/your/path/to/data")
    exit(1)

logging.info(f"Using DATA_DIR: {DATA_DIR}")
SET_NUMBER = 4          # Set number for the simulation
DURATION = 604800       # Duration in seconds (1 week)
# DURATION = 2629746       # Duration in seconds (1 month)
RANDOM_SEED = 40       # Random seed for reproducibility 
START_TIME = 0         # Start time of the simulation
FAR_MIN = 4
FAR_MAX = 1000

# Construct file paths for data files
file_template = f'set{SET_NUMBER}_{DURATION}dur_seed{RANDOM_SEED}_start{START_TIME}.hdf'
foreground_path = os.path.join(DATA_DIR, f'foreground_{file_template}')
background_path = os.path.join(DATA_DIR, f'background_{file_template}')
injection_path = os.path.join(DATA_DIR, f'injections_{file_template}')


def get_injection_ids(debug=False):
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
    
def generate_datasets(ix, inj_id):
    injection_ids, _, detectors = get_injection_ids()

    # Load foreground data
    logging.debug(f"Injection {inj_id} ({ix+1}/{len(injection_ids)})")
    fdata = {}
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

    # Load background data
    bdata = {}
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

    return [fdata['H1'].value, fdata['L1'].value, fdata['L1'].times.value], [bdata['H1'].value, bdata['L1'].value, bdata['L1'].times.value]

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.debug(123)
    get_injection_ids()
    # generate_datasets()
    logging.debug(456)