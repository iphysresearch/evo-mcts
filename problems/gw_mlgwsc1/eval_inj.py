import h5py
import logging
import sys
from gen_inst import generate_datasets
import importlib.util
import os

def eval_inj(ix, inj_id, data, direction, RESULTS_DIR, gpt_runid):
    data_H1, data_L1, data_L1_times = data
    
    # Dynamically import the pipeline function from gpt_{gpt_runid}.py
    try:
        # Get the current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to the specific gpt_{gpt_runid}.py file
        module_path = os.path.join(current_dir, f'gpt_{gpt_runid}.py')
        
        # Load the module dynamically
        spec = importlib.util.spec_from_file_location(f"gpt_{gpt_runid}", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the pipeline function from the loaded module
        pipeline = getattr(module, 'pipeline_v2')
        logging.info(f"Successfully loaded pipeline from gpt_{gpt_runid}.py")
    except Exception as e:
        logging.error(f"Failed to load pipeline from gpt_{gpt_runid}.py: {e}")
        # Fallback to the default pipeline
        try:
            from gpt import pipeline_v2 as pipeline
            logging.info("Falling back to pipeline_v2")
        except:
            from gpt import pipeline
            logging.info("Falling back to default pipeline")
    
    peak_times, peak_heights, peak_deltat = pipeline(data_H1, data_L1, data_L1_times)

    assert len(peak_times) == len(peak_heights) == len(peak_deltat), \
        f"peak_times {len(peak_times)}, peak_heights {len(peak_heights)}, peak_deltat {len(peak_deltat)} \
          must have the same length"

    with h5py.File(os.path.join(RESULTS_DIR, f"inj_{ix}_{inj_id}_{direction}.hdf5"), "w") as f:
        f.create_dataset("peak_times", data=peak_times)
        f.create_dataset("peak_heights", data=peak_heights)
        f.create_dataset("peak_deltat", data=peak_deltat)


if __name__ == "__main__":

    ix = int(sys.argv[1])
    inj_id = int(sys.argv[2])
    RESULTS_DIR = sys.argv[3]
    gpt_runid = sys.argv[4]
    # Get foreground and background data with error handling
    try:
        [fdata_H1, fdata_L1, fdata_L1_times], [bdata_H1, bdata_L1, bdata_L1_times] = generate_datasets(ix, str(inj_id))
    except Exception as e:
        print(e)
        logging.error(f"Error generating datasets for injection {inj_id}: {e}")
        # return [], []

    eval_inj(ix, inj_id, [fdata_H1, fdata_L1, fdata_L1_times], "f", RESULTS_DIR, gpt_runid)

    eval_inj(ix, inj_id, [bdata_H1, bdata_L1, bdata_L1_times], "b", RESULTS_DIR, gpt_runid)