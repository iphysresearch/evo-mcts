import hydra
import logging 
import os
from pathlib import Path
import subprocess
from utils.utils import init_client
from ahd_adapter import AHD as LHH

ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    workspace_dir = Path(cfg.get('workspace_dir', None)) if hasattr(cfg, 'workspace_dir') or 'workspace_dir' in cfg else Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")
    logging.info(f"Using LLM: {cfg.get('model', cfg.llm_client.model)}")
    logging.info(f"Using Algorithm: {cfg.algorithm}")
    logging.info(f"Using Temperature: {cfg.llm_client.temperature}")

    client = init_client(cfg)
    # Main algorithm
    lhh = LHH(cfg, ROOT_DIR, workspace_dir, client)
    best_code_overall, best_code_path_overall = lhh.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Code Path Overall: {best_code_path_overall}")

if __name__ == "__main__":
    main()