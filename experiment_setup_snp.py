
import os
from pathlib import Path
import datetime
import logging
import sys

MAIN_DIR = "/gpfs/gibbs/pi/gerstein/tu54/imaging_project/expression-prediction/thyroid"
EXP_NAME = "Thyroid-by-tile-NIC-CNN"

def get_run_folder(args):
    args_str = f"lr{args.learning_rate}-test_size{args.test_size}-batch_size{args.batch_size}-epochs{args.epochs}-column{args.snp_column}"
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f"run_{now_str}_{args_str}"

def create_path(parent_dir, child_dirs):
    path = parent_dir
    for child_dir in child_dirs:
        path = path / child_dir
        path.mkdir(exist_ok=True)
    return path

def initialize_experiment(args):
    current_path = create_path(Path(MAIN_DIR), ["data", EXP_NAME, get_run_folder(args)])
    return current_path
