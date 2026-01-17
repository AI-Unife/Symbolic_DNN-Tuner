"""
FLOPs utility functions for calculating FLOPs of Keras models.

This module provides functions to calculate FLOPs for Keras models
and to analyze multiple models in a directory.
"""
from tqdm import tqdm
from pathlib import Path
from components.colors import colors

# Set TensorFlow environment variables to reduce log verbosity
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow import keras
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
tf.get_logger().setLevel('ERROR')

from flops.flops_calculator import analyze_model
from utils.model_utils import check_if_path_is_model, load_model_simple

def calculate_model_flops(model_path):
    """
    Calculate the FLOPs of the given Keras model.
    :param model_path: Keras model Path
    :return: Total FLOPs and detailed FLOPs dictionary
    """
    model_path = Path(model_path)

    model = load_model_simple(str(model_path))
    if model is None:
        return None
    
    try:
        flops, res_dict = analyze_model(model)

        return {
            "model": Path(model_path).name,
            "path": str(model_path),
            "total_flops": flops.total_float_ops,
            "gflops": flops.total_float_ops / 1e9,
            "mflops": flops.total_float_ops / 1e6,
            "details": res_dict
        }
    except Exception as e:
        print(colors.FAIL + f"Error calculating FLOPs for model {model_path}: {e}" + colors.ENDC)
        return None

def calculate_multiple_models_flops(directory_path, recursive=True):
    """
    Calculate FLOPs for all .keras models in a directory
    :param directory_path: Path to directory containing .keras files
    :param recursive: If True, search in subdirectories too
    :return: list of results dicts
    """
    results = []

    directory_path = Path(directory_path)
    
    if not directory_path.is_dir():
        print(colors.FAIL + f"{directory_path} is not a valid directory." + colors.ENDC)
        return results
    
    model_files = []
    
    if recursive:
        print(colors.CYAN + f"Scanning directory recursively: {directory_path}" + colors.ENDC)
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and check_if_path_is_model(str(file_path)):
                model_files.append(str(file_path))
    else:
        print(colors.CYAN + f"Scanning directory: {directory_path}" + colors.ENDC)
        for file_path in directory_path.iterdir():
            if file_path.is_file() and check_if_path_is_model(str(file_path)):
                model_files.append(str(file_path))

    if not model_files:
        print(colors.FAIL + f"No models files found in {directory_path}" + colors.ENDC)
        return results
    print(colors.OKGREEN + f"Found {len(model_files)} model(s) to analyze..." + colors.ENDC)

    with tqdm(total=len(model_files), desc="Calculating FLOPs", unit="model") as pbar:
        for model_path in model_files:
            result = calculate_model_flops(model_path)
            if result:
                results.append(result)
            pbar.update(1) # update progress bar
    print(colors.OKGREEN + f"FLOPs calculation completed. total results: {len(results)}" + colors.ENDC)
    return results