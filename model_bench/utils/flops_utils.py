import os
import sys
from tqdm import tqdm
from pathlib import Path
from components.colors import colors

# Set TensorFlow environment variables to reduce log verbosity
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
    :param model: Keras model
    :return: Total FLOPs and detailed FLOPs dictionary
    """

    model = load_model_simple(model_path)
    if model is None:
        return None
    
    try:
        flops, res_dict = analyze_model(model)

        return {
            "model": os.path.basename(model_path),
            "path": model_path,
            "total_flops": flops.total_float_ops,
            "gflops": flops.total_float_ops / 1e9,
            "mflops": flops.total_float_ops / 1e6,
            "details": res_dict
        }
    except Exception as e:
        print(colors.FAIL, f"Error calculating FLOPs for model {model_path}: {e}", colors.ENDC)
        return None

def calculate_multiple_models_flops(directory_path, recursive=True):
    """
    Calculate FLOPs for all .keras models in a directory
    :param directory_path: Path to directory containing .keras files
    :param recursive: If True, search in subdirectories too
    :return: list of results dicts
    """
    results = []
    
    if not os.path.isdir(directory_path):
        print(f"Error: {directory_path} is not a directory")
        return results
    
    model_files = []
    
    if recursive:
        print(colors.CYAN, f"Scanning directory recursively: {directory_path}", colors.ENDC)
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if check_if_path_is_model(file_path):
                    model_files.append(file_path)
    else:
        print(colors.CYAN, f"Scanning directory: {directory_path}", colors.ENDC)
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if check_if_path_is_model(file_path):
                model_files.append(file_path)

    if not model_files:
        print(colors.FAIL, f"No models files found in {directory_path}", colors.ENDC)
        return results
    print(colors.OKGREEN, f"Found {len(model_files)} model(s) to analyze...", colors.ENDC)

    with tqdm(total=len(model_files), desc="Calculating FLOPs", unit="model") as pbar:
        for model_path in model_files:
            result = calculate_model_flops(model_path)
            if result:
                results.append(result)
            pbar.update(1) # update progress bar
    print(colors.OKGREEN, f"FLOPs calculation completed. total results: {len(results)}", colors.ENDC)
    return results