"""
Export utility functions for saving test results to CSV files.

This module provides functions to export latency and FLOPs analysis results
to CSV format for easier analysis and reporting.
"""

import csv
import questionary
from datetime import datetime
from pathlib import Path
from components.colors import colors

def ensure_export_directory(export_type):
    """
    create export directory path if it doesn't exist
    :param export_type: Type of export ('latency' or 'flops')
    :return: Absolute path of the directory
    """
    project_root = Path(__file__).parent.parent
    default_dir = project_root / "exports" / export_type
    
    use_default = questionary.confirm(
        f"Do you want to use the default export directory:\n {default_dir}?",
        default=True
    ).ask()

    if use_default:
        default_dir.mkdir(parents=True, exist_ok=True)
        print(colors.CYAN + f"Using default directory: {default_dir}" + colors.ENDC)
        return default_dir

    while True:
        custom_dir = questionary.path(
            "choose export directory", only_directories=True
        ).ask()

        if not custom_dir:
            print(colors.WARNING + "No directory provided. Insert a valid directory path." + colors.ENDC)
            continue

        custom_path = Path(custom_dir).expanduser().resolve()

        try:
            custom_path.mkdir(parents=True, exist_ok=True)
            print(colors.OKGREEN + f"✓ Export directory set to: {custom_path}" + colors.ENDC)
            return custom_path
        except Exception as e:
            print(colors.FAIL + f"✗ Cannot create directory: {e}" + colors.ENDC)
            retry = questionary.confirm("Try another directory?").ask()
            if not retry:
                default_dir.mkdir(parents=True, exist_ok=True)
                return default_dir

def generate_filename(prefix, extension="csv"):
    """Generate a timestamped filename with the given prefix and extension."""

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{timestamp}.{extension}"

def export_latency_results(results):
    """
    Export latency results to a CSV file.
    :param results: List of latency result dicts
    :param filename: Optional filename. If None, a timestamped filename will be generated.
    :return: Full path of the exported file,  or None if export failed
    """

    if not results:
        print(colors.WARNING, "No latency results to export.", colors.ENDC)
        return

    export_dir = ensure_export_directory("latency")

    filename = generate_filename("latency")
    filepath = export_dir / filename

    try:
        with open(filepath, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["Model", "HW_config", "Latency(s)", "Latency(ms)"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    "Model": result.get("model", "N/A"),
                    "HW_config": result.get("HW config", "N/A"),
                    "Latency(s)": result.get("latency(s)", 0),
                    "Latency(ms)": result.get("latency(s)", 0) * 1000
                }
                writer.writerow(row)
        
        print(colors.OKGREEN + f"✓ Latency results exported to: {filepath}" + colors.ENDC)
        return str(filepath)
    except Exception as e:
        print(colors.FAIL + f"✗ Error exporting latency results: {e}" + colors.ENDC)
        return None


def export_flops_results(results):
    """
    Export FLOPs analysis results to a CSV file.
    :param results: List of FLOPs result dicts
    :param filename: Optional filename. If None, a timestamped filename will be generated.
    :return: Full path of the exported file, or None if export failed
    """
    if not results:
        print(colors.WARNING + "No FLOPs results to export." + colors.ENDC)
        return None

    export_dir = ensure_export_directory("flops")

    filename = generate_filename("flops")
    filepath = export_dir / filename

    try:
        with open(filepath, mode="w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["Model", "Path", "Total_FLOPs", "GFLOPs", "MFLOPs"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    "Model": result.get("model", "N/A"),
                    "Path": result.get("path", "N/A"),
                    "Total_FLOPs": result.get("total_flops", 0),
                    "GFLOPs": result.get("gflops", 0),
                    "MFLOPs": result.get("mflops", 0)
                }
                writer.writerow(row)

        print(colors.OKGREEN + f"✓ FLOPs results exported to: {filepath}" + colors.ENDC)
        return str(filepath)
    except Exception as e:
        print(colors.FAIL + f"✗ Error exporting FLOPs results: {e}" + colors.ENDC)
        return None
