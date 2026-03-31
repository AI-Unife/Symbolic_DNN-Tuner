"""
analizzare i file di log contenuti in algorithm_logs e per ognuno di essi graficarli:
        - accuracy, score e params come line plot o simili
        - evidence: contare quante volte ogni azione ha funzionato e quante no. (bar plot)
        - diangosis e tuning potrebbe essere interessante mostrare sia quante volte ogni problema e 
        soluzione sono stati trovati/applicati (bar plot) sia trovare il modo di mostrare per ogni 
        iterazione quali problemi sono stati trovati e che soluzioni sono state applicate.

Usage:
    python 1-grafici.py

The user will be asked to select the parent folder containing the experiments.
"""

import os
import sys
import csv
import json
import yaml
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt

# Try to import tkinter, fallback if not available
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = False
except ImportError:
    TKINTER_AVAILABLE = False

def select_parent_directory() -> Optional[Path]:
    """
    Ask the user to select the parent folder containing the experiments.
    
    Returns:
        Path of the selected folder or None if cancelled
    """
    if TKINTER_AVAILABLE:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        directory = filedialog.askdirectory(
            title="Select the parent folder containing the experiments",
            initialdir=os.path.expand_user("~")
        )
        
        return Path(directory) if directory else None
    else:
        # Fallback: ask via command line
        print("GUI not available, please enter the path manually")
        path_input = input("Path of the parent folder containing the experiments: ").strip()
        if path_input and Path(path_input).exists():
            return Path(path_input)
        return None


@dataclass
class ExperimentResult:
    """Represents a single iteration of an experiment"""
    iteration: int
    accuracy: Optional[float]
    flops: Optional[float]
    nparams: Optional[float]
    latency: Optional[float]
    hw_cost: Optional[float]
    hw_total_cost: Optional[float]
    hw_config: Optional[str]
    score: Optional[float]
    # hyperparams: Optional[Dict[str, Any]]
    score: Optional[float] = None  # Calculated as -accuracy if no modules are present

class ResultsAnalyzer:
    """Analyzer for experiment results"""
    
    def __init__(self, experiment_dir: str):
        """
        Initialize the analyzer.
        
        Args:
            experiment_dir: Experiment folder containing algorithm_logs/
        """
        self.experiment_dir = Path(experiment_dir)
        self.algorithm_logs_dir = self.experiment_dir / "algorithm_logs"
        self.results: List[ExperimentResult] = []
        self.has_flops_module = False
        self.has_hardware_module = False
        self.config: Dict[str, Any] = {}  # Experiment configuration

    def load_results(self) -> bool:
        """
        Load all results from log files.
        
        Returns:
            True if logs were loaded successfully, False otherwise
        """
        # Load config.yaml file
        self._load_config_yaml()
        
        if not self.algorithm_logs_dir.exists():
            print(f"  algorithm_logs folder not found in {self.experiment_dir}")
            return False
        
        # Load accuracies
        accuracies = self._load_accuracies()
        if not accuracies:
            print(f"  No acc_report.txt file found")
            return False
        
        scores = self._load_scores()
        if scores:
            print(f"  score_report.txt file found, will use scores calculated during training")
        else:
            print(f"  No score_report.txt file found, will use -accuracy as approximation")
        
        # Load hyperparameters
        # hyperparams_list = self._load_hyperparams()
        
        # Load FLOPS data
        flops_data = self._load_flops_data()
        if flops_data:
            self.has_flops_module = True
        
        # Load hardware data
        hw_data = self._load_hardware_data()
        if hw_data:
            self.has_hardware_module = True
        
        # Combine the data
        max_iterations = max(
            len(accuracies),
            # len(hyperparams_list),
            len(flops_data) if flops_data else 0,
            len(hw_data) if hw_data else 0
        )
        
        for i in range(max_iterations):
            result = ExperimentResult(
                iteration=i + 1,
                accuracy=accuracies[i] if i < len(accuracies) else None,
                flops=flops_data[i][1] if flops_data and i < len(flops_data) else None,
                nparams=flops_data[i][0] if flops_data and i < len(flops_data) else None,
                latency=hw_data[i][0] if hw_data and i < len(hw_data) else None,
                hw_cost=hw_data[i][1] if hw_data and i < len(hw_data) else None,
                hw_total_cost=hw_data[i][2] if hw_data and i < len(hw_data) else None,
                hw_config=hw_data[i][3] if hw_data and i < len(hw_data) else None,
                score=scores[i] if scores and i < len(scores) else None,
                # hyperparams=hyperparams_list[i] if i < len(hyperparams_list) else None,
            )
            
            # Calculate score: -accuracy if no modules are present
            if result.score is None:
                if result.accuracy is not None:
                    result.score = -result.accuracy
                else:
                    result.score = None
            
            self.results.append(result)
        
        return True
    
    def _load_config_yaml(self) -> None:
        """Load configuration from config.yaml file"""
        config_file = self.experiment_dir / "config.yaml"
        if not config_file.exists():
            print(f"  config.yaml file not found in {self.experiment_dir}")
            self.config = {}
            return
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"  Error reading config.yaml: {e}")
            self.config = {}

    def _load_accuracies(self) -> List[Optional[float]]:
        """Load accuracies from acc_report.txt file"""
        acc_file = self.algorithm_logs_dir / "acc_report.txt"
        if not acc_file.exists():
            return []
        
        accuracies = []
        try:
            with open(acc_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == "None" or line == "":
                        accuracies.append(None)
                    else:
                        accuracies.append(float(line))
        except Exception as e:
            print(f"  Error reading {acc_file}: {e}")
            return []
        
        return accuracies
    
    def _load_scores(self) -> List[Optional[float]]:
        """Load scores from score_report.txt file"""
        score_file = self.algorithm_logs_dir / "score_report.txt"
        if not score_file.exists():
            return []
        
        scores = []
        try:
            with open(score_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line == "None" or line == "":
                        scores.append(None)
                    else:
                        scores.append(float(line))
        except Exception as e:
            print(f"  Error reading {score_file}: {e}")
            return []
        
        return scores
    
    def _load_flops_data(self) -> Optional[List[Tuple[float, float]]]:
        """Load FLOPS data from flops_report.txt file"""
        flops_file = self.algorithm_logs_dir / "flops_report.txt"
        if not flops_file.exists():
            flops_file = self.algorithm_logs_dir / "params_report.txt"
            if not flops_file.exists():
                flops_file = self.experiment_dir / "flops_report.txt"
                if not flops_file.exists():
                    return None

        flops_data = []
        try:
            with open(flops_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(' ')
                    if len(parts) >= 2:
                        flops_data.append((float(parts[0]), float(parts[1])))
                    else:
                        # If it's a file with only params, add None for FLOPS
                        flops_data.append((float(parts[0]), None))
        except Exception as e:
            print(f"  Error reading {flops_file}: {e}")
            return None
        
        return flops_data if flops_data else None
    
    def _load_hardware_data(self) -> Optional[List[Tuple[float, float, float, str]]]:
        """Load hardware data from hardware_report.txt file"""
        hw_file = self.algorithm_logs_dir / "hardware_report.txt"
        if not hw_file.exists():
            return None
        
        hw_data = []
        try:
            with open(hw_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(',')
                    if len(parts) >= 4:
                        hw_data.append((
                            float(parts[0]),  # latency
                            float(parts[1]),  # cost
                            float(parts[2]),  # total_cost
                            parts[3].strip()   # config
                        ))
        except Exception as e:
            print(f"  Error reading {hw_file}: {e}")
            return None
        
        return hw_data if hw_data else None


def analyze_all_experiments(parent_dir: Path, output_dir: Optional[Path] = None):
    """
    Analyze all experiments in the parent folder.
    
    Args:
        parent_dir: Parent folder containing the experiments
        output_dir: Output folder (default: parent_dir)
    """
    if output_dir is None:
        output_dir = parent_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all subfolders with algorithm_logs
    experiments = []
    for item in parent_dir.iterdir():
        if item.is_dir() and (item / "algorithm_logs").exists():
            experiments.append(item)
    
    if not experiments:
        print(f"No experiments found in {parent_dir}")
        return
    
    print(f"\nFound {len(experiments)} experiments\n")
    
    # Analyze each experiment
    summary_data = []
    
    for exp_dir in sorted(experiments):
        print(f"📈 Analyzing: {exp_dir.name}")
        
        analyzer = ResultsAnalyzer(exp_dir)
        if not analyzer.load_results():
            continue

    # Iniziano modifiche   !!! 
        # creo 3 grafici per ogni esperimento nella stessa figura: accuracy, score e nparams (se disponibili)
        num_plots = 0
        plotAccuracy = False
        plotScore = False
        plotParams = False

        for i in range(len(analyzer.results)):
            if analyzer.results[i].accuracy is not None:
                num_plots += 1
                plotAccuracy = True
                break
        for i in range(len(analyzer.results)):
            if analyzer.results[i].score is not None and analyzer.results[i].score < 0:
                num_plots += 1
                plotScore = True
                break
        for i in range(len(analyzer.results)):
            if analyzer.results[i].nparams is not None:
                num_plots += 1
                plotParams = True
                break

        print(f"  - {num_plots} plots to generate for {exp_dir.name}")
        
        if num_plots == 0:
            continue

        fig, axes = plt.subplots(1,num_plots, figsize=(10 * num_plots, 6))
        # Grafici linea per l'accuratezza
        idx = 0
        if plotAccuracy:
            axes[idx].plot(
                [r.iteration for r in analyzer.results if r.accuracy is not None],
                [r.accuracy for r in analyzer.results if r.accuracy is not None],
                label=exp_dir.name, color='red'
            )
            axes[idx].set_title(f"Accuracy - {exp_dir.name}")
            axes[idx].set_xlabel("Iteration")
            axes[idx].set_ylabel("Accuracy")
            axes[idx].grid()
            axes[idx].plot()
            idx += 1

        # Grafici linea per lo score
        if plotScore:
            axes[idx].plot(
                [r.iteration for r in analyzer.results if r.score is not None and r.score<0], 
                [r.score for r in analyzer.results if r.score is not None and r.score<0],
                label=exp_dir.name, color='blue'
            )
            axes[idx].set_title(f"Score - {exp_dir.name}")
            axes[idx].set_xlabel("Iteration")
            axes[idx].set_ylabel("Score")
            axes[idx].grid()
            axes[idx].plot()
            idx += 1

        # Grafici linea per il numero di parametri
        if plotParams:
            axes[idx].plot(
                [r.iteration for r in analyzer.results if r.nparams is not None],
                [r.nparams for r in analyzer.results if r.nparams is not None],
                label=exp_dir.name, color='green'
            )
            axes[idx].set_title(f"Number of Parameters - {exp_dir.name}")
            axes[idx].set_xlabel("Iteration")
            axes[idx].set_ylabel("Number of Parameters")
            axes[idx].grid()
            axes[idx].plot()

        plt.tight_layout()
        plt.savefig(output_dir / f"{exp_dir.name}_graphs.png")
        plt.close()


def main():
    """Main function"""
    print("\n" + "="*70)
    print("  EXPERIMENT RESULTS ANALYZER - Symbolic DNN Tuner")
    print("="*70 + "\n")
    
    # Check if arguments have been passed
    if len(sys.argv) > 1:
        # Batch mode: use command line arguments
        parent_dir = Path(sys.argv[1])
        if not parent_dir.exists():
            print(f"Folder not found: {parent_dir}")
            return
        
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Input folder: {parent_dir}")
        if output_dir:
            print(f"Output folder: {output_dir}\n")
    else:
        # Interactive mode
        parent_dir = select_parent_directory()
        if parent_dir is None:
            print("No folder selected")
            return
        
        print(f"\nSelected folder: {parent_dir}\n")
        
        # Ask if save output in a different folder (only if not in stdin)
        try:
            use_custom_output = input("Save results in a different folder? (y/n): ").lower() == 'y'
            
            output_dir = None
            if use_custom_output:
                if TKINTER_AVAILABLE:
                    root = tk.Tk()
                    root.withdraw()
                    output_path = filedialog.askdirectory(
                        title="Select the output folder",
                        initialdir=str(parent_dir)
                    )
                    if output_path:
                        output_dir = Path(output_path)
                    else:
                        print("⚠ No output folder selected, using parent folder")
                else:
                    output_path = input("Output folder path: ").strip()
                    if output_path and Path(output_path).exists():
                        output_dir = Path(output_path)
                    else:
                        print("⚠ Invalid path, using parent folder")
        except EOFError:
            # If stdin is closed, use non-interactive mode
            output_dir = None
    
    # Analyze experiments
    analyze_all_experiments(parent_dir, output_dir)
    
    print("\n" + "="*70)
    print("  Analysis completed!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
