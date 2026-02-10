"""
Script per l'analisi dei risultati degli esperimenti di tuning.

Questo script analizza i file di log generati dai moduli (flops_module, hardware_module)
e dalla pipeline di tuning, creando:
1. Un CSV per ogni esperimento che raggruppa i risultati
2. Un CSV totale con il miglior risultato per ogni esperimento

Uso:
    python analyze_results.py

L'utente verrà chiesto di selezionare la cartella parent che contiene gli esperimenti.
"""

import os
import sys
import csv
import json
import ast
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

# Try to import tkinter, fallback if not available
try:
    import tkinter as tk
    from tkinter import filedialog
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False


@dataclass
class ExperimentResult:
    """Rappresenta una singola iterazione di un esperimento"""
    iteration: int
    accuracy: Optional[float]
    flops: Optional[float]
    nparams: Optional[float]
    latency: Optional[float]
    hw_cost: Optional[float]
    hw_total_cost: Optional[float]
    hw_config: Optional[str]
    # hyperparams: Optional[Dict[str, Any]]
    score: Optional[float] = None  # Calcolato come -accuracy se non sono presenti moduli


class ResultsAnalyzer:
    """Analizzatore dei risultati degli esperimenti"""
    
    def __init__(self, experiment_dir: str):
        """
        Inizializza l'analizzatore.
        
        Args:
            experiment_dir: Cartella dell'esperimento contenente algorithm_logs/
        """
        self.experiment_dir = Path(experiment_dir)
        self.algorithm_logs_dir = self.experiment_dir / "algorithm_logs"
        self.results: List[ExperimentResult] = []
        self.has_flops_module = False
        self.has_hardware_module = False
        
    def load_results(self) -> bool:
        """
        Carica tutti i risultati dai file di log.
        
        Returns:
            True se i log sono stati caricati con successo, False altrimenti
        """
        if not self.algorithm_logs_dir.exists():
            print(f"  ⚠ Cartella algorithm_logs non trovata in {self.experiment_dir}")
            return False
        
        # Carica accuratezze
        accuracies = self._load_accuracies()
        if not accuracies:
            print(f"  ⚠ Nessun file acc_report.txt trovato")
            return False
        
        # Carica iperparametri
        # hyperparams_list = self._load_hyperparams()
        
        # Carica dati FLOPS
        flops_data = self._load_flops_data()
        if flops_data:
            self.has_flops_module = True
        
        # Carica dati hardware
        hw_data = self._load_hardware_data()
        if hw_data:
            self.has_hardware_module = True
        
        # Combina i dati
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
                # hyperparams=hyperparams_list[i] if i < len(hyperparams_list) else None,
            )
            
            # Calcola lo score: -accuracy se non ci sono moduli
            if result.accuracy is not None:
                if self.has_flops_module or self.has_hardware_module:
                    # Lo score è già stato calcolato durante il training,
                    # ma non è salvato nei log. Usiamo -accuracy come approssimazione
                    result.score = -result.accuracy
                else:
                    result.score = -result.accuracy
            
            self.results.append(result)
        
        return True
    
    def _load_accuracies(self) -> List[Optional[float]]:
        """Carica le accuratezze dal file acc_report.txt"""
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
            print(f"  ❌ Errore leggendo {acc_file}: {e}")
            return []
        
        return accuracies
    
    def _load_hyperparams(self) -> List[Optional[Dict[str, Any]]]:
        """Carica gli iperparametri dal file hyper-neural.txt"""
        hp_file = self.algorithm_logs_dir / "hyper-neural.txt"
        if not hp_file.exists():
            return []
        
        hyperparams = []
        try:
            with open(hp_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        hyperparams.append(None)
                        continue
                    try:
                        # Parsa il dict Python
                        hp_dict = ast.literal_eval(line)
                        hyperparams.append(hp_dict)
                    except Exception as e:
                        print(f"  ⚠ Errore parsing iperparametri: {e}")
                        hyperparams.append(None)
        except Exception as e:
            print(f"  ❌ Errore leggendo {hp_file}: {e}")
            return []
        
        return hyperparams
    
    def _load_flops_data(self) -> Optional[List[Tuple[float, float]]]:
        """Carica i dati FLOPS dal file flops_report.txt"""
        flops_file = self.algorithm_logs_dir / "flops_report.txt"
        if not flops_file.exists():
            return None
        
        flops_data = []
        try:
            with open(flops_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        flops_data.append((float(parts[0]), float(parts[1])))
        except Exception as e:
            print(f"  ❌ Errore leggendo {flops_file}: {e}")
            return None
        
        return flops_data if flops_data else None
    
    def _load_hardware_data(self) -> Optional[List[Tuple[float, float, float, str]]]:
        """Carica i dati hardware dal file hardware_report.txt"""
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
            print(f"  ❌ Errore leggendo {hw_file}: {e}")
            return None
        
        return hw_data if hw_data else None
    
    def get_best_result(self) -> Optional[ExperimentResult]:
        """Ritorna il risultato migliore (score minore)"""
        valid_results = [r for r in self.results if r.score is not None]
        if not valid_results:
            return None
        return min(valid_results, key=lambda r: r.score)
    
    def save_experiment_csv(self, output_csv: Path) -> bool:
        """
        Salva i risultati dell'esperimento in CSV.
        
        Args:
            output_csv: Path del file CSV di output
            
        Returns:
            True se il salvataggio è riuscito
        """
        if not self.results:
            print(f"  ⚠ Nessun risultato da salvare")
            return False
        
        try:
            with open(output_csv, 'w', newline='') as f:
                # Determina le colonne dinamicamente
                fieldnames = [
                    'iteration', 'accuracy', 'score',
                    'nparams', 'flops',
                    'latency', 'hw_cost', 'hw_total_cost', 'hw_config'
                ]
                
                # Aggiungi campi per iperparametri
                all_hp_keys = set()
                # for result in self.results:
                #     if result.hyperparams:
                #         all_hp_keys.update(result.hyperparams.keys())
                
                fieldnames.extend(sorted(all_hp_keys))
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in self.results:
                    row = {
                        'iteration': result.iteration,
                        'accuracy': result.accuracy,
                        'score': result.score,
                        'nparams': result.nparams,
                        'flops': result.flops,
                        'latency': result.latency,
                        'hw_cost': result.hw_cost,
                        'hw_total_cost': result.hw_total_cost,
                        'hw_config': result.hw_config,
                    }
                    
                    # Aggiungi iperparametri
                    # if result.hyperparams:
                    #     for key in all_hp_keys:
                    #         row[key] = result.hyperparams.get(key, '')
                    
                    writer.writerow(row)
            
            return True
        except Exception as e:
            print(f"  ❌ Errore salvando CSV: {e}")
            return False


def select_parent_directory() -> Optional[Path]:
    """
    Chiede all'utente di selezionare la cartella parent che contiene gli esperimenti.
    
    Returns:
        Path della cartella selezionata o None se cancellato
    """
    if TKINTER_AVAILABLE:
        root = tk.Tk()
        root.withdraw()  # Nascondi la finestra principale
        
        directory = filedialog.askdirectory(
            title="Seleziona la cartella parent che contiene gli esperimenti",
            initialdir=os.path.expand_user("~")
        )
        
        return Path(directory) if directory else None
    else:
        # Fallback: chiedi via command line
        print("GUI non disponibile, inserisci il percorso manualmente")
        path_input = input("Percorso della cartella parent contenente gli esperimenti: ").strip()
        if path_input and Path(path_input).exists():
            return Path(path_input)
        return None


def analyze_all_experiments(parent_dir: Path, output_dir: Optional[Path] = None):
    """
    Analizza tutti gli esperimenti nella cartella parent.
    
    Args:
        parent_dir: Cartella parent che contiene gli esperimenti
        output_dir: Cartella di output (default: parent_dir)
    """
    if output_dir is None:
        output_dir = parent_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Trova tutte le sottocartelle con algorithm_logs
    experiments = []
    for item in parent_dir.iterdir():
        if item.is_dir() and (item / "algorithm_logs").exists():
            experiments.append(item)
    
    if not experiments:
        print(f"❌ Nessun esperimento trovato in {parent_dir}")
        return
    
    print(f"\n📊 Trovati {len(experiments)} esperimenti\n")
    
    # Analizza ogni esperimento
    summary_data = []
    
    for exp_dir in sorted(experiments):
        print(f"📈 Analizzando: {exp_dir.name}")
        
        analyzer = ResultsAnalyzer(exp_dir)
        if not analyzer.load_results():
            continue
        
        # Salva CSV per questo esperimento
        if output_dir != parent_dir:
            exp_csv = output_dir / f"{exp_dir.name}_results.csv"
        else:
            exp_csv = exp_dir / f"{exp_dir.name}_results.csv"
        if analyzer.save_experiment_csv(exp_csv):
            print(f"  ✓ CSV salvato: {exp_csv.name}")
        
        # Raccoglie info per il CSV totale
        best_result = analyzer.get_best_result()
        if best_result:
            summary_data.append({
                'experiment': exp_dir.name,
                'best_iteration': best_result.iteration,
                'best_accuracy': best_result.accuracy,
                'best_score': best_result.score,
                'best_nparams': best_result.nparams,
                'best_flops': best_result.flops,
                'best_latency': best_result.latency,
                'best_hw_cost': best_result.hw_cost,
                'best_hw_total_cost': best_result.hw_total_cost,
                'best_hw_config': best_result.hw_config,
            })
            print(f"  ✓ Miglior risultato: iterazione {best_result.iteration}, "
                  f"accuracy={best_result.accuracy:.4f}, score={best_result.score:.4f}\n")
        else:
            print(f"  ⚠ Nessun risultato valido trovato\n")
    
    # Salva CSV totale
    if summary_data:
        summary_csv = output_dir / "summary_best_results.csv"
        try:
            with open(summary_csv, 'w', newline='') as f:
                fieldnames = [col for col in summary_data[0].keys()]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                # Ordina per score
                sorted_data = sorted(summary_data, key=lambda x: x['best_score'] if x['best_score'] is not None else float('inf'))
                writer.writerows(sorted_data)
            
            print(f"✅ CSV riepilogativo salvato: {summary_csv.name}")
        except Exception as e:
            print(f"❌ Errore salvando CSV riepilogativo: {e}")
    else:
        print("❌ Nessun esperimento valido caricato")


def main():
    """Funzione principale"""
    print("\n" + "="*70)
    print("  📊 ANALIZZATORE RISULTATI ESPERIMENTI - Symbolic DNN Tuner")
    print("="*70 + "\n")
    
    # Verifica se sono stati passati argomenti
    if len(sys.argv) > 1:
        # Modalità batch: usa argomenti da command line
        parent_dir = Path(sys.argv[1])
        if not parent_dir.exists():
            print(f"❌ Cartella non trovata: {parent_dir}")
            return
        
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else None
        if output_dir and not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Cartella input: {parent_dir}")
        if output_dir:
            print(f"📁 Cartella output: {output_dir}\n")
    else:
        # Modalità interattiva
        parent_dir = select_parent_directory()
        if parent_dir is None:
            print("❌ Nessuna cartella selezionata")
            return
        
        print(f"\n📁 Cartella selezionata: {parent_dir}\n")
        
        # Chiedi se salvare i CSV in una cartella diversa (solo se non in stdin)
        try:
            use_custom_output = input("Salvare i risultati in una cartella diversa? (s/n): ").lower() == 's'
            
            output_dir = None
            if use_custom_output:
                if TKINTER_AVAILABLE:
                    root = tk.Tk()
                    root.withdraw()
                    output_path = filedialog.askdirectory(
                        title="Seleziona la cartella di output",
                        initialdir=str(parent_dir)
                    )
                    if output_path:
                        output_dir = Path(output_path)
                    else:
                        print("⚠ Nessuna cartella di output selezionata, uso la cartella parent")
                else:
                    output_path = input("Percorso della cartella di output: ").strip()
                    if output_path and Path(output_path).exists():
                        output_dir = Path(output_path)
                    else:
                        print("⚠ Percorso non valido, uso la cartella parent")
        except EOFError:
            # Se stdin è chiuso, usa la modalità non-interattiva
            output_dir = None
    
    # Analizza gli esperimenti
    analyze_all_experiments(parent_dir, output_dir)
    
    print("\n" + "="*70)
    print("  ✅ Analisi completata!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
