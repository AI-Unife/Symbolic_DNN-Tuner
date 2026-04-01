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
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import pandas as pd

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
    evidence: Optional[Tuple[Tuple[str, str], bool]]        # Nuovo campo per memorizzare i dati di evidence
    diagnosis: Optional[List[str]]                          # Nuovo campo per memorizzare le diagnosi trovate in ogni iterazione
    tuning: Optional[List[str]]                             # Nuovo campo per memorizzare le soluzioni di tuning applicate in ogni iterazione
    score: Optional[float] = None  # Calculated as -accuracy if no modules are present
    # hyperparams: Optional[Dict[str, Any]]

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

        evidence_data = self._load_evidence_data()      # Carico i dati di evidence

        diagnosis_data = self._load_diagnosis_data()      # Carico i dati di diagnosis (se disponibili)

        tuning_data = self._load_tuning_data()      # Carico i dati di tuning (se disponibili)
        
        # Combine the data
        max_iterations = max(
            len(accuracies),
            # len(hyperparams_list),
            len(flops_data) if flops_data else 0,
            len(hw_data) if hw_data else 0,
            len(evidence_data) if evidence_data else 0,      #considero anche la lunghezza dei dati di evidence per determinare il numero di iterazioni da analizzare
            len(diagnosis_data) if diagnosis_data else 0,     #considero anche la lunghezza dei dati di diagnosis per determinare il numero di iterazioni da analizzare
            len(tuning_data) if tuning_data else 0           #considero anche la lunghezza dei dati di tuning per determinare il numero di iterazioni da analizzare
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
                evidence=evidence_data[i] if evidence_data and i < len(evidence_data) else None,     # Aggiungo i dati di evidence al risultato
                diagnosis=diagnosis_data[i] if diagnosis_data and i < len(diagnosis_data) else None,     # Aggiungo i dati di diagnosis al risultato
                tuning=tuning_data[i] if tuning_data and i < len(tuning_data) else None               # Aggiungo i dati di tuning al risultato
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
    
    # !!! NUOVA FUNZIONE PER CARICARE I DATI DI EVIDENCE
    def _load_evidence_data(self) -> Optional[List[Tuple[Tuple[str, str], bool]]]:
        """Load evidence data from evidence.txt file"""
        evidence_file = self.algorithm_logs_dir / "evidence.txt"
        if not evidence_file.exists():
            return None
        
        evidence_data = []
        try:
            with open(evidence_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    line = line.replace("[","").replace("]","").replace("action", "").replace("(", "").replace(")", "").replace(" ", "")    # Rimuovo le parole e i simboli superflui per ottenere solo i dati
                    parts = line.split(',')
                    if len(parts) >= 0:
                        for i in range (0, len(parts)-2, 3):    #i dati sono in tripletta, quindi ciclo con step di 3 per prenderli correttamente
                            action = (parts[i+0], parts[i+1])
                            success = parts[i+2].lower() == 'true'  # Converto la stringa che trovo in booleano
                            evidence_data.append((action, success))

        except Exception as e:
            print(f"  Error reading {evidence_file}: {e}")
            return None
        
        return evidence_data if evidence_data else None
    
    # !!! NUOVA FUNZIONE PER CARICARE I DATI DI DIAGNOSIS
    def _load_diagnosis_data(self) -> Optional[List[List[str]]]:
        """Load diagnosis data from diagnosis_symbolic_logs.txt file"""
        diagnosis_file = self.algorithm_logs_dir / "diagnosis_symbolic_logs.txt"
        if not diagnosis_file.exists():
            return None
        
        diagnosis_data = []
        try:
            with open(diagnosis_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    line = line.replace("[","").replace("]","").replace("'", "").replace(" ", "")    # Rimuovo i simboli superflui per ottenere solo i dati
                    diagnoses = line.split(',')  # Ogni diagnosi è separata da una virgola, quindi splitto la linea in base a questo carattere
                    diagnosis_data.append(diagnoses)

        except Exception as e:
            print(f"  Error reading {diagnosis_file}: {e}")
            return None
        
        return diagnosis_data if diagnosis_data else None
    
    # !!! NUOVA FUNZIONE PER CARICARE I DATI DI TUNING
    def _load_tuning_data(self) -> Optional[List[List[str]]]:
        """Load tuning data from tuning_symbolic_logs.txt file"""
        tuning_file = self.algorithm_logs_dir / "tuning_symbolic_logs.txt"
        if not tuning_file.exists():
            return None
        
        tuning_data = []
        try:
            with open(tuning_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    line = line.replace("[","").replace("]","").replace("'", "").replace(" ", "")    # Rimuovo i simboli superflui per ottenere solo i dati
                    tunings = line.split(',')  # Ogni soluzione di tuning è separata da una virgola, quindi splitto la linea in base a questo carattere
                    tuning_data.append(tunings)

        except Exception as e:
            print(f"  Error reading {tuning_file}: {e}")
            return None
        
        return tuning_data if tuning_data else None


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
        # prima di tutto controllo quali dati sono disponibili per decidere quanti grafici creare 
        num_plots = 0
        plotAccuracy = False
        plotScore = False
        plotParams = False

        for i in range(len(analyzer.results)):  # controllo se c'è almeno un dato di accuratezza valido per decidere se creare il grafico dell'accuratezza
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

        print(f"  - {num_plots} line plots to generate for {exp_dir.name}")
        
        if num_plots == 0:
            continue

        fig, axes = plt.subplots(1,num_plots, figsize=(10 * num_plots, 6))      # adatto la dimensione della figura al numero di grafici da creare
        # Grafici linea per l'accuratezza
        idx = 0
        if plotAccuracy:
            axes[idx].plot(
                [r.iteration for r in analyzer.results if r.accuracy is not None],  #asse x sono le iterazioni
                [r.accuracy for r in analyzer.results if r.accuracy is not None],   #asse y sono le accuratezze, filtro solo quelle non None
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
                [r.iteration for r in analyzer.results if r.score is not None and r.score<0],   #asse x sono le iterazioni
                [r.score for r in analyzer.results if r.score is not None and r.score<0],       #asse y sono gli score, filtro solo quelli non None e negativi 
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
                [r.iteration for r in analyzer.results if r.nparams is not None],       #asse x sono le iterazioni
                [r.nparams for r in analyzer.results if r.nparams is not None],         #asse y sono il numero di parametri, filtro solo quelli non None
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

        # Grafico a barre per i dati di evidence
        dati_evidence = []          # Lista per memorizzare i dati di evidence da graficare  
        for r in analyzer.results:
            if r.evidence is not None:
                dati_evidence.append(r.evidence)

        if dati_evidence:       # Se ci sono dati di evidence da graficare, genero il grafico a barre
            print (f"  - Generating evidence bar plot for {exp_dir.name} ")
            fig, ax = plt.subplots(figsize=(15, 8))
            df = pd.DataFrame(dati_evidence, columns=['Tupla', 'Condizione'])       # Creo un DataFrame con due colonne: una per la tupla (azione) e una per la condizione (successo o fallimento) per aiutarmi nel definire assi e nei calcoli

            df['Etichetta'] = df['Tupla'].apply(lambda x: "-".join(map(str, x)))    # Creo una nuova colonna "Etichetta" unendo i valori della tupla in una stringa, in modo da poterla usare come etichetta sull'asse x del grafico a barre

            conteggi = pd.crosstab(df['Etichetta'], df['Condizione'])           # Utilizzo crosstab per contare quante volte ogni etichetta ha avuto successo (True) e fallimento (False), ottenendo una tabella con le etichette come righe e le condizioni come colonne, con i conteggi corrispondenti.

            conteggi.plot(kind='bar', color=['red', 'green'], ax=ax) # Generiamo bar plot: Rosso per False, Verde per True

            ax.set_title('Conteggio True vs False per Categoria')
            ax.set_xlabel('Categorie (Tuple)')
            ax.set_ylabel('Frequenza')
            ax.set_xticklabels(conteggi.index, rotation=45, ha='right', rotation_mode='anchor') # Ruoto e allineo le etichette sull'asse x
            ax.legend(['Falso', 'Vero'])
            fig.tight_layout()
            fig.savefig(output_dir / f"{exp_dir.name}_evidence.png")
            plt.close(fig)

        # Grafico la frequenza dei metodi intrapresi (indipendentemente dal successo) per capire qual è il più usato
        print (f"  - Generating method frequency bar plot for {exp_dir.name} ")
        grafici_tuning = 0
        grafico_evidence = False
        grafico_t=False

        if any(r.tuning is not None for r in analyzer.results):
            grafici_tuning+=1
            grafico_t=True
        if any(r.evidence is not None for r in analyzer.results):
            grafici_tuning+=1
            grafico_evidence = True

        if grafici_tuning >0:
            fig, ax = plt.subplots(1,grafici_tuning, figsize=(10 * grafici_tuning, 6))
            if grafici_tuning == 1:
                ax = [ax]
            idx=0
            if grafico_evidence:
                df['Metodo'] = df['Tupla'].apply(lambda x: x[0])   # Creo una nuova colonna "Metodo" estraendo solo la prima parte della tupla, che rappresenta l'azione intrapresa, in modo da poterla usare come categoria per il grafico a barre
                frequenza_azioni = df['Metodo'].value_counts()     # Calcolo la frequenza di ogni metodo intrapresa utilizzando value_counts, ottenendo una serie con i metodi come indice e le frequenze come valori
                frequenza_azioni.plot(kind='bar', color='blue', ax=ax[idx])  # Genero un bar plot con le frequenze dei metodi, usando il colore blu
                ax[idx].set_title('Frequenza dei Metodi Intrapresi')
                ax[idx].set_xlabel('Metodo Intrapreso')
                ax[idx].set_ylabel('Frequenza')
                ax[idx].set_xticklabels(frequenza_azioni.index, rotation=45, ha='right', rotation_mode='anchor')
                idx+=1
            if grafico_t:
                tuning_data = []      # Lista per memorizzare le diagnosi trovate nei diagnosis_symbolic_logs
                for r in analyzer.results:
                    if r.tuning is not None:
                        tuning_data.extend(r.tuning)   # Aggiungo tutte le diagnosi trovate in questa iterazione alla lista generale

                df_tuning = pd.DataFrame(tuning_data, columns=['Tuning'])   # Creo un DataFrame con una colonna "Tuning" per i dati di tuning_symbolic_logs
                frequenza_tuning = df_tuning['Tuning'].value_counts()        # Calcolo la frequenza di ogni diagnosi utilizzando value_counts, ottenendo una serie con le diagnosi come indice e le frequenze come valori
                frequenza_tuning.plot(kind='bar', color='orange', ax=ax[idx])       # Genero un bar plot con le frequenze delle diagnosi, usando il colore arancione
                ax[idx].set_title('Frequenza dei Tuning (tuning_symbolic_logs)')
                ax[idx].set_xlabel('Tuning Applicato')
                ax[idx].set_ylabel('Frequenza')
                ax[idx].set_xticklabels(frequenza_tuning.index, rotation=45, ha='right', rotation_mode='anchor')

            fig.tight_layout()
            fig.savefig(output_dir / f"{exp_dir.name}_action_frequency.png")
            plt.close(fig)

        # Grafico le specifiche diagnosi sia da dati di evidence che da diagnosis_symbolic_logs
        print (f"  - Generating diagnosis frequency bar plot for {exp_dir.name} ")
        grafici_diagnosi = 0
        grafico_evidence = False
        grafico_d=0

        if any(r.diagnosis is not None for r in analyzer.results):
            grafici_diagnosi+=1
            grafico_d=True
        if any(r.evidence is not None for r in analyzer.results):
            grafici_diagnosi+=1
            grafico_evidence = True
        if grafici_diagnosi >0:
            
            fig, ax = plt.subplots(1,grafici_diagnosi, figsize=(10 * grafici_diagnosi, 6))
            if grafici_diagnosi == 1:
                ax = [ax]
            idx=0
            if grafico_evidence:
                df['Diagnosi'] = df['Tupla'].apply(lambda x: x[1])   # Creo una nuova colonna "Diagnosi" estraendo solo la seconda parte della tupla, che rappresenta la diagnosi, in modo da poterla usare come categoria per il grafico a barre
                frequenza_azioni = df['Diagnosi'].value_counts()     # Calcolo la frequenza di ogni diagnosi utilizzando value_counts, ottenendo una serie con le diagnosi come indice e le frequenze come valori
                frequenza_azioni.plot(kind='bar', color='blue', ax=ax[0])  # Genero un bar plot con le frequenze delle diagnosi, usando il colore blu
                ax[0].set_title('Frequenza delle Diagnosi')
                ax[0].set_xlabel('Diagnosi')
                ax[0].set_ylabel('Frequenza')
                ax[0].set_xticklabels(frequenza_azioni.index, rotation=45, ha='right', rotation_mode='anchor')
                idx=1

            # Se sono disponibili anche i dati di diagnosis_symbolic_logs, genero un secondo grafico a barre per mostrare la frequenza delle diagnosi trovate in questi log
            if grafico_d:   # Controllo se ci sono dati di diagnosis_symbolic_logs disponibili in almeno un'iterazione
                diagnosis_data = []      # Lista per memorizzare le diagnosi trovate nei diagnosis_symbolic_logs
                for r in analyzer.results:
                    if r.diagnosis is not None:
                        diagnosis_data.extend(r.diagnosis)   # Aggiungo tutte le diagnosi trovate in questa iterazione alla lista generale

                df_diagnosis = pd.DataFrame(diagnosis_data, columns=['Diagnosi'])   # Creo un DataFrame con una colonna "Diagnosi" per i dati di diagnosis_symbolic_logs
                frequenza_diagnosi = df_diagnosis['Diagnosi'].value_counts()        # Calcolo la frequenza di ogni diagnosi utilizzando value_counts, ottenendo una serie con le diagnosi come indice e le frequenze come valori
                frequenza_diagnosi.plot(kind='bar', color='orange', ax=ax[idx])       # Genero un bar plot con le frequenze delle diagnosi, usando il colore arancione
                ax[idx].set_title('Frequenza delle Diagnosi (diagnosis_symbolic_logs)')
                ax[idx].set_xlabel('Diagnosi')
                ax[idx].set_ylabel('Frequenza')
                ax[idx].set_xticklabels(frequenza_diagnosi.index, rotation=45, ha='right', rotation_mode='anchor')

            fig.tight_layout()
            fig.savefig(output_dir / f"{exp_dir.name}_diagnosis_frequency.png")
            plt.close(fig)

            # Per ogni iterazione, mostro quali diagnosi sono state trovate e quali soluzioni di tuning sono state applicate, creando un grafico con le iterazioni sull'asse x e le diagnosi/soluzioni sull'asse y
            print (f"  - Generating diagnosis and tuning timeline plot for {exp_dir.name} ")
            fig, ax = plt.subplots(figsize=(15, 8))
            timeline_data = []   # Lista per memorizzare i dati da graficare nella timeline
            for r in analyzer.results:
                if r.diagnosis is not None:
                    for diag in r.diagnosis:
                        timeline_data.append((r.iteration, diag, 'diagnosis'))   # Aggiungo una tupla con iterazione, diagnosi e tipo "diagnosis" alla lista della timeline
                if r.tuning is not None:
                    for tune in r.tuning:
                        timeline_data.append((r.iteration, tune, 'tuning'))     # Aggiungo una tupla con iterazione, soluzione di tuning e tipo "tuning" alla lista della timeline
            if timeline_data:
                df_timeline = pd.DataFrame(timeline_data, columns=['Iteration', 'Evento', 'Tipo'])   # Creo un DataFrame con colonne per iterazione, evento (diagnosi o tuning) e tipo (diagnosis o tuning)
                df_timeline['Color'] = df_timeline['Tipo'].apply(lambda x: 'blue' if x == 'diagnosis' else 'orange')  # Creo una colonna "Color" per distinguere visivamente diagnosi e tuning nel grafico
                ax.scatter([], [], color='blue', label='Diagnosis')   # Aggiungo un punto fittizio per la legenda delle diagnosi
                ax.scatter([], [], color='orange', label='Tuning')    # Aggiungo un punto fittizio per la legenda delle soluzioni di tuning
                ax.legend()
                for _, row in df_timeline.iterrows():
                    ax.scatter(row['Iteration'], row['Evento'], color=row['Color'])   # Aggiungo un punto al grafico per ogni evento, usando il colore corrispondente al tipo
                ax.set_title('Timeline di Diagnosi e Tuning')
                ax.set_xlabel('Iterazione')
                ax.set_ylabel('Evento')
                ax.grid()
                fig.tight_layout()
                fig.savefig(output_dir / f"{exp_dir.name}_diagnosis_tuning_timeline.png")
                plt.close(fig)

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
