import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import pandas as pd

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
            experiment_dir: Experiment folder containing algorithm_logs
        """
        self.experiment_dir = Path(experiment_dir)
        self.algorithm_logs_dir = os.path.join(self.experiment_dir, "algorithm_logs")
        self.results: List[ExperimentResult] = []
        self.has_flops_module = False
        self.has_hardware_module = False
        self.config: Dict[str, Any] = {}  # Experiment configuration
        self.exp_name = self.experiment_dir.name  # Nome dell'esperimento basato sul nome della cartella

    def load_results(self) -> bool:
        """
        Load all results from log files.
        
        Returns:
            True if logs were loaded successfully, False otherwise
        """
        # Load config.yaml file
        self._load_config_yaml()
        
        if not os.path.exists(self.algorithm_logs_dir):
            print(f"!  algorithm_logs folder not found in {self.experiment_dir}",file=sys.stderr)
            return False
        
        # Load accuracies
        accuracies = self._load_accuracies()
        if not accuracies:
            print(f"!  No acc_report.txt file found",file=sys.stderr)
            return False
        
        scores = self._load_scores()
        if not scores:
            print(f"!  No score_report.txt file found, will use -accuracy as approximation",file=sys.stderr)
        
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
        config_file = os.path.join(self.experiment_dir, "config.yaml")
        if not os.path.exists(config_file):
            print(f"!  config.yaml file not found in {self.experiment_dir}",file=sys.stderr)
            self.config = {}
            return
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"!  Error reading config.yaml: {e}",file=sys.stderr)
            self.config = {}

    def _load_accuracies(self) -> List[Optional[float]]:
        """Load accuracies from acc_report.txt file"""
        acc_file = os.path.join(self.algorithm_logs_dir, "acc_report.txt")
        if not os.path.exists(acc_file):
            print(f"!  No acc_report.txt file found",file=sys.stderr)
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
            print(f"!  Error reading {acc_file}: {e}",file=sys.stderr)
            return []
        
        return accuracies
    
    def _load_scores(self) -> List[Optional[float]]:
        """Load scores from score_report.txt file"""
        score_file = os.path.join(self.algorithm_logs_dir, "score_report.txt")
        if not os.path.exists(score_file):
            print(f"!  No score_report.txt file found",file=sys.stderr)
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
            print(f"!  Error reading {score_file}: {e}",file=sys.stderr)
            return []
        
        return scores
    
    def _load_flops_data(self) -> Optional[List[Tuple[float, float]]]:
        """Load FLOPS data from flops_report.txt file"""
        flops_file = os.path.join(self.algorithm_logs_dir, "flops_report.txt")
        if not os.path.exists(flops_file):
            flops_file = os.path.join(self.algorithm_logs_dir, "params_report.txt")
            if not os.path.exists(flops_file):
                flops_file = os.path.join(self.experiment_dir, "flops_report.txt")
                if not os.path.exists(flops_file):
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
            print(f"!  Error reading {flops_file}: {e}",file=sys.stderr)
            return None
        
        return flops_data if flops_data else None
    
    def _load_hardware_data(self) -> Optional[List[Tuple[float, float, float, str]]]:
        """Load hardware data from hardware_report.txt file"""
        hw_file = os.path.join(self.algorithm_logs_dir, "hardware_report.txt")
        if not os.path.exists(hw_file):
            print(f"!  No hardware_report.txt file found",file=sys.stderr)
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
            print(f"!  Error reading {hw_file}: {e}",file=sys.stderr)
            return None
        
        return hw_data if hw_data else None
    
    # !!! NUOVA FUNZIONE PER CARICARE I DATI DI EVIDENCE
    def _load_evidence_data(self) -> Optional[List[Tuple[Tuple[str, str], bool]]]:
        """Load evidence data from evidence.txt file"""
        evidence_file = os.path.join(self.algorithm_logs_dir, "evidence.txt")
        if not os.path.exists(evidence_file):
            print(f"!  No evidence.txt file found",file=sys.stderr)
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
            print(f"!  Error reading {evidence_file}: {e}",file=sys.stderr)
            return None
        
        return evidence_data if evidence_data else None
    
    # !!! NUOVA FUNZIONE PER CARICARE I DATI DI DIAGNOSIS
    def _load_diagnosis_data(self) -> Optional[List[List[str]]]:
        """Load diagnosis data from diagnosis_symbolic_logs.txt file"""
        diagnosis_file = os.path.join(self.algorithm_logs_dir, "diagnosis_symbolic_logs.txt")
        if not os.path.exists(diagnosis_file):
            print(f"!  No diagnosis_symbolic_logs.txt file found",file=sys.stderr)
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
            print(f"!  Error reading {diagnosis_file}: {e}",file=sys.stderr)
            return None
        
        return diagnosis_data if diagnosis_data else None
    
    # !!! NUOVA FUNZIONE PER CARICARE I DATI DI TUNING
    def _load_tuning_data(self) -> Optional[List[List[str]]]:
        """Load tuning data from tuning_symbolic_logs.txt file"""
        tuning_file = os.path.join(self.algorithm_logs_dir, "tuning_symbolic_logs.txt")
        if not os.path.exists(tuning_file):
            print(f"!  No tuning_symbolic_logs.txt file found",file=sys.stderr)
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
            print(f"!  Error reading {tuning_file}: {e}",file=sys.stderr)
            return None
        
        return tuning_data if tuning_data else None
    
def plotaccuracy_confronto(analyzers: list[ResultsAnalyzer], output_dir: Optional[Path] = None):
        # creo 3 grafici di confronto nella stessa figura: accuracy, score e nparams (se disponibili)
        # prima di tutto controllo quali dati sono disponibili per decidere quanti grafici creare 
        num_plots = 0
        plotAccuracy = False
        plotScore = False
        plotParams = False

        for analyzer in analyzers:
            for i in range(len(analyzer.results)):  # controllo se c'è almeno un dato di accuratezza valido per decidere se creare il grafico dell'accuratezza
                if analyzer.results[i].accuracy is not None:
                    num_plots += 1
                    plotAccuracy = True
                    break
            if plotAccuracy:
                break

        for analyzer in analyzers:
            for i in range(len(analyzer.results)):
                if analyzer.results[i].score is not None and analyzer.results[i].score < 0:
                    num_plots += 1
                    plotScore = True
                    break
            if plotScore:
                break

        for analyzer in analyzers:
            for i in range(len(analyzer.results)):
                if analyzer.results[i].nparams is not None:
                    num_plots += 1
                    plotParams = True
                    break
            if plotParams:
                break
        
        if num_plots == 0:
            return None  # Se non ci sono dati da graficare, esco dalla funzione senza creare alcun grafico

        fig, axes = plt.subplots(1,num_plots, figsize=(10 * num_plots, 6),dpi=200)      # adatto la dimensione della figura al numero di grafici da creare

        cmap = plt.get_cmap('gist_rainbow') 
        num_lines = len(analyzers)

        # Grafici linea per l'accuratezza
        idx = 0
        if plotAccuracy:
            for i, analyzer in enumerate(analyzers):
                axes[idx].plot(
                    [r.iteration for r in analyzer.results if r.accuracy is not None],  #asse x sono le iterazioni
                    [r.accuracy for r in analyzer.results if r.accuracy is not None],   #asse y sono le accuratezze, filtro solo quelle non None
                    label=analyzer.experiment_dir.name, color=cmap(i/num_lines),linewidth=2)  
                axes[idx].set_title(f"Accuracy", fontsize=20)
                axes[idx].set_xlabel("Iteration", fontsize=16)
                axes[idx].set_ylabel("Accuracy", fontsize=16)
                axes[idx].grid()
                axes[idx].plot()
            idx += 1

        # Grafici linea per lo score
        if plotScore:
            for i, analyzer in enumerate(analyzers):
                axes[idx].plot(
                    [r.iteration for r in analyzer.results if r.score is not None and r.score<0],   #asse x sono le iterazioni
                    [r.score for r in analyzer.results if r.score is not None and r.score<0],       #asse y sono gli score, filtro solo quelli non None e negativi 
                    label=analyzer.experiment_dir.name, color=cmap(i/num_lines),linewidth=2)
                axes[idx].set_title(f"Score", fontsize=20)
                axes[idx].set_xlabel("Iteration", fontsize=16)
                axes[idx].set_ylabel("Score", fontsize=16)
                axes[idx].grid()
                axes[idx].plot()
            idx += 1

        # Grafici linea per il numero di parametri
        if plotParams:
            for i, analyzer in enumerate(analyzers):
                axes[idx].plot(
                    [r.iteration for r in analyzer.results if r.nparams is not None],       #asse x sono le iterazioni
                    [r.nparams for r in analyzer.results if r.nparams is not None],         #asse y sono il numero di parametri, filtro solo quelli non None
                    label=analyzer.experiment_dir.name, color=cmap(i/num_lines),linewidth=2)
                axes[idx].set_title(f"Number of Parameters", fontsize=20)
                axes[idx].set_xlabel("Iteration", fontsize=16)
                axes[idx].set_ylabel("Number of Parameters", fontsize=16)
                axes[idx].grid()
                axes[idx].plot()

        plt.legend(title="Experiments", bbox_to_anchor=(1, 1.1), loc='lower right',fontsize=14, title_fontsize=16)
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir)
            output_path=os.path.join(output_path, f"Comparison_Accuracy_Score_Params.png")
            plt.savefig(output_path, bbox_inches='tight')
        return fig

def plotaccuracy(analyzer: ResultsAnalyzer, exp: Path, output_dir: Optional[Path] = None):
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
        
        if num_plots == 0:
            return None  # Se non ci sono dati da graficare, esco dalla funzione senza creare alcun grafico

        fig, axes = plt.subplots(1,num_plots, figsize=(10 * num_plots, 6),dpi=200)      # adatto la dimensione della figura al numero di grafici da creare
        # Grafici linea per l'accuratezza
        idx = 0
        if plotAccuracy:
            axes[idx].plot(
                [r.iteration for r in analyzer.results if r.accuracy is not None],  #asse x sono le iterazioni
                [r.accuracy for r in analyzer.results if r.accuracy is not None],   #asse y sono le accuratezze, filtro solo quelle non None
                label=exp.name, color='red'
            )
            axes[idx].set_title(f"Accuracy - {exp.name}", fontsize=20)
            axes[idx].set_xlabel("Iteration", fontsize=16)
            axes[idx].set_ylabel("Accuracy", fontsize=16)
            axes[idx].grid()
            axes[idx].plot()
            idx += 1

        # Grafici linea per lo score
        if plotScore:
            axes[idx].plot(
                [r.iteration for r in analyzer.results if r.score is not None and r.score<0],   #asse x sono le iterazioni
                [r.score for r in analyzer.results if r.score is not None and r.score<0],       #asse y sono gli score, filtro solo quelli non None e negativi 
                label=exp.name, color='blue'
            )
            axes[idx].set_title(f"Score - {exp.name}", fontsize=20)
            axes[idx].set_xlabel("Iteration", fontsize=16)
            axes[idx].set_ylabel("Score", fontsize=16)
            axes[idx].grid()
            axes[idx].plot()
            idx += 1

        # Grafici linea per il numero di parametri
        if plotParams:
            axes[idx].plot(
                [r.iteration for r in analyzer.results if r.nparams is not None],       #asse x sono le iterazioni
                [r.nparams for r in analyzer.results if r.nparams is not None],         #asse y sono il numero di parametri, filtro solo quelli non None
                label=exp.name, color='green'
            )
            axes[idx].set_title(f"Number of Parameters - {exp.name}", fontsize=20)
            axes[idx].set_xlabel("Iteration", fontsize=16)
            axes[idx].set_ylabel("Number of Parameters", fontsize=16)
            axes[idx].grid()
            axes[idx].plot()

        plt.tight_layout()
        if output_dir:
            output_path = Path(output_dir) 
            output_path=os.path.join(output_path, f"{exp.name}_accuracy_score_params.png")
            plt.savefig(output_path)
        return fig

def plotevidence_confronto(analyzers, output_dir: Optional[Path] = None):
    dati_evidence = []    
    for analyzer in analyzers:
        exp_name = analyzer.experiment_dir.name 
        for r in analyzer.results:
            if r.evidence is not None:
                dati_evidence.append({
                    'Azione': "-".join(map(str, r.evidence[0])),
                    'Esito': r.evidence[1],
                    'Esperimento': exp_name
                })

    if dati_evidence:
        df = pd.DataFrame(dati_evidence)

        # Divido i dati in base azione, esito e esperimento, poi conto infine crea questa tabella (pivot) che per righe ha combinazione azione-esito per colonna ha esperimento e come valore il numero di volte che quella combinazione azione-esito è stata osservata in quell'esperimento
        pivot_df = df.groupby(['Azione', 'Esito', 'Esperimento']).size().unstack(fill_value=0)
        
        # mi assicuro che per ogni azione ci sia colonna per successo che fallimento
        tutte_azioni = df['Azione'].unique()
        nuovo_indice = pd.MultiIndex.from_product([tutte_azioni, [False, True]], names=['Azione', 'Esito'])     #creo prodotto cartesiano: per ogni azione deve esserci sia True che false 
        pivot_df = pivot_df.reindex(nuovo_indice, fill_value=0)

        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Disegniamo il grafico stacked, un bar plot ma su cui posso mettere una colonna sopra l'altra. 
        # Poiché l'indice è (Azione, Esito), Pandas metterà le barre di Fail e Success una accanto all'altra per ogni azione.
        pivot_df.plot(kind='bar', ax=ax, stacked=True, colormap='gist_rainbow')

        # Creo le etichette che andranno messe sull'asse x
        labels = []
        for i, (azione, esito) in enumerate(pivot_df.index):
            if  not esito: # Etichettiamo solo una delle due barre
                labels.append(azione)
            else:
                labels.append("")
        
        ax.set_xticklabels(labels, rotation=45, ha='right', rotation_mode='anchor',fontsize=12) #metto le etichette sull'asse x, ruotate di 45 gradi e allineate a destra per evitare sovrapposizioni
        ax.tick_params(axis='x', which='major', pad=15)             # abbasso un po' le etichette
        
        # Aggiungiamo una riga sottile per separare visivamente i gruppi di azioni
        for i in range(2, len(pivot_df), 2):
            ax.axvline(i - 0.5, color='gray', linestyle='--', alpha=0.2)

        ax.set_title('Evidence Distribution: Failures (F) vs Successes (S)', fontsize=18)
        ax.set_xlabel('Action', fontsize=14)
        ax.set_ylabel('Frequency', fontsize=14)
        
        # Legenda unica per gli esperimenti
        ax.legend(title="Experiments", bbox_to_anchor=(1, 1), loc='upper left')
        
        # Annotazione manuale per Success/Fail sotto le barre. Per ogni barra capisco se si riferisce ad unsuccesso o ad un fallimento
        sub_labels = [('F' if i%2==0 else 'S') for i in range(len(pivot_df))]
        for i, txt in enumerate(sub_labels):
            ax.text(i, -3, txt, ha='center', fontsize=12, alpha=0.7)

        fig.tight_layout()

        if output_dir:
            output_path = Path(output_dir) 
            output_path = os.path.join(output_path, f"Group_Evidence.png")
            plt.savefig(output_path, bbox_inches='tight')

        return fig
    
def plotevidence(analyzer: ResultsAnalyzer, exp: Path, output_dir: Optional[Path] = None):
        # Grafico a barre per i dati di evidence
        dati_evidence = []          # Lista per memorizzare i dati di evidence da graficare  
        for r in analyzer.results:
            if r.evidence is not None:
                dati_evidence.append(r.evidence)

        if dati_evidence:       # Se ci sono dati di evidence da graficare, genero il grafico a barre
            fig, ax = plt.subplots(figsize=(15, 8))
            df = pd.DataFrame(dati_evidence, columns=['Tupla', 'Condizione'])       # Creo un DataFrame con due colonne: una per la tupla (azione) e una per la condizione (successo o fallimento) per aiutarmi nel definire assi e nei calcoli

            df['Etichetta'] = df['Tupla'].apply(lambda x: "-".join(map(str, x)))    # Creo una nuova colonna "Etichetta" unendo i valori della tupla in una stringa, in modo da poterla usare come etichetta sull'asse x del grafico a barre

            conteggi = pd.crosstab(df['Etichetta'], df['Condizione'])           # Utilizzo crosstab per contare quante volte ogni etichetta ha avuto successo (True) e fallimento (False), ottenendo una tabella con le etichette come righe e le condizioni come colonne, con i conteggi corrispondenti.

            conteggi.plot(kind='bar', color=['red', 'green'], ax=ax) # Generiamo bar plot: Rosso per False, Verde per True

            ax.set_title('Evidence Distribution: Failures (red) vs Successes (green)', fontsize=18)
            ax.set_xlabel('Action', fontsize=14)
            ax.set_ylabel('Frequency', fontsize=14)
            ax.set_xticklabels(conteggi.index, rotation=45, ha='right', rotation_mode='anchor') # Ruoto e allineo le etichette sull'asse x
            ax.legend(['Failure', 'Success'])
            fig.tight_layout()
            if output_dir:
                output_path = Path(output_dir)
                output_path = os.path.join(output_path, f"{exp.name}_evidence.png")
                plt.savefig(output_path)
        return fig

def plottuning_confronto(analyzers: List[ResultsAnalyzer], output_dir: Optional[Path] = None):
    crea_grafico= False
    for analyzer in analyzers:
        if any(r.tuning is not None for r in analyzer.results):
            crea_grafico = True

    if not crea_grafico:
        return None  # Se non ci sono dati di tuning da graficare, esco dalla funzione senza creare alcun grafico

    # Grafico a barre per i dati di tuning
    fig, ax= plt.subplots(figsize=(15, 8))
    
    tuning_data = []      # Lista per memorizzare le diagnosi trovate nei diagnosis_symbolic_logs
    for analyzer in analyzers:
        exp_name = analyzer.experiment_dir.name 
        for r in analyzer.results:
            if r.tuning is not None:
                for tuning in r.tuning:
                    tuning_data.append({
                        'Tuning': tuning,
                        'Esperimento': exp_name
                    })

    df = pd.DataFrame(tuning_data)

    df_tuning = df.groupby(['Tuning', 'Esperimento']).size().unstack(fill_value=0)

    df_tuning.plot(kind='bar', ax=ax, stacked=True, colormap='gist_rainbow')
    ax.set_title('Frequency of Tuning (tuning_symbolic_logs)', fontsize=18)
    ax.set_xlabel('Action', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_xticklabels(df_tuning.index, rotation=45, ha='right', rotation_mode='anchor')

    ax.legend(title="Experiments", bbox_to_anchor=(1, 1), loc='upper left')

    fig.tight_layout()
    if output_dir:
        output_path = Path(output_dir)
        output_path = os.path.join(output_path, f"Comparison_Tuning.png")
        plt.savefig(output_path, bbox_inches='tight')
    return fig

def plottuning(analyzer: ResultsAnalyzer, exp: Path, output_dir: Optional[Path] = None):
    # Grafico la frequenza dei metodi intrapresi (indipendentemente dal successo) per capire qual è il più usato
    grafici_tuning = 0
    grafico_evidence = False
    grafico_t=False

    if any(r.tuning is not None for r in analyzer.results):
        grafici_tuning+=1
        grafico_t=True
    if any(r.evidence is not None for r in analyzer.results):
        grafici_tuning+=1
        grafico_evidence = True

    dati_evidence = []          # Lista per memorizzare i dati di evidence da graficare  
    for r in analyzer.results:
        if r.evidence is not None:
            dati_evidence.append(r.evidence)

    df = pd.DataFrame(dati_evidence, columns=['Tupla', 'Condizione'])

    if grafici_tuning >0:
        fig, ax = plt.subplots(1,grafici_tuning, figsize=(10 * grafici_tuning, 6))
        if grafici_tuning == 1:
            ax = [ax]
        idx=0
        if grafico_evidence:
            df['Metodo'] = df['Tupla'].apply(lambda x: x[0])   # Creo una nuova colonna "Metodo" estraendo solo la prima parte della tupla, che rappresenta l'azione intrapresa, in modo da poterla usare come categoria per il grafico a barre
            frequenza_azioni = df['Metodo'].value_counts()     # Calcolo la frequenza di ogni metodo intrapresa utilizzando value_counts, ottenendo una serie con i metodi come indice e le frequenze come valori
            frequenza_azioni.plot(kind='bar', color='blue', ax=ax[idx])  # Genero un bar plot con le frequenze dei metodi, usando il colore blu
            ax[idx].set_title('Frequency of Actions Taken', fontsize=18)
            ax[idx].set_xlabel('Applied Tuning', fontsize=14)
            ax[idx].set_ylabel('Frequency', fontsize=14)
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
            ax[idx].set_title('Frequency of Tuning (tuning_symbolic_logs)', fontsize=18)
            ax[idx].set_xlabel('Applied Tuning', fontsize=14)
            ax[idx].set_ylabel('Frequency', fontsize=14)
            ax[idx].set_xticklabels(frequenza_tuning.index, rotation=45, ha='right', rotation_mode='anchor')

        fig.tight_layout()
        if output_dir:
            output_path = Path(output_dir)
            output_path = os.path.join(output_path, f"{exp.name}_tuning.png")
            plt.savefig(output_path)
        return fig

def plotdiagnosis_confronto(analyzers: List[ResultsAnalyzer], output_dir: Optional[Path] = None):
    crea_grafico= False
    for analyzer in analyzers:
        if any(r.diagnosis is not None for r in analyzer.results):
            crea_grafico = True

    if not crea_grafico:
        return None  # Se non ci sono dati di tuning da graficare, esco dalla funzione senza creare alcun grafico

    # Grafico a barre per i dati di tuning
    fig, ax= plt.subplots(figsize=(15, 8))
    
    diagnosis_data = []      # Lista per memorizzare le diagnosi trovate nei diagnosis_symbolic_logs
    for analyzer in analyzers:
        exp_name = analyzer.experiment_dir.name 
        for r in analyzer.results:
            if r.diagnosis is not None:
                for diagnosis in r.diagnosis:
                    diagnosis_data.append({
                        'Diagnosi': diagnosis,
                        'Esperimento': exp_name
                    })

    df = pd.DataFrame(diagnosis_data)

    df_diagnosis = df.groupby(['Diagnosi', 'Esperimento']).size().unstack(fill_value=0)

    df_diagnosis.plot(kind='bar', ax=ax, stacked=True, colormap='gist_rainbow')
    ax.set_title('Frequency of Diagnoses', fontsize=18)
    ax.set_xlabel('Diagnosis', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_xticklabels(df_diagnosis.index, rotation=45, ha='right', rotation_mode='anchor')

    ax.legend(title="Experiments", bbox_to_anchor=(1, 1), loc='upper left')

    fig.tight_layout()
    if output_dir:
        output_path = Path(output_dir)
        output_path = os.path.join(output_path, f"Comparison_Diagnosis.png")
        plt.savefig(output_path, bbox_inches='tight')
    return fig

def plotdiagnosis(analyzer: ResultsAnalyzer, exp: Path, output_dir: Optional[Path] = None):
    # Grafico le specifiche diagnosi sia da dati di evidence che da diagnosis_symbolic_logs
    grafici_diagnosi = 0
    grafico_evidence = False
    grafico_d=False

    if any(r.diagnosis is not None for r in analyzer.results):
        grafici_diagnosi+=1
        grafico_d=True
    if any(r.evidence is not None for r in analyzer.results):
        grafici_diagnosi+=1
        grafico_evidence = True
        
    if grafici_diagnosi >0:

        dati_evidence = []          # Lista per memorizzare i dati di evidence da graficare  
        for r in analyzer.results:
            if r.evidence is not None:
                dati_evidence.append(r.evidence)

        df = pd.DataFrame(dati_evidence, columns=['Tupla', 'Condizione'])
        
        fig, ax = plt.subplots(1,grafici_diagnosi, figsize=(10 * grafici_diagnosi, 6))
        if grafici_diagnosi == 1:
            ax = [ax]
        idx=0
        if grafico_evidence:
            df['Diagnosi'] = df['Tupla'].apply(lambda x: x[1])   # Creo una nuova colonna "Diagnosi" estraendo solo la seconda parte della tupla, che rappresenta la diagnosi, in modo da poterla usare come categoria per il grafico a barre
            frequenza_azioni = df['Diagnosi'].value_counts()     # Calcolo la frequenza di ogni diagnosi utilizzando value_counts, ottenendo una serie con le diagnosi come indice e le frequenze come valori
            frequenza_azioni.plot(kind='bar', color='blue', ax=ax[0])  # Genero un bar plot con le frequenze delle diagnosi, usando il colore blu
            ax[0].set_title('Frequency of Diagnoses', fontsize=18)
            ax[0].set_xlabel('Diagnosis', fontsize=14)
            ax[0].set_ylabel('Frequency', fontsize=14)
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
            ax[idx].set_title('Frequency of Diagnoses (diagnosis_symbolic_logs)', fontsize=18)
            ax[idx].set_xlabel('Diagnosis', fontsize=14)
            ax[idx].set_ylabel('Frequency', fontsize=14)
            ax[idx].set_xticklabels(frequenza_diagnosi.index, rotation=45, ha='right', rotation_mode='anchor')

        fig.tight_layout()
        if output_dir:
            output_path = Path(output_dir) 
            output_path = os.path.join(output_path, f"{exp.name}_diagnosis.png")
            plt.savefig(output_path)
        return fig


def plottimeline_confronto(analyzers: List[ResultsAnalyzer], output_dir: Optional[Path] = None):
    timeline_data = []
    
    for analyzer in analyzers:
        for r in analyzer.results:
            # Raccogliamo Diagnosis
            if r.diagnosis is not None:
                for diag in r.diagnosis:
                    timeline_data.append({'Iteration': r.iteration, 'Evento': diag, 'Tipo': 'Diagnosis'})
            
            # Raccogliamo Tuning
            if r.tuning is not None:
                for tune in r.tuning:
                    timeline_data.append({'Iteration': r.iteration, 'Evento': tune, 'Tipo': 'Tuning'})

    if not timeline_data:
        return None

    df = pd.DataFrame(timeline_data)

    # Calcolola frequenza così da poterla usare quando uso la mappa di colore
    df_counts = df.groupby(['Iteration', 'Evento', 'Tipo']).size().reset_index(name='Frequenza')

    fig, ax = plt.subplots(figsize=(20, 10))

    # Dividiamo i dati per applicare due mappe di colore diverse
    diag_df = df_counts[df_counts['Tipo'] == 'Diagnosis']
    tune_df = df_counts[df_counts['Tipo'] == 'Tuning']

    # Disegno i punti
    if not diag_df.empty:
        sc_diag = ax.scatter(diag_df['Iteration'], diag_df['Evento'], 
                             c=diag_df['Frequenza'], cmap='Blues', 
                             s=150, edgecolors='black', linewidth=0.5, label='Diagnosis')
        plt.colorbar(sc_diag, ax=ax, label='Diagnosis Frequency', pad=0.01)

    if not tune_df.empty:
        sc_tune = ax.scatter(tune_df['Iteration'], tune_df['Evento'], 
                             c=tune_df['Frequenza'], cmap='Reds', 
                             s=150, edgecolors='black', linewidth=0.5, label='Tuning')
        plt.colorbar(sc_tune, ax=ax, label='Tuning Frequency', pad=0.08)

    ax.set_title('Timeline of Diagnoses and Tuning', fontsize=18)
    ax.set_xlabel('Iteration', fontsize=14)
    ax.set_ylabel('Event', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    
    if output_dir:
        output_path = Path(output_dir) 
        output_path = os.path.join(output_path, f"Group_timeline_heatmap.png")
        plt.savefig(output_path, bbox_inches='tight')
    return fig

def plottimeline(analyzer: ResultsAnalyzer, exp: Path, output_dir: Optional[Path] = None):
    # Per ogni iterazione, mostro quali diagnosi sono state trovate e quali soluzioni di tuning sono state applicate, creando un grafico con le iterazioni sull'asse x e le diagnosi/soluzioni sull'asse y
    #grafico prima le diagnosi e poi le soluzioni di tuning, usando colori diversi per distinguerle visivamente
    fig, ax = plt.subplots(figsize=(15, 8))
    timeline_data_diagnosis = []   # Lista per memorizzare i dati da graficare nella timeline diagnosis
    timeline_data_tuning = []   # Lista per memorizzare i dati da graficare nella timeline tuning
    for r in analyzer.results:
        if r.diagnosis is not None:
            for diag in r.diagnosis:
                timeline_data_diagnosis.append((r.iteration, diag))   # Aggiungo una tupla con iterazione, diagnosi 
        if r.tuning is not None:
            for tune in r.tuning:
                timeline_data_tuning.append((r.iteration, tune))     # Aggiungo una tupla con iterazione, soluzione di tuning 
    if timeline_data_diagnosis or timeline_data_tuning:   # Controllo se ci sono dati da graficare nella timeline
        df_diagnosis = pd.DataFrame(timeline_data_diagnosis, columns=['Iteration', 'Evento'])   # Creo un DataFrame con colonne per iterazione, evento diagnosi
        df_tuning = pd.DataFrame(timeline_data_tuning, columns=['Iteration', 'Evento'])   # Creo un DataFrame con colonne per iterazione, evento tuning
        df=df_diagnosis.copy()
        df = pd.concat([df, df_tuning], ignore_index=True)   # Unisco i due DataFrame in uno solo, in modo da poter graficare diagnosi e tuning nello stesso grafico
        df['Color'] = df['Evento'].apply(lambda x: 'blue' if x in [diag for r in analyzer.results for diag in (r.diagnosis or [])] else 'orange')  # Creo una colonna "Color" per distinguere visivamente diagnosi e tuning nel grafico
        ax.scatter([], [], color='blue', label='Diagnosis')   # Aggiungo un punto fittizio per la legenda delle diagnosi
        ax.scatter([], [], color='orange', label='Tuning')    # Aggiungo un punto fittizio per la legenda delle soluzioni di tuning
        ax.legend()
        for _, row in df.iterrows():
            ax.scatter(row['Iteration'], row['Evento'], color=row['Color'])   # Aggiungo un punto al grafico per ogni evento, usando il colore corrispondente al tipo
        ax.set_title('Timeline of Diagnoses and Tuning', fontsize=18)
        ax.set_xlabel('Iteration', fontsize=14)
        ax.set_ylabel('Event', fontsize=14)
        ax.grid()
        fig.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir)
            output_path=os.path.join(output_path, f"{exp.name}_timeline.png")
            plt.savefig(output_path)
        return fig

def analyze_all_experiments(parent_dir: Path, output_dir: Path = None):
    # Find all subfolders with algorithm_logs
    experiments = []
    parent_dir = Path(parent_dir)
    for item in parent_dir.iterdir():
        if item.is_dir() and os.path.exists(os.path.join(item, "algorithm_logs")):
            experiments.append(item)
    
    if not experiments:
        print(f"! No experiments found in {parent_dir}",file=sys.stderr)
        
    # Analyze each experiment
    lista_analizer = []
    
    for exp_dir in sorted(experiments):        
        analyzer = ResultsAnalyzer(exp_dir)
        if not analyzer.load_results():
            continue

        lista_analizer.append(analyzer)

    return experiments,lista_analizer