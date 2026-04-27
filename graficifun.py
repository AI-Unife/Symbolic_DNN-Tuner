import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


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
            return False
        
        # Load accuracies
        accuracies = self._load_accuracies()
        if not accuracies:
            return False
        
        scores = self._load_scores()
        
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
            self.config = {}
            return
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f) or {}
        except Exception as e:
            self.config = {}

    def _load_accuracies(self) -> List[Optional[float]]:
        """Load accuracies from acc_report.txt file"""
        acc_file = os.path.join(self.algorithm_logs_dir, "acc_report.txt")
        if not os.path.exists(acc_file):
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
            return []
        
        return accuracies
    
    def _load_scores(self) -> List[Optional[float]]:
        """Load scores from score_report.txt file"""
        score_file = os.path.join(self.algorithm_logs_dir, "score_report.txt")
        if not os.path.exists(score_file):
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
            return None
        
        return flops_data if flops_data else None
    
    def _load_hardware_data(self) -> Optional[List[Tuple[float, float, float, str]]]:
        """Load hardware data from hardware_report.txt file"""
        hw_file = os.path.join(self.algorithm_logs_dir, "hardware_report.txt")
        if not os.path.exists(hw_file):
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
            return None
        
        return hw_data if hw_data else None
    
    # !!! NUOVA FUNZIONE PER CARICARE I DATI DI EVIDENCE
    def _load_evidence_data(self) -> Optional[List[Tuple[Tuple[str, str], bool]]]:
        """Load evidence data from evidence.txt file"""
        evidence_file = os.path.join(self.algorithm_logs_dir, "evidence.txt")
        if not os.path.exists(evidence_file):
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
            return None
        
        return evidence_data if evidence_data else None
    
    # !!! NUOVA FUNZIONE PER CARICARE I DATI DI DIAGNOSIS
    def _load_diagnosis_data(self) -> Optional[List[List[str]]]:
        """Load diagnosis data from diagnosis_symbolic_logs.txt file"""
        diagnosis_file = os.path.join(self.algorithm_logs_dir, "diagnosis_symbolic_logs.txt")
        if not os.path.exists(diagnosis_file):
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
            return None
        
        return diagnosis_data if diagnosis_data else None
    
    # !!! NUOVA FUNZIONE PER CARICARE I DATI DI TUNING
    def _load_tuning_data(self) -> Optional[List[List[str]]]:
        """Load tuning data from tuning_symbolic_logs.txt file"""
        tuning_file = os.path.join(self.algorithm_logs_dir, "tuning_symbolic_logs.txt")
        if not os.path.exists(tuning_file):
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

    #Determinare i titoli dei grafici da creare in base ai dati disponibili
    cols = sum([plotAccuracy, plotScore, plotParams])
    titles = []
    if plotAccuracy: titles.append("Accuracy")
    if plotScore: titles.append("Score")
    if plotParams: titles.append("Number of Parameters")

    # Creo la figura con subplot per i grafici di confronto usando Plotly
    fig = make_subplots(rows=1, cols=cols, subplot_titles=titles)
    colors = px.colors.qualitative.Plotly  # Palette standard (10 colori distinti)
    
    for i, analyzer in enumerate(analyzers):
        exp_name = analyzer.experiment_dir.name
        exp_color = colors[i % len(colors)] 
        idx = 1 
        # --- LOGICA ACCURACY ---
        if plotAccuracy:
            x_acc = [r.iteration for r in analyzer.results if r.accuracy is not None]
            y_acc = [r.accuracy for r in analyzer.results if r.accuracy is not None]
            fig.add_trace(go.Scatter(x=x_acc, y=y_acc, name=exp_name, mode='lines',line=dict(color=exp_color),
                                     legendgroup=exp_name, showlegend=True), row=1, col=idx)
            idx = idx +1  # Incremento l'indice solo se ho aggiunto il grafico dell'accuratezza

        # --- LOGICA SCORE  ---
        if plotScore:
            x_score = [r.iteration for r in analyzer.results if r.score is not None and r.score<0]
            y_score = [r.score for r in analyzer.results if r.score is not None and r.score<0]
            fig.add_trace(go.Scatter(x=x_score, y=y_score, name=exp_name, mode='lines',line=dict(color=exp_color),
                                     legendgroup=exp_name, showlegend=False), row=1, col=idx)
            idx += 1  # Incremento l'indice solo se ho aggiunto il grafico dell'accuratezza

        # --- LOGICA NUMBER OF PARAMETERS  ---
        if plotParams:
            x_par = [r.iteration for r in analyzer.results if r.nparams is not None]
            y_par = [r.nparams for r in analyzer.results if r.nparams is not None]
            fig.add_trace(go.Scatter(x=x_par, y=y_par, name=exp_name, mode='lines',line=dict(color=exp_color),
                                     legendgroup=exp_name, showlegend=False), row=1, col=idx)

    # Personalizzazione interattiva
    fig.update_layout(
        height=400, 
        width= 600 * cols,
        hovermode="x unified", # Mostra tutti i dati vicini al mouse sulla stessa X
        template="plotly_white",
        autosize=False
    )
    if output_dir:
        output_path = Path(output_dir) 
        output_path=os.path.join(output_path, f"Comparison_Accuracy_Score_Params.png")
        fig.write_image(output_path)
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
    
    cols = sum([plotAccuracy, plotScore, plotParams])   # Calcolo il numero di grafici da creare in base ai dati disponibili
    titles = []                                         #Determinare i titoli dei grafici da creare in base ai dati disponibili
    if plotAccuracy: titles.append("Accuracy")
    if plotScore: titles.append("Score")
    if plotParams: titles.append("Number of Parameters")

    # Creo la figura con subplot per i grafici di confronto usando Plotly
    fig = make_subplots(rows=1, cols=cols, subplot_titles=titles)
    
    idx = 1 
    # --- LOGICA ACCURACY ---
    if plotAccuracy:
        data_acc = [(r.iteration, r.accuracy, i) for i, r in enumerate(analyzer.results) if r.accuracy is not None]
        x_acc, y_acc, custom_acc = zip(*data_acc)
        fig.add_trace(go.Scatter(x=x_acc, y=y_acc,customdata=custom_acc,name="accuracy", mode='lines+markers',line=dict(color='red'),showlegend=False), row=1, col=idx)
        idx = idx +1  # Incremento l'indice solo se ho aggiunto il grafico dell'accuratezza

    # --- LOGICA SCORE  ---
    if plotScore:
        x_score = [r.iteration for r in analyzer.results if r.score is not None and r.score<0]
        y_score = [r.score for r in analyzer.results if r.score is not None and r.score<0]
        fig.add_trace(go.Scatter(x=x_score, y=y_score,name="score", mode='lines+markers',line=dict(color='blue'),showlegend=False), row=1, col=idx)
        idx += 1  # Incremento l'indice solo se ho aggiunto il grafico dell'accuratezza

    # --- LOGICA NUMBER OF PARAMETERS  ---
    if plotParams:
        x_par = [r.iteration for r in analyzer.results if r.nparams is not None]
        y_par = [r.nparams for r in analyzer.results if r.nparams is not None]
        fig.add_trace(go.Scatter(x=x_par, y=y_par,name="nparam", mode='lines+markers',line=dict(color='green'),showlegend=False), row=1, col=idx)

    # Personalizzazione interattiva
    fig.update_layout(
        height=400, 
        width=600 * cols,
        hovermode="closest", 
        clickmode="event+select",  # Abilita la modalità di click per selezionare i punti dati
        template="plotly_white",
        autosize=False
    )
    if output_dir:
        output_path = Path(output_dir) 
        output_path=os.path.join(output_path, f"{exp.name}_Accuracy_Score_Params.png")
        fig.write_image(output_path)
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

        fig = go.Figure()

        # X sarà un indice numerico per posizionare bene le barre
        x_indices = list(range(len(pivot_df)))

        # Aggiungiamo una traccia per ogni esperimento (Colonna del pivot)
        for esperimento in pivot_df.columns:
            fig.add_trace(go.Bar(
                name=esperimento,
                x=x_indices,
                y=pivot_df[esperimento].values,
                hovertext=[f"Azione: {a}<br>Esito: {'S' if e else 'F'}" for a, e in pivot_df.index]
            ))

        # Creiamo le etichette per le azioni (una ogni due barre)
        tick_vals = [i * 2 + 0.5 for i in range(len(tutte_azioni))]
        tick_text = tutte_azioni

        # Configurazione Layout
        fig.update_layout(
            barmode='stack',  # Mantiene gli esperimenti uno sopra l'altro
            title='Evidence Distribution: Failures (F) vs Successes (S)',
            xaxis=dict(
                tickvals=tick_vals,
                ticktext=tick_text,
                tickangle=-45,  # Etichette AZIONI in obliquo
                title="Actions",
                automargin=True
            ),
            yaxis=dict(title='Frequency'),
            legend_title="Experiments",
            template="plotly_white",
            height=700,
            width=1800,
            margin=dict(l=100, r=50, b=150, t=80),
            autosize=False 
        )

        for i, (azione, esito) in enumerate(pivot_df.index):
            fig.add_annotation(
                x=i,
                y=0,
                text="S" if esito else "F",
                showarrow=False,
                yshift=-15,      # Posiziona la lettera appena sotto l'asse
                font=dict(size=10, color="gray")
            )

        # Linee di separazione tra le azioni
        for i in range(len(tutte_azioni)):
            fig.add_vline(x=(i * 2) + 1.5, line_width=1, line_dash="dash", line_color="gray", opacity=0.2)

        
        if output_dir:
            output_path = Path(output_dir)
            output_path = os.path.join(output_path, f"Comparison_Evidence.png")
            fig.write_image(output_path)
        return fig
        
def plotevidence(analyzer: ResultsAnalyzer, exp: Path, output_dir: Optional[Path] = None):
        # Grafico a barre per i dati di evidence
        dati_evidence = []          # Lista per memorizzare i dati di evidence da graficare  
        for r in analyzer.results:
            if r.evidence is not None:
                dati_evidence.append(r.evidence)

        if dati_evidence:       # Se ci sono dati di evidence da graficare, genero il grafico a barre
            df = pd.DataFrame(dati_evidence, columns=['Tupla', 'Condizione'])       # Creo un DataFrame con due colonne: 'Tupla' per l'azione e 'Condizione' per il successo/fallimento
            df['Etichetta'] = df['Tupla'].apply(lambda x: "-".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x))    #Creo la colonna 'Etichetta' che contiene in stringa la tupla di azione

            conteggi = pd.crosstab(df['Etichetta'], df['Condizione'])   # Creo una tabella di contingenza che conta quante volte ogni combinazione di azione (Etichetta) e condizione (successo/fallimento) si è verificata

            cols_mappa = {False: None, True: None} #creo un dizionario con vero e falso così posso graficare indipendentemente le condizioni
            for c in conteggi.columns:
                if c in [False, 0, "False", "0"]:   #se la colonna rappresenta i fallimenti, la mappo a False. Non ho idea di come sia stato interpretato il dato
                    cols_mappa[False] = c
                elif c in [True, 1, "True", "1"]:
                    cols_mappa[True] = c

            fig = go.Figure()

            # Aggiunta traccia Fallimenti (Rosso) - solo se esistono dati
            if cols_mappa[False] is not None:
                fig.add_trace(go.Bar(
                    x=conteggi.index.tolist(),
                    y=conteggi[cols_mappa[False]].values,
                    name='Failure',
                    marker_color='red'
                ))

            # Aggiunta traccia Successi (Verde) - solo se esistono dati
            if cols_mappa[True] is not None:
                fig.add_trace(go.Bar(
                    x=conteggi.index.tolist(),
                    y=conteggi[cols_mappa[True]].values,
                    name='Success',
                    marker_color='green'
                ))

            # Configurazione Layout "Anti-Taglio"
            fig.update_layout(
                title=f'Evidence Distribution: Failures (red) vs Successes (green) - {exp.name if exp else ""}',
                xaxis_title='Action',
                yaxis_title='Frequency',
                barmode='group',
                template='plotly_white',
                xaxis=dict(
                    type='category',
                    tickmode='array',
                    tickvals=list(range(len(conteggi.index))),
                    ticktext=conteggi.index.tolist(),
                    tickangle=-45,          # Inclinazione per leggibilità
                    automargin=True,       # Risolve il problema del testo tagliato
                ),
                # Margine sinistro (l) aumentato per evitare che la prima etichetta sparisca
                margin=dict(l=120, r=50, b=150, t=80),
                height=700,
                width=1600,
                autosize=False
            )

            if output_dir:
                output_path = Path(output_dir)
                output_path = os.path.join(output_path, f"{exp.name}_Evidence.png")
                fig.write_image(output_path)
            return fig

def plottuning_confronto(analyzers: List[ResultsAnalyzer], output_dir: Optional[Path] = None):
    crea_grafico= False
    for analyzer in analyzers:
        if any(r.tuning is not None for r in analyzer.results):
            crea_grafico = True

    if not crea_grafico:
        return None  # Se non ci sono dati di tuning da graficare, esco dalla funzione senza creare alcun grafico
    
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

    fig=go.Figure()
    for esperimento in df_tuning.columns:
        fig.add_trace(go.Bar(name=esperimento, x=df_tuning.index.tolist(), y=df_tuning[esperimento].values, hovertext=[f"Tuning: {t}" for t in df_tuning.index]))

    fig.update_layout(
        title='Frequency of Tuning (tuning_symbolic_logs) by Experiment',
        xaxis_title='Tuning',
        yaxis_title='Frequency',
        barmode='stack',
        template='plotly_white',
        xaxis=dict(
            type='category',
            tickmode='array',
            tickvals=list(range(len(df_tuning.index))),     # Posizioni numeriche per le etichette
            ticktext=df_tuning.index.tolist(),
            tickangle=-45,          # Inclinazione per leggibilità
            automargin=True,       # Risolve il problema del testo tagliato
        ),
        margin=dict(l=120, r=50, b=150, t=80),
        height=700,
        width=1800,
        autosize=False
    )
    if output_dir:
        output_path = Path(output_dir)
        output_path = os.path.join(output_path, f"Comparison_Tuning.png")
        fig.write_image(output_path)
    return fig

def plottuning(analyzer, exp, output_dir=None):
    # Prima di tutto controllo se ci sono dati di tuning da graficare, se non ci sono esco dalla funzione senza creare alcun grafico
    grafico_t = any(r.tuning is not None for r in analyzer.results)
    grafico_evidence = any(r.evidence is not None for r in analyzer.results)
    num_subplot = sum([grafico_t, grafico_evidence])

    if num_subplot == 0:
        return None

    # Creazione della figura con subplot
    fig = make_subplots(
        rows=1, cols=num_subplot, 
        subplot_titles=((['Frequency of Tuning form file evidence'] if grafico_evidence else []) + (['Frequency of Tuning from file tuning'] if grafico_t else []))
    )

    idx = 1 # Parto a fare il primo grafico

    # Logica da file EVIDENCE (Grafico Blu)
    if grafico_evidence:
        dati_evidence = [r.evidence for r in analyzer.results if r.evidence is not None]        #estraggo dati
        df = pd.DataFrame(dati_evidence, columns=['Tupla', 'Condizione'])
        # Estraiamo l'azione dalla tupla (primo elemento)
        df['Metodo'] = df['Tupla'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) else x)
        frequenza_azioni = df['Metodo'].value_counts()

        fig.add_trace(go.Bar( x=frequenza_azioni.index.tolist(),y=frequenza_azioni.values,marker_color='blue',name='Tuning',showlegend=False),row=1, col=idx)
        idx += 1

    # Logica del file TUNING (Grafico Arancione)
    if grafico_t:
        tuning_data = []                    #estraggo dati
        for r in analyzer.results:
            if r.tuning is not None:
                tuning_data.extend(r.tuning)
        
        df_tuning = pd.DataFrame(tuning_data, columns=['Tuning'])
        frequenza_tuning = df_tuning['Tuning'].value_counts()

        fig.add_trace(go.Bar(x=frequenza_tuning.index.tolist(),y=frequenza_tuning.values,marker_color='orange',name='Tuning',showlegend=False),row=1, col=idx)

    #Configurazione Layout e Rifiniture "Anti-Taglio"
    fig.update_layout(
        template='plotly_white',
        height=700,
        width=800 * num_subplot,
        margin=dict(l=100, r=50, b=150, t=100), # Margini abbondanti per etichette oblique
        autosize=False
    )

    # Applichiamo la rotazione e l'allineamento a tutti gli assi X
    fig.update_xaxes(
        tickangle=-45, 
        automargin=True,
        tickfont=dict(size=12)
    )
    
    fig.update_yaxes(title_text="Frequency")

    # Salvataggio
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(output_path / f"{exp.name}_tuning.png"))
    return fig


def plotdiagnosis_confronto(analyzers: List[ResultsAnalyzer], output_dir: Optional[Path] = None):
    crea_grafico= False
    for analyzer in analyzers:
        if any(r.diagnosis is not None for r in analyzer.results):
            crea_grafico = True

    if not crea_grafico:
        return None  # Se non ci sono dati di tuning da graficare, esco dalla funzione senza creare alcun grafico
    
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
    fig=go.Figure()
    for esperimento in df_diagnosis.columns:
        fig.add_trace(go.Bar(
            name=esperimento,
            x=df_diagnosis.index.tolist(),
            y=df_diagnosis[esperimento].values,
            hovertext=[f"Diagnosis: {d}" for d in df_diagnosis.index]
        ))
    fig.update_layout(
        title='Frequency of Diagnoses (diagnosis_symbolic_logs) by Experiment',
        xaxis_title='Diagnosis',
        yaxis_title='Frequency',
        barmode='stack',
        template='plotly_white',
        xaxis=dict(
            type='category',
            tickmode='array',
            tickvals=list(range(len(df_diagnosis.index))),
            ticktext=df_diagnosis.index.tolist(),
            tickangle=-45,          # Inclinazione per leggibilità
            automargin=True,       # Risolve il problema del testo tagliato
        ),
        margin=dict(l=120, r=50, b=150, t=80),
        height=700,
        width=1800,
        autosize=False
    )
    if output_dir:
        output_path = Path(output_dir)
        output_path = os.path.join(output_path, f"Comparison_Diagnosis.png")
        fig.write_image(output_path)
    return fig

def plotdiagnosis(analyzer, exp, output_dir=None):
    # Analisi preliminare dei dati
    grafico_t = any(r.diagnosis is not None for r in analyzer.results)
    grafico_evidence = any(r.evidence is not None for r in analyzer.results)
    num_subplot = sum([grafico_t, grafico_evidence])

    if num_subplot == 0:
        return None

    # Creazione della figura con sottoplot
    fig = make_subplots(
        rows=1, cols=num_subplot, 
        subplot_titles=((['Frequency of Diagnoses from file evidence'] if grafico_evidence else []) + (['Frequency of Diagnoses from file diagnosis'] if grafico_t else []))
    )

    idx = 1 # Indice colonna Plotly (parte da 1)

    # Logica EVIDENCE (Grafico Blu)
    if grafico_evidence:
        dati_evidence = [r.evidence for r in analyzer.results if r.evidence is not None]
        df = pd.DataFrame(dati_evidence, columns=['Tupla', 'Condizione'])
        # Estraiamo la diagnosi dalla tupla (secondo elemento)
        df['Diagnosi'] = df['Tupla'].apply(lambda x: x[1] if isinstance(x, (list, tuple)) else x)
        frequenza_azioni = df['Diagnosi'].value_counts()

        fig.add_trace(go.Bar(x=frequenza_azioni.index.tolist(),y=frequenza_azioni.values,marker_color='blue',name='Diagnosis',showlegend=False),row=1, col=idx)
        idx += 1

    # Logica TUNING (Grafico Arancione)
    if grafico_t:
        diagnosis_data = []
        for r in analyzer.results:
            if r.diagnosis is not None:
                diagnosis_data.extend(r.diagnosis)
        
        df_diagnosis = pd.DataFrame(diagnosis_data, columns=['Diagnosi'])
        frequenza_diagnosi = df_diagnosis['Diagnosi'].value_counts()

        fig.add_trace(go.Bar(x=frequenza_diagnosi.index.tolist(),y=frequenza_diagnosi.values,marker_color='orange',name='Diagnosis', showlegend=False),row=1, col=idx)

    #Configurazione Layout e Rifiniture "Anti-Taglio"
    fig.update_layout(
        template='plotly_white',
        height=700,
        width=800 * num_subplot,
        margin=dict(l=100, r=50, b=150, t=100), # Margini abbondanti per etichette oblique
        autosize=False
    )

    # Applichiamo la rotazione e l'allineamento a tutti gli assi X
    fig.update_xaxes(
        tickangle=-45, 
        automargin=True,
        tickfont=dict(size=12)
    )
    
    fig.update_yaxes(title_text="Frequency")

    # Salvataggio
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(output_path / f"{exp.name}_diagnosis.png"))
    return fig

def plottimeline_confronto(analyzers, output_dir=None):
    timeline_data = []
    
    for analyzer in analyzers:
        # Recuperiamo il nome dell'esperimento se disponibile per il dettaglio al passaggio del mouse
        exp_name = getattr(analyzer, 'name', 'Unknown')
        for r in analyzer.results:
            if r.diagnosis is not None:
                for diag in r.diagnosis:
                    timeline_data.append({'Iteration': r.iteration, 'Evento': diag, 'Tipo': 'Diagnosis', 'Exp': exp_name})
            if r.tuning is not None:
                for tune in r.tuning:
                    timeline_data.append({'Iteration': r.iteration, 'Evento': tune, 'Tipo': 'Tuning', 'Exp': exp_name})

    if not timeline_data:
        return None

    df = pd.DataFrame(timeline_data)
    # Raggruppiamo per contare la frequenza (dimensione del punto)
    df_counts = df.groupby(['Iteration', 'Evento', 'Tipo']).size().reset_index(name='Frequenza')

    fig = go.Figure()

    # Traccia per DIAGNOSIS (Scala di Blu)
    diag_df = df_counts[df_counts['Tipo'] == 'Diagnosis']
    if not diag_df.empty:
        fig.add_trace(go.Scatter(
            x=diag_df['Iteration'],
            y=diag_df['Evento'],
            mode='markers',
            name='Diagnosis',
            marker=dict(
                size= 10,
                color=diag_df['Frequenza'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Diag Freq", x=1.0),
                line=dict(width=1, color='black')
            ),
            hovertemplate="<b>%{y}</b><br>Iteration: %{x}<br>Frequency: %{marker.color}<extra></extra>",showlegend=False
        ))

    # Traccia per TUNING (Scala di Rossi)
    tune_df = df_counts[df_counts['Tipo'] == 'Tuning']
    if not tune_df.empty:
        fig.add_trace(go.Scatter(
            x=tune_df['Iteration'],
            y=tune_df['Evento'],
            mode='markers',
            name='Tuning',
            marker=dict(
                size= 10,
                color=tune_df['Frequenza'],
                colorscale='Reds',
                showscale=True,
                colorbar=dict(title="Tune Freq", x=1.15), # Spostata a destra per non sovrapporsi alla prima
                line=dict(width=1, color='black')
            ),
            hovertemplate="<b>%{y}</b><br>Iteration: %{x}<br>Frequency: %{marker.color}<extra></extra>",showlegend=False
        ))

    # Layout e Rifiniture
    fig.update_layout(
        title='Timeline of Diagnoses and Tuning',
        xaxis=dict(title='Iteration', gridcolor='lightgray'),
        yaxis=dict(title='Event', gridcolor='lightgray', autorange="reversed"), # Reversed per leggere dall'alto
        template='plotly_white',
        height=700,
        width=1800,
        margin=dict(l=150, r=200, b=100, t=100), # Spazio per le colorbar a destra
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        autosize=False
    )

    # Salvataggio
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(output_path / "Comparison_timeline.png"))

    return fig

def plottimeline(analyzer, exp, output_dir=None):
    timeline_data_diagnosis = []
    timeline_data_tuning = []

    # Raccolta dati: distinguo tra diagnosi e tuning per poterli graficare con colori diversi
    for r in analyzer.results:
        if r.diagnosis is not None:
            for diag in r.diagnosis:
                timeline_data_diagnosis.append({'Iteration': r.iteration, 'Evento': diag})
        if r.tuning is not None:
            for tune in r.tuning:
                timeline_data_tuning.append({'Iteration': r.iteration, 'Evento': tune})

    if not timeline_data_diagnosis and not timeline_data_tuning:
        return None

    fig = go.Figure()

    # Traccio prima DIAGNOSIS (Blu)
    if timeline_data_diagnosis:
        df_diag = pd.DataFrame(timeline_data_diagnosis)

        fig.add_trace(go.Scatter(x=df_diag['Iteration'],y=df_diag['Evento'], mode='markers',name='Diagnosis',
            marker=dict(color='blue', size=12),hovertemplate="<b>Diagnosis</b><br>Iteration: %{x}<br>Evento: %{y}" ))

    # Traccio poi TUNING (Arancione)
    if timeline_data_tuning:
        df_tune = pd.DataFrame(timeline_data_tuning)

        fig.add_trace(go.Scatter(x=df_tune['Iteration'],y=df_tune['Evento'],mode='markers', name='Tuning',
            marker=dict(color='orange', size=10, symbol='diamond'),hovertemplate="<b>Tuning</b><br>Iteration: %{x}<br>Evento: %{y}" ))

    #Configurazione Layout
    fig.update_layout(
        title=f'Timeline of Diagnoses and Tuning - {exp.name if exp else ""}',
        xaxis=dict(title='Iteration', gridcolor='lightgray'), # Griglia leggera per facilitare la lettura
        yaxis=dict(title='Event', gridcolor='lightgray', automargin=True), 
        template='plotly_white',
        height=700,
        width=1600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=150, r=50, b=80, t=100), # Ampio margine sinistro per i nomi degli eventi
        autosize=False
    )

    # Salvataggio
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(output_path / f"{exp.name}_timeline.png"))

    return fig

def analyze_all_experiments(parent_dir: Path, output_dir: Path = None):
    # Find all subfolders with algorithm_logs
    experiments = []
    parent_dir = Path(parent_dir)
    for item in parent_dir.iterdir():
        if item.is_dir() and os.path.exists(os.path.join(item, "algorithm_logs")):
            experiments.append(item)
    
    if not experiments:
        return None, None
        
    # Analyze each experiment
    lista_analizer = []
    
    for exp_dir in sorted(experiments):        
        analyzer = ResultsAnalyzer(exp_dir)
        if not analyzer.load_results():
            continue

        lista_analizer.append(analyzer)

    return experiments,lista_analizer