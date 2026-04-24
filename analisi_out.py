from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

class FileOUT:
    """Classe per determinare lo spazio di ricerca"""

    def __init__(self, exp_dir: Path):
        #setto le cartelle 
        self.exp_dir = Path(exp_dir)
        #exp_name è il nome dell'ultima cartella di exp_dir
        self.exp_name=self.exp_dir.name
        #creo una lista di dizionari vuoti per memorizzare i risultati
        self.spazio_ricerca = []
        self.trend = []
    
    def load_search_space(self) -> bool:
        """Carica lo spazio di ricerca da un file .out nella cartella dell'esperimento."""
        # cerco nella cartella dell'esperimento un file che termina con .out
        search_space_file = None
        for i in self.exp_dir.iterdir():
            if i.is_file() and i.suffix == ".out":
                search_space_file = i
                break

        if not search_space_file:
            print(f"⚠ Search space file not found: {search_space_file}")
            return False
        
        try:
            with open(search_space_file, "r") as f:
                #devo estrarre la giusta sequenza di spazi di ricerca. Può essere che uno venga calcolato e poi scartato perchè dopo l'inizio dell'iterazione si trovano problemi
                dict = {}
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if "--- ITERATION" in line:
                        #Prima di passare alla prossima iterazione, salvo lo spazio di ricerca che ho trovato nell'iterazione precedente 
                        if dict:
                            self.spazio_ricerca.append(dict)
                        # Estraggo l'iterazione corrente
                        line =line.split("--- ITERATION")[1].strip()
                        line = line.split(" ")[0].strip() # Prendo solo la parte numerica prima di eventuali spazi
                        iteration = int(line)
                        dict={"iteration": iteration}

                    elif line.startswith("Chosen point:"):
                        # Estraggo la parte dopo "Chosen point:"
                        space_str = line.split("Chosen point:")[1].strip()
                        # Converto la stringa in un dizionario
                        space_dict = eval(space_str)
                        dict.update(space_dict)

                # Dopo l'ultima iterazione, salvo lo spazio di ricerca trovato
                if dict:
                    self.spazio_ricerca.append(dict)
                
            return True
        except Exception as e:
            print(f"⚠ Error loading search space: {e}")
            return False
        
    def stampa_spazio_ricerca(self):
        """Stampa lo spazio di ricerca in modo leggibile"""
        print("\n🔍 Search Space:")

        for point in self.spazio_ricerca:
            iteration = point.get("iteration", "N/A")
            print(f"Iteration {iteration}:")
            for key, value in point.items():
                if key != "iteration":
                    print(f"  {key}: {value}")
            print("-" * 40)
    
    def grafici_spazio_ricerca(self, output_path: Optional[Path] = None):
        """Crea grafici per visualizzare lo spazio di ricerca"""
        if not self.spazio_ricerca:
            print("⚠ No search space data to plot.")
            return
        
        # Converti la lista di dizionari in un DataFrame
        df = pd.DataFrame(self.spazio_ricerca)

        # devo creare un grafico per ogni parametro di ricerca che mi dica come quella quantità stia cambiano con l'avanzare delle iterazioni tutte nella stessa finestra
        grafici_da_fare = df.shape[1]-1
        num_col=5
        fig, ax= plt.subplots(grafici_da_fare//num_col + (grafici_da_fare % num_col > 0), num_col, figsize=(14,7))      #voglio i grafici in file da 5 grafici per riga
        for column in df.columns:
            if column != "iteration":
                sns.lineplot(data=df, x="iteration", y=column, ax=ax[(df.columns.get_loc(column)-1)//num_col][(df.columns.get_loc(column)-1)%num_col])
                
        # Rimuovo eventuali assi vuoti
        for i in range(grafici_da_fare, (grafici_da_fare//num_col + (grafici_da_fare % num_col > 0)) * num_col):
            fig.delaxes(ax[i//num_col][i%num_col])

        plt.suptitle(f"Search Space Evolution Over Iterations - {self.exp_dir.name}")

        plt.tight_layout()
        if output_path:
            output_path = Path(output_path)
            output_path=os.path.join(output_path, f"{self.exp_dir.name}_search_space.png")
            plt.savefig(output_path)
        #plt.show()
        return fig
    
    def load_trend(self) -> bool:
        """Carica l'andamento delle iterazioni da un file .out nella cartella dell'esperimento."""
        # cerco nella cartella dell'esperimento un file che termina con .out
        search_space_file = None

        for i in self.exp_dir.iterdir():
            if i.is_file() and i.suffix == ".out":
                search_space_file = i
                break

        if not search_space_file:
            return False
        
        try:
            with open(search_space_file, "r") as f:
                dict = {}
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if "--- ITERATION" in line:
                        if dict:
                            self.trend.append(dict)
                        # Estraggo l'iterazione corrente
                        line =line.split("--- ITERATION")[1].strip()
                        line = line.split(" ")[0].strip() # Prendo solo la parte numerica prima di eventuali spazi
                        iteration = int(line)
                        dict={"iteration": iteration}

                    elif "Epoch" in line and "Loss" in line and "Accuracy" in line:
                        line=line.split()
                        loss=float(line[4][:-1])
                        accuracy=float(line[6][:-1])

                        if "loss" in dict:
                            dict["loss"].append(loss)
                        else:
                            dict["loss"] = [loss]
                        if "accuracy" in dict:
                            dict["accuracy"].append(accuracy)
                        else:
                            dict["accuracy"] = [accuracy]

                    elif "Epoch" in line and "loss" in line and "acc" in line:
                        # Rimpiazzo , e = con spazi
                        line=line.replace("=", " ")
                        line=line.split(" ")
                        loss=float(line[3][:-1])
                        accuracy=float(line[7][:-1])
                        if "loss" in dict:
                            dict["loss"].append(loss)
                        else:
                            dict["loss"] = [loss]
                        if "accuracy" in dict:
                            dict["accuracy"].append(accuracy)
                        else:
                            dict["accuracy"] = [accuracy]
                    
                    elif "loss" in line and "accuracy" in line:
                        line=line.split(" ")
                        loss=float(line[5])
                        accuracy=float(line[8])
                        if "loss" in dict:
                            dict["loss"].append(loss)
                        else:
                            dict["loss"] = [loss]
                        if "accuracy" in dict:
                            dict["accuracy"].append(accuracy)
                        else:
                            dict["accuracy"] = [accuracy]

                if dict:
                    self.trend.append(dict)
                
            return True
        except Exception as e:
            print(f"⚠ Error loading trend: {e}")
            return False
        
    def stampa_trend(self ):
        """Stampa il trend in modo leggibile"""
        print("\n🔍 Trend:")

        for point in self.trend:
            iteration = point.get("iteration", "N/A")
            print(f"Iteration {iteration}:")
            for key, value in point.items():
                if key != "iteration":
                    print(f"  {key}: {value}")
            print("-" * 40)
    
    def grafici_trend(self, i: int, output_path: Optional[Path] = None):
        """Specificata un'iterazione, crea grafici per visualizzare l'andamento di accuracy e loss in quella iterazione"""

        if not self.trend:
            print(f"⚠ No trend data to plot for {self.exp_name}.")
            return
        
        if i >= len(self.trend):
            print(f"⚠ Iteration index {i} out of range. Available iterations: 0 to {len(self.trend)-1} in {self.exp_name}.")
            return
        
        if "loss" not in self.trend[i] or "accuracy" not in self.trend[i]:
            print(f"⚠ Trend data for iteration {i} is incomplete. Missing 'loss' or 'accuracy' in {self.exp_name}.")
            return
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 7))
        sns.lineplot(x=range(len(self.trend[i]["loss"])), y=self.trend[i]["loss"], ax=ax[0])
        ax[0].set_title(f"Loss Trend - Iteration {self.trend[i]['iteration']}")
        sns.lineplot( x=range(len(self.trend[i]["accuracy"])), y=self.trend[i]["accuracy"], ax=ax[1])
        ax[1].set_title(f"Accuracy Trend - Iteration {self.trend[i]['iteration']}")

        plt.tight_layout()
        if output_path:
            output_path = Path(output_path)
            output_path=os.path.join(output_path, f"{self.exp_dir.name}_trend_{i}.png")
            plt.savefig(output_path)
        #plt.show()
        return fig

def find_out(exp_dir: Path):
    """Trova file .out"""
    list_of_out_files = []
    exp_dir = Path(exp_dir)
    for i in exp_dir.iterdir():
        if i.is_dir():
            print(f"🔍 Searching for search space in: {i}")
            esperimento = FileOUT(i)

            if esperimento.load_trend ():
                print(f"✅ Trend found in: {i}")
            else:
                print(f"⚠ No trend found in: {i}")
                continue

            if esperimento.load_search_space():
                print(f"✅ Search space found in: {i}")
            else:
                print(f"⚠ No search space found in: {i}")
                continue

            list_of_out_files.append(esperimento)

    return list_of_out_files

def main():
    """Main function"""
    print("\n" + "="*70)
    print("  EXPERIMENT RESULTS ANALYZER - Symbolic DNN Tuner")
    print("="*70 + "\n")
    
    # Check if arguments have been passed
    if len(sys.argv) > 1:
        # Batch mode: use command line arguments
        exp_dir = Path(sys.argv[1])
        if not exp_dir.exists():
            print(f"Folder not found: {exp_dir}")
            return
        
        print(f"Input folder: {exp_dir}")
    else:
        # Interactive mode
        path_input = input("Path of the folder containing the experiments: ").strip()
        if path_input and Path(path_input).exists():
            exp_dir = Path(path_input)
        
        if exp_dir is None:
            print("No folder selected")
            return
        
        print(f"\nSelected folder: {exp_dir}\n")
    
    # Analyze experiments
    list_of_out_files = find_out(exp_dir)

    for esperimento in list_of_out_files:
        print(f"\nAnalyzing experiment: {esperimento.exp_name}")
        esperimento.stampa_trend()
        esperimento.stampa_spazio_ricerca()
        esperimento.grafici_trend(0, output_path=exp_dir)
        esperimento.grafici_spazio_ricerca(output_path=exp_dir)

    print("\n" + "="*70)
    print("  Analysis completed!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()