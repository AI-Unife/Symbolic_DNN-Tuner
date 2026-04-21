from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

class SpazioRicerca:
    """Classe per determinare lo spazio di ricerca"""

    def __init__(self, exp_dir: Path):
        #setto le cartelle 
        self.exp_dir = Path(exp_dir)
        #exp_name è il nome dell'ultima cartella di exp_dir
        self.exp_name=self.exp_dir.name
        #creo una lista di dizionari vuoti per memorizzare i risultati
        self.spazio_ricerca = []
    
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
        
    def stampa_spazio_ricerca(self, spazio_ricerca: List[Dict[str, Any]]):
        """Stampa lo spazio di ricerca in modo leggibile"""
        print("\n🔍 Search Space:")

        for point in spazio_ricerca:
            iteration = point.get("iteration", "N/A")
            print(f"Iteration {iteration}:")
            for key, value in point.items():
                if key != "iteration":
                    print(f"  {key}: {value}")
            print("-" * 40)
    
    def grafici(self, spazio_ricerca: List[Dict[str, Any]], output_path: Optional[Path] = None):
        """Crea grafici per visualizzare lo spazio di ricerca"""
        if not spazio_ricerca:
            print("⚠ No search space data to plot.")
            return
        
        # Converti la lista di dizionari in un DataFrame
        df = pd.DataFrame(spazio_ricerca)

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
        return fig

def find_search_space(exp_dir: Path):
    """Trova lo spazio di ricerca in un esperimento"""
    spazi_ricerca = []
    exp_dir = Path(exp_dir)
    for i in exp_dir.iterdir():
        if i.is_dir():
            print(f"🔍 Searching for search space in: {i}")
            search_space = SpazioRicerca(i)
            if search_space.load_search_space():
                print(f"✅ Search space found in: {i}")
                spazi_ricerca.append(search_space)
            else:
                print(f"⚠ No search space found in: {i}")
                spazi_ricerca.append(None)
    return spazi_ricerca 

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
    spazi_ricerca = find_search_space(exp_dir)

    for search_space in spazi_ricerca:
        if search_space:
            search_space.stampa_spazio_ricerca(search_space.spazio_ricerca)
            search_space.grafici(search_space.spazio_ricerca)

    print("\n" + "="*70)
    print("  Analysis completed!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()