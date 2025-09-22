import os
import shutil 
# Percorso alla cartella principale
tunerType = "results"  
root_folder = tunerType + "/"
dest_folder = "experiments/" + tunerType + "/"
# dest_folder = "/Users/alicebizzarri/Desktop/experiments/filtered_BO/"
# root_folder = "experiments/esp_giallo/"


# Cammina attraverso tutte le sottocartelle
for dirpath, dirnames, filenames in os.walk(root_folder):
    for filename in filenames:
        if filename.endswith(".csv"):
            dest_path = os.path.join(dest_folder, dirpath.split('/')[-1])
            os.makedirs(dest_path, exist_ok=True)
            old_file = os.path.join(dirpath, filename)
            new_file = os.path.join(dest_path, filename)
        elif filename.startswith("25_") and filename.endswith(".out"):
            dest_path = os.path.join(dest_folder, dirpath.split('/')[-1])
            os.makedirs(dest_path, exist_ok=True)
            old_file = os.path.join(dirpath, filename)
            new_file = os.path.join(dest_path, "output.log")
        else:
            continue
        try:
            shutil.copy(old_file, new_file)
            print(f"Spostato: {old_file} -> {new_file}")
        except Exception as e:
            print(f"Errore nello spostare {old_file} --> {new_file}: {e}")
# exit()
import os
import re
import pandas as pd
import numpy as np
import ast
# Path alla cartella che contiene tutti gli esperimenti
EXPERIMENTS_DIR = "experiments/"

# Lista per salvare tutti i dati aggregati
all_results = []
all_modules = ['accuracy_module', 'flops_module', 'hardware_module']
# Cicla tutte le sottocartelle
for tuner in os.listdir(EXPERIMENTS_DIR):
    if tuner == 'results':
        if tuner not in ['.DS_Store']:  # Sostituisci con i nomi dei tuner che ti interessano
            for exp_name in os.listdir(os.path.join(EXPERIMENTS_DIR, tuner)):
                exp_path = os.path.join(EXPERIMENTS_DIR, tuner, exp_name)
                if not os.path.isdir(exp_path):
                    continue

                output_log_path = os.path.join(exp_path, 'output.log')
                results_csv_path = os.path.join(exp_path, 'results.csv')

                # Skip se non esistono i file richiesti
                if not os.path.isfile(output_log_path) or not os.path.isfile(results_csv_path):
                    continue

                # Dizionario per memorizzare le info dell'esperimento
                exp_info = {}

                # Estrai info da output.log
                with open(output_log_path, 'r') as f:
                    content = f.read()

                # Regex per estrarre i campi richiesti
                # exp_name_match = re.search(r'EXPERIMENT NAME:\s+(.+)', content)
                dataset_match = re.search(r'DATASET NAME:\s+(.+)', content)
                max_eval_match = re.search(r'MAX NET EVAL:\s+(\d+)', content)
                epochs_match = re.search(r'EPOCHS FOR TRAINING:\s+(\d+)', content)
                modules_match = re.search(r'MODULE LIST:\s+(\[.+\])', content)
                seed_match = re.search(r'SEED:\s+(\d+)', content)

                # exp_info['Tuner'] = tuner
                exp_info['Tuner'] = tuner
                exp_info['Experiment Name'] = exp_name #exp_name_match.group(1).strip() if exp_name_match else ''
                exp_info['Dataset Name'] = dataset_match.group(1).strip() if dataset_match else ''
                exp_info['Max Net Eval'] = int(max_eval_match.group(1)) if max_eval_match else np.nan
                exp_info['Epochs for Training'] = int(epochs_match.group(1)) if epochs_match else np.nan
                exp_info['seed'] = int(seed_match.group(1)) if seed_match else np.nan
                modules = modules_match.group(1).strip() if modules_match else ''
                for module in all_modules:
                    if module not in modules:
                        exp_info[module] = False
                    else:
                        exp_info[module] = True

                # Estrai NUMERO FRAMES, CHANNEL, POLARITY dall'Experiment Name
                name_parts = exp_info['Experiment Name'].split('_')
                if len(name_parts) >= 3:
                    # try:
                    exp_info['Tuner'] = name_parts[6] if len(name_parts) > 6 else tuner
                    if name_parts[7] == 'ruled':
                        exp_info['Tuner'] += '_ruled'
                    #     exp_info['Numero Frames'] = int(name_parts[-3]) if name_parts[-3].isdigit() else np.nan
                    #     exp_info['Channel'] = int(name_parts[-2]) if name_parts[-2].isdigit() else np.nan
                    #     exp_info['Polarity'] = int(name_parts[-1]) if name_parts[-1].isdigit() else np.nan
                    #     exp_info['Mode'] = "depth" if 'depth' in exp_info['Experiment Name'] else "fwdPass" if 'fwdPass' in exp_info['Experiment Name'] else "hybrid"
                    # except ValueError:
                    #     exp_info['Tuner'] = tuner
                    #     exp_info['Numero Frames'] = np.nan
                    #     exp_info['Channel'] = np.nan
                    #     exp_info['Polarity'] = np.nan
                    #     exp_info['Mode'] = "unknown"
                # else:
                #     exp_info['Numero Frames'] = np.nan
                #     exp_info['Channel'] = np.nan
                #     exp_info['Polarity'] = np.nan
                #     exp_info['Mode'] = "unknown"

                print(f"Processing experiment: {exp_info['Experiment Name']}")
                # Leggi results.csv e prendi il massimo per ogni metrica disponibile
                try:
                    results_df = pd.read_csv(results_csv_path)
                except pd.errors.EmptyDataError:
                    print(f"⚠️ Risultati vuoti per l'esperimento: {exp_info['Experiment Name']}")
                    continue

                # Seleziona solo le colonne numeriche (accuracy, flops, latency, ecc.)
                numeric_cols = results_df.select_dtypes(include=[np.number]).columns

                # Calcola il massimo per ciascuna colonna numerica se non è tutta NaN
                exp_info[f'Best Iteration'] = results_df.loc[results_df['accuracy'].idxmax(), 'iteration'] 
                for col in numeric_cols:
                    if col not in 'iteration':
                        if not results_df[col].dropna().empty:
                            exp_info[f'Best {col}'] = results_df[col].max()                            
                        else:
                            exp_info[f'Best {col}'] = np.nan
                exp_info['Eval'] = len(results_df['accuracy'])
                all_results.append(exp_info)

# Crea DataFrame finale
total_df = pd.DataFrame(all_results)

# Salva su CSV
total_df.to_csv('total_results.csv', index=False)

print('✅ File total_results.csv creato con successo!')