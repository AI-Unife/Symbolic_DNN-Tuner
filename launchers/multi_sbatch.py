import itertools
import os

# Lista dei moduli disponibili
MODULE_LIST = ["accuracy_module", "flops_module", "hardware_module"]

# Numero di esperimenti richiesti
N_EXPERIMENTS = 7

DATA_NAME = "gesture"
MODE = "fwdPass"

max_eval = 100
epochs = 20

# Genera tutte le combinazioni possibili dei moduli (da 1 a tutti)
all_combinations = []
for r in range(1, len(MODULE_LIST) + 1):
    all_combinations.extend(itertools.combinations(MODULE_LIST, r))

# Seleziona solo le prime N_EXPERIMENTS combinazioni
selected_combinations = [MODULE_LIST[:1], MODULE_LIST[:2], MODULE_LIST]

# Cartella per gli script generati
# os.makedirs("generated_scripts", exist_ok=True)

# Genera uno script SLURM per ogni combinazione selezionata
for i, combination in enumerate(selected_combinations):
    modules_str = " ".join(combination)  # Per il comando Python
    modules_exp_name = "_".join(combination)  # Per il nome esperimento

    script_content = f"""#!/bin/bash

#SBATCH --job-name=Tuner_{i+1}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:GH100:1
#SBATCH --mem=64G


TIMESTAMP=$(date +%y_%m_%d_%H_%M_%S_%N)

# Genera il nome dell'esperimento
NAME_EXP="$(date +%y_%m_%d_%H_%M)_{MODE}_{DATA_NAME}_{modules_exp_name}_{max_eval}_{epochs}"

mkdir -p ${{NAME_EXP}}

# Reindirizza manualmente stdout e stderr
exec > ${{NAME_EXP}}/${{TIMESTAMP}}.out 2> ${{NAME_EXP}}/${{TIMESTAMP}}.err

module load cuda/12.2
module load miniconda3/24.4.0
conda activate tf

python main.py --mod_list {modules_str} --name ${{NAME_EXP}} --max_eval {max_eval} --epochs {epochs} --data_name {DATA_NAME}

if [[ -n "$SLURM_JOB_ID" ]]; then
    mv slurm-${{SLURM_JOB_ID}}.out ${{NAME_EXP}}/
fi
"""

    # Scrive lo script in un file
    script_filename = f"experiment_{i+1}.slurm"
    with open(script_filename, "w") as script_file:
        script_file.write(script_content)

    # Rende eseguibile lo script
    # os.chmod(script_filename, 0o755)

    # Avvia l'esperimento con sbatch
    os.system(f"sbatch {script_filename}")

print(f"Avviati {len(selected_combinations)} esperimenti con combinazioni diverse di moduli!")