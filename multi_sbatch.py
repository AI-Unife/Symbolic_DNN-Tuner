import os

# Definizione delle configurazioni
configurations = [
    # {"MODE": "depth", "FRAMES": 4, "CHANNEL": 4},
    {"MODE": "depth", "FRAMES": 8, "CHANNEL": 8}, #mancano 2
    # {"MODE": "depth", "FRAMES": 16, "CHANNEL": 16},
    # {"MODE": "depth", "FRAMES": 32, "CHANNEL": 32},
    # {"MODE": "depth", "FRAMES": 64, "CHANNEL": 64},
    # {"MODE": "fwdPass", "FRAMES": 4, "CHANNEL": 2}, 
    # {"MODE": "fwdPass", "FRAMES": 8, "CHANNEL": 2},
    # {"MODE": "fwdPass", "FRAMES": 16, "CHANNEL": 2},
    {"MODE": "fwdPass", "FRAMES": 32, "CHANNEL": 2}, # mancano 2
    # {"MODE": "fwdPass", "FRAMES": 64, "CHANNEL": 2}, # annullato
    {"MODE": "hybrid", "FRAMES": 16, "CHANNEL": 4},
    {"MODE": "hybrid", "FRAMES": 16, "CHANNEL": 8},
    {"MODE": "hybrid", "FRAMES": 32, "CHANNEL": 4},
    {"MODE": "hybrid", "FRAMES": 32, "CHANNEL": 8},
    # {"MODE": "hybrid", "FRAMES": 64, "CHANNEL": 4},# annullato
    # {"MODE": "hybrid", "FRAMES": 64, "CHANNEL": 8},# annullato
    # {"MODE": "hybrid", "FRAMES": 64, "CHANNEL": 16},# annullato
]

# Crea una directory per gli script
os.makedirs("sbatch_scripts", exist_ok=True)

# Testo comune del file sbatch
header = """#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:GH100:1

TIMESTAMP=$(date +%y_%m_%d_%H_%M_%S_%N)
FRAMES={frames}
CHANNEL={channel}
EPOCH=20
MAX_EVAL=100
MODE="{mode}"

# Genera il nome dell'esperimento
NAME_EXP="$(date +%y_%m_%d_%H)_${SLURM_JOB_ID}_${{MODE}}_gesture_accuracy_module_${{MAX_EVAL}}_${{EPOCH}}_${{FRAMES}}_${{CHANNEL}}_2"

mkdir -p ${{NAME_EXP}}

# Reindirizza manualmente stdout e stderr
exec > ${{NAME_EXP}}/${{TIMESTAMP}}.out 2> ${{NAME_EXP}}/${{TIMESTAMP}}.err

module load cuda/12.2
module load miniconda3/24.4.0
conda activate tf

cd ${{HOME}}/Symbolic_DNN-Tuner
python main.py --mod_list accuracy_module --name ${{NAME_EXP}} --max_eval ${{MAX_EVAL}} --epochs ${{EPOCH}} --data_name gesture --mode ${{MODE}} --frames ${{FRAMES}} --channels ${{CHANNEL}}

if [[ -n "$SLURM_JOB_ID" ]]; then
    mv slurm-${{SLURM_JOB_ID}}.out ${{NAME_EXP}}/
fi
"""

# Genera un file per ogni configurazione
for config in configurations:
    filename = f"sbatch_scripts/sbatch_{config['MODE']}_frames{config['FRAMES']}_channels{config['CHANNEL']}.slurm"
    job_name = f"{config['MODE']}_{config['FRAMES']}_{config['CHANNEL']}"
    
    with open(filename, "w") as f:
        f.write(header.format(
            job_name=job_name,
            frames=config["FRAMES"],
            mode=config["MODE"],
            channel=config["CHANNEL"],
        ))
    for i in range(5):
        os.system(f"sbatch {filename}")

print("Script sbatch generati nella cartella 'sbatch_scripts' e run avviata 5 volte per esperimento.")
