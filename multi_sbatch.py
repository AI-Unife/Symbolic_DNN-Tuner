import os

# Definizione delle configurazioni
configurations = [
    # {"dataset": "CIFAR-10", "MODULE": "accuracy_module"},
    # {"dataset": "CIFAR-10", "MODULE": "accuracy_module flops_module"},
    # {"dataset": "CIFAR-10", "MODULE": "accuracy_module flops_module hardware_module"},
    # {"dataset": "CIFAR-10", "MODULE": "accuracy_module hardware_module"},
    {"dataset": "CIFAR-100", "MODULE": "accuracy_module"},
    {"dataset": "CIFAR-100", "MODULE": "accuracy_module flops_module"},
    {"dataset": "CIFAR-100", "MODULE": "accuracy_module flops_module hardware_module"},
    {"dataset": "CIFAR-100", "MODULE": "accuracy_module hardware_module"},
    
    # {"MODE": "depth", "FRAMES": 4, "CHANNEL": 4},
    # {"MODE": "depth", "FRAMES": 8, "CHANNEL": 8}, 
    # {"MODE": "depth", "FRAMES": 16, "CHANNEL": 16},
    # {"MODE": "depth", "FRAMES": 32, "CHANNEL": 32},
    # {"MODE": "depth", "FRAMES": 64, "CHANNEL": 64},
    # {"MODE": "fwdPass", "FRAMES": 4, "CHANNEL": 2}, 
    # {"MODE": "fwdPass", "FRAMES": 8, "CHANNEL": 2},
    # {"MODE": "fwdPass", "FRAMES": 16, "CHANNEL": 2},
    # {"MODE": "fwdPass", "FRAMES": 32, "CHANNEL": 2}, 
    # {"MODE": "fwdPass", "FRAMES": 64, "CHANNEL": 2}, 
    # {"MODE": "hybrid", "FRAMES": 16, "CHANNEL": 4},
    # {"MODE": "hybrid", "FRAMES": 16, "CHANNEL": 8},
    # {"MODE": "hybrid", "FRAMES": 32, "CHANNEL": 4},
    # {"MODE": "hybrid", "FRAMES": 32, "CHANNEL": 8},
    # {"MODE": "hybrid", "FRAMES": 64, "CHANNEL": 4},
    # {"MODE": "hybrid", "FRAMES": 64, "CHANNEL": 8},
    # {"MODE": "hybrid", "FRAMES": 64, "CHANNEL": 16},
]

# Crea una directory per gli script
os.makedirs("sbatch_scripts", exist_ok=True)

# Testo comune del file sbatch
header = """#!/bin/bash

#SBATCH --job-name={job_name}
#SBATCH --partition=gpuResB
#SBATCH --gres=gpu:1
#SBATCH --qos=gpuResB_qos

TIMESTAMP=$(date +%y_%m_%d_%H_%M_%S_%N)

MODULE="{module}"
DATASET="{dataset}"
EPOCH=50
MAX_EVAL=300

# Genera il nome dell'esperimento
NAME_EXP="$(date +%y_%m_%d_%H)_${{SLURM_JOB_ID}}_{mod_name}_${{MAX_EVAL}}_${{EPOCH}}"

mkdir -p ${{NAME_EXP}}

# Reindirizza manualmente stdout e stderr
exec > ${{NAME_EXP}}/${{TIMESTAMP}}.out 2> ${{NAME_EXP}}/${{TIMESTAMP}}.err

module load cuda/12.2
module load miniconda3/24.4.0
conda activate tf

cd ${{HOME}}/Symbolic_DNN-Tuner
python main.py --mod_list ${{MODULE}} --name ${{NAME_EXP}} --max_eval ${{MAX_EVAL}} --epochs ${{EPOCH}} --data_name ${{DATASET}} 

if [[ -n "$SLURM_JOB_ID" ]]; then
    mv slurm-${{SLURM_JOB_ID}}.out ${{NAME_EXP}}/
fi
"""

# Genera un file per ogni configurazione
for config in configurations:
    # filename = f"sbatch_scripts/sbatch_{config['MODE']}_frames{config['FRAMES']}_channels{config['CHANNEL']}.slurm"
    # job_name = f"{config['MODE']}_{config['FRAMES']}_{config['CHANNEL']}"

    filename = "sbatch_scripts/sbatch_{}_{}.slurm".format(config['dataset'], "_".join(config['MODULE'].split(" ")))
    job_name = "{}-c{}".format("".join([mod[0] for mod in config['MODULE'].split(" ")]), config['dataset'][-2:])
    
    with open(filename, "w") as f:
        # f.write(header.format(
        #     job_name=job_name,
        #     frames=config["FRAMES"],
        #     mode=config["MODE"],
        #     channel=config["CHANNEL"],
        # ))
        f.write(header.format(
            job_name=job_name,
            mod_name=config["dataset"].replace("-", "")+ "_" + config["MODULE"].replace(" ", "_"),
            module=config["MODULE"],
            dataset=config["dataset"],
        ))
    # for i in range(3):
    os.system(f"sbatch {filename}")

print("Script sbatch generati nella cartella 'sbatch_scripts' e run avviata.")
