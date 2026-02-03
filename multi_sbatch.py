import subprocess
from itertools import product

datasets_cifar = ['CIFAR-10', 'CIFAR-100', 'tinyImageNet']
optimizers = ['filtered', 'RS_ruled']
seeds = [42, 123, 96, 7]


def generate_jobs():
    job_configs = []

    # CIFAR - flops
    for dataset, optimizer, seed in product(datasets_cifar, optimizers, seeds):
        job_configs.append({
            "data_name": dataset,
            "opt": optimizer,
            "seed": seed,
        })
    return job_configs



def save_job_configs_to_file(job_configs, filename="params.txt"):
    # Ordina per seed
    job_configs = sorted(job_configs, key=lambda x: x["seed"])

    with open(filename, "w") as f:
        for config in job_configs:
            line = ",".join(str(config[k]) for k in ["data_name", "opt", "seed"])
            f.write(line + "\n")
            
# generate_params.py
from pathlib import Path

def generate_params_file(output_path: str = "params_gesture.txt"):
    """
    Genera un file params.txt per job array, dove ogni riga è una combinazione di parametri
    a partire da 'gesture'. Ignora prefissi e ID.
    """
    dataset = "roigesture"
    eval = 100
    epochs = 30

    # definizione delle configurazioni
    configs = [
        ("depth",   [(4, 4, 2), (8, 8, 2), (16, 16, 2), (32, 32, 2), (64, 64, 2)]),
        ("fwdPass", [(4, 2, 2), (8, 2, 2), (16, 2, 2), (32, 2, 2), (64, 2, 2)]),
        ("hybrid",  [(16, 4, 2), (16, 8, 2), (32, 4, 2), (32, 8, 2), (64, 4, 2), (64, 8, 2), (64, 16, 2)]),
    ]

    lines = []
    for mode, params_list in configs:
        for p in params_list:
            line = f"{dataset},{eval},{epochs},{mode},{p[0]},{p[1]},{p[2]}"
            lines.append(line)

    # scrive il file
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"✅ File '{output_path}' generato con {len(lines)} combinazioni.")

def main():
    job_configs = generate_jobs()
    save_job_configs_to_file(job_configs)
    # generate_params_file()

if __name__ == "__main__":
    main()
