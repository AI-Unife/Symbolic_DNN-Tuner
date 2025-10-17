import subprocess
from itertools import product

datasets_cifar = ['CIFAR-10', 'CIFAR-100', 'Imagenet16120']
optimizers = ['filtered', 'standard', 'basic', 'RS', 'RS_ruled']
seeds = [42, 84, 123]


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

def main():
    job_configs = generate_jobs()
    save_job_configs_to_file(job_configs)

if __name__ == "__main__":
    main()
