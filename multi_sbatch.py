import subprocess
from itertools import product

datasets_cifar = ['CIFAR-10', 'CIFAR-100']
optimizers = ['filtered', 'standard', 'basic', 'RS', 'RS_ruled']
modules_list = [
    ['accuracy_module'],
    # ['accuracy_module', 'flops_module'],
    # ['accuracy_module', 'hardware_module'],
    ['accuracy_module', 'flops_module', 'hardware_module']
]
gesture_modes = ['fwd']#, 'hybrid', 'depth']
frames_list = [32, 64]
channels_list =[2]# [4, 8, 16, 32, 64]
seeds = [42, 84, 123, 256]


def generate_jobs():
    job_configs = []

    # CIFAR - accuracy only
    for dataset, optimizer, seed in product(datasets_cifar, optimizers, seeds):
        job_configs.append({
            "data_name": dataset,
            "opt": optimizer,
            "mod_list": "accuracy_module flops_module",
            "seed": seed,
            "frames": 1,
            "channels": 1
        })

    # # Gesture - accuracy only
    # for optimizer, mode, frames, channel, seed in product(optimizers, gesture_modes, frames_list, channels_list, seeds):
    #     job_configs.append({
    #         "data_name": "gesture",
    #         "opt": optimizer,
    #         "mod_list": "accuracy_module",
    #         "gesture_mode": mode,
    #         "frames": frames,
    #         "channels": channel,
    #         "seed": seed
    #     })

    # # # CIFAR - all module combinations
    # for dataset, optimizer, modules, seed in product(datasets_cifar, optimizers, modules_list, seeds):
    #     job_configs.append({
    #         "data_name": dataset,
    #         "opt": optimizer,
    #         "mod_list": " ".join(modules),
    #         "seed": seed,
    #         "frames": 1,
    #         "channels": 1
    #     })
    # for optimizer, mode, frames, channel, modules, seed in product(optimizers, gesture_modes, frames_list, channels_list, modules_list, seeds):
    #     job_configs.append({
    #         "data_name": dataset,
    #         "opt": optimizer,
    #         "mod_list": " ".join(modules),
    #         "seed": seed,
    #         "frames": 1,
    #         "channels": 1
    #     })

    return job_configs



def save_job_configs_to_file(job_configs, filename="params.txt"):
    # Ordina per seed
    job_configs = sorted(job_configs, key=lambda x: x["seed"])

    with open(filename, "w") as f:
        for config in job_configs:
            line = ",".join(str(config[k]) for k in ["data_name", "opt", "mod_list", "seed", "frames", "channels"])
            f.write(line + "\n")

def main():
    job_configs = generate_jobs()
    save_job_configs_to_file(job_configs)

if __name__ == "__main__":
    main()