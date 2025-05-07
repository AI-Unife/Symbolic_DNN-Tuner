import re
import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import colorsys

with open("results.txt", "r") as file:
    lines = file.readlines()
    start = 0
    end = 0
    start_pattern = re.compile(r"BEST RESULT")
    end_pattern = re.compile(r"--------------------------------------------------")
    gpu_pattern = re.compile(r"GPU:\s([\d\:\d\:\d\.]+)")
    cpu_pattern = re.compile(r"CPU:\s([\d\:\d\:\d\.]+)")
    accuracy_pattern = re.compile(r"Accuracy massima:\s([\d\.]+)")
    flops_pattern = re.compile(r"FLOPS:\s([\d\,]+)")
    params_pattern = re.compile(r"Params:\s([\d\,]+)")
    mode_pattern = re.compile(r"Exp: Mode\s([\w\.\-]+)")
    frames_pattern = re.compile(r"Frames\s([\d]+)")
    channel_pattern = re.compile(r"channel\s([\d]+)")
    pol_pattern = re.compile(r"Polarity\s([\w\.\-]+)")
    error_pattern = re.compile(r"Model not found or error in evaluation")
    results = {'name': [], 'gpu_time': [], 'cpu_time': [], 'accuracy': [], 'flops': [], 'params': []}
    for line in lines:
        mode_match = mode_pattern.search(line)
        if mode_match:
            frames_match = frames_pattern.search(line)
            channel_match = channel_pattern.search(line)
            pol_match = pol_pattern.search(line)
            name = mode_match.group(1) + "_" + frames_match.group(1) + "_" + channel_match.group(1) + "_" + pol_match.group(1)
            if 'name' not in results:
                results['name'] = []
            results['name'].append(name)
        if start_pattern.search(line):
            start = 1
            end = 0
        if start == 1 and end == 0:
            gpu_match = gpu_pattern.search(line)
            if gpu_match:
                gpu_time = gpu_match.group(1)
        
                results['gpu_time'].append(gpu_time)
            cpu_match = cpu_pattern.search(line)
            if cpu_match:
                cpu_time = cpu_match.group(1)
                results['cpu_time'].append(cpu_time)
            accuracy_match = accuracy_pattern.search(line)
            if accuracy_match:
                accuracy = accuracy_match.group(1)
                results['accuracy'].append(accuracy)
            flops_match = flops_pattern.search(line)
            if flops_match:
                flops = flops_match.group(1)
                results['flops'].append(flops[:-1])
            params_match = params_pattern.search(line)
            if params_match:
                params = params_match.group(1)
                results['params'].append(params)    
            error_match = error_pattern.search(line)
            if error_match:
                results['gpu_time'].append("0:00:00")
                results['cpu_time'].append("0:00:00")
                results['flops'].append("0,000,000")
                results['params'].append("0,000,000")
        if end_pattern.search(line):
            end = 1
            start = 0

print("Results:")
for key, value in results.items():
    print(f"{key}: {value}\n")

if len(results['gpu_time']) < len(results['name']):
    results['gpu_time'] = ["0:00:00" for _ in range(len(results['name']))]
data = pd.DataFrame(results)
# Conversioni
data['gpu_time'] = pd.to_timedelta(data['gpu_time']).dt.total_seconds() * 1000
data['cpu_time'] = pd.to_timedelta(data['cpu_time']).dt.total_seconds() * 1000
data['flops'] = data['flops'].str.replace(',', '').astype(int)
data['params'] = data['params'].str.replace(',', '').astype(int)
data['accuracy'] = data['accuracy'].astype(float)


combined_file = "results.csv"

# Carica il file esistente se c'è
if os.path.exists(combined_file):
    existing_data = pd.read_csv(combined_file)
    combined = pd.concat([existing_data, data], ignore_index=True)

    # Rimuove righe duplicate basate su tutte le colonne
    combined = combined.drop_duplicates()

    # Salva
    combined.to_csv(combined_file, index=False)
    combined.to_excel("results.xlsx", index=False)
    print("Appended to results.csv and saved.")
else:
    # Se il file non esiste, salva direttamente
    data.to_csv(combined_file, index=False)
    data.to_excel("results.xlsx", index=False)
    print("Created results.csv and saved.")
    

data = data.sort_values('accuracy', ascending=False).groupby('name').head(3).reset_index()
print(data)
configurations = [
    {"MODE": "depth", "FRAMES": 4, "CHANNEL": 4, "POL": 2},
    {"MODE": "depth", "FRAMES": 8, "CHANNEL": 8, "POL": 2}, 
    {"MODE": "depth", "FRAMES": 16, "CHANNEL": 16, "POL": 2},
    {"MODE": "depth", "FRAMES": 32, "CHANNEL": 32, "POL": 2},
    {"MODE": "depth", "FRAMES": 64, "CHANNEL": 64, "POL": 2},
    # {"MODE": "depth", "FRAMES": 4, "CHANNEL": 4, "POL": 1},
    # {"MODE": "depth", "FRAMES": 8, "CHANNEL": 8, "POL": 1}, 
    # {"MODE": "depth", "FRAMES": 16, "CHANNEL": 16, "POL": 1},
    # {"MODE": "depth", "FRAMES": 32, "CHANNEL": 32, "POL": 1},
    # {"MODE": "depth", "FRAMES": 64, "CHANNEL": 64, "POL": 1},
    {"MODE": "fwdPass", "FRAMES": 4, "CHANNEL": 2, "POL": 2}, 
    {"MODE": "fwdPass", "FRAMES": 8, "CHANNEL": 2, "POL": 2},
    {"MODE": "fwdPass", "FRAMES": 16, "CHANNEL": 2, "POL": 2},
    {"MODE": "fwdPass", "FRAMES": 32, "CHANNEL": 2, "POL": 2},
    {"MODE": "fwdPass", "FRAMES": 64, "CHANNEL": 2, "POL": 2}, 
    {"MODE": "hybrid", "FRAMES": 16, "CHANNEL": 4, "POL": 2},
    {"MODE": "hybrid", "FRAMES": 16, "CHANNEL": 8, "POL": 2},
    {"MODE": "hybrid", "FRAMES": 32, "CHANNEL": 4, "POL": 2},
    {"MODE": "hybrid", "FRAMES": 32, "CHANNEL": 8, "POL": 2},
]

config_keys = [f"{cfg['MODE']}_{cfg['FRAMES']}_{cfg['CHANNEL']}_{cfg['POL']}" for cfg in configurations]

# # Mappa ogni configurazione a un colore unico da una mappa colormap
# cmap = plt.get_cmap('tab20', len(config_keys))  # puoi cambiare mappa (es. 'viridis', 'hsv'...)
# color_map = {key: cmap(i) for i, key in enumerate(config_keys)}


# Colori base per ogni MODE (in RGB [0-1])
base_colors = {
    "depth": (1.0, 1.0, 0.0),      # giallo
    "fwdPass": (0.0, 0.0, 1.0),    # blu
    "hybrid": (0.0, 1.0, 0.0),     # verde
}

# Crea mappa configurazioni -> colori variando luminosità
color_map = {}
mode_grouped = {}

# Raggruppa per MODE
for cfg in configurations:
    key = f"{cfg['MODE']}_{cfg['FRAMES']}_{cfg['CHANNEL']}_{cfg['POL']}"
    mode_grouped.setdefault(cfg['MODE'], []).append(key)

# Per ogni MODE, genera variazioni di luminosità
for mode, keys in mode_grouped.items():
    base_rgb = base_colors[mode]
    h, l, s = colorsys.rgb_to_hls(*base_rgb)

    for i, key in enumerate(keys):
        # varia leggermente la luminosità (entro range [0.4–0.85])
        new_l = 0.4 + (0.45 * i / max(1, len(keys)-1))
        new_rgb = colorsys.hls_to_rgb(h, new_l, s)
        color_map[key] = new_rgb

def get_color(name):
    return color_map.get(name, 'gray')  # default gray se non trova la configurazione
plt.figure(figsize=(15, 15))

for i in range(len(data)):
    x = data['cpu_time'][i]
    y = data['accuracy'][i]
    name = data['name'][i]
    color = get_color(name)
    name = f"F: {name.split('_')[1]} C: {name.split('_')[2]}"
    plt.scatter(x, y, color=color, s=100)

legend_elements = [mpatches.Patch(color=color_map[key], label=key) for key in config_keys] + [mpatches.Patch(color='gray', label='depth_X_X_1')]

# Legenda per i gruppi MODE (colori base)
# legend_elements = [
#     mpatches.Patch(color=base_colors["depth"], label="depth"),
#     mpatches.Patch(color=base_colors["fwdPass"], label="fwdPass"),
#     mpatches.Patch(color=base_colors["hybrid"], label="hybrid")
# ]

plt.legend(handles=legend_elements, fontsize=10, title="Configurazioni", title_fontsize=12, loc='lower right')

plt.xlabel('Tempo CPU (ms)')
plt.ylabel('Accuratezza')
plt.title('Accuratezza vs Tempo CPU')
plt.grid(True)
plt.tight_layout()
plt.savefig('CPU_acc.png')

# Scatter plot: GPU time vs Accuracy
plt.figure(figsize=(15, 15))
plt.scatter(data['gpu_time'], data['accuracy'], color='green')
for i in range(len(data)):
    plt.text(data['gpu_time'][i]+0.02, data['accuracy'][i], data['name'][i], fontsize=16)
plt.xlabel('Tempo GPU (ms)')
plt.ylabel('Accuratezza')
plt.title('Accuratezza vs Tempo GPU')
plt.grid(True)
plt.tight_layout()
plt.savefig('GPU_acc.png')