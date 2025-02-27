import re
import matplotlib.pyplot as plt
import os
import numpy as np


def plot_with_trend(x, y, label, color, subplot_index, total_plots, best='min'):
    plt.subplot(total_plots, 1, subplot_index)
    plt.plot(x, y, marker='o', linestyle='-', color=color, label=label)

    # Calcola il trend lineare
    if len(x) > 1:
        # coeffs = np.polyfit(x, y, 1)  # Regressione lineare di primo grado
        # trend = np.poly1d(coeffs)
        # Calcola il massimo trovato fino a ogni iterazione
        if best == 'min':
            best_so_far = np.minimum.accumulate(y)
        else:
            best_so_far = np.maximum.accumulate(y)
        plt.plot(x, best_so_far, linestyle="dashed", color="green", label="Best so far")

    plt.xlabel('Iterazione')
    plt.ylabel(label)
    plt.title(f'Andamento di {label}')
    plt.legend()
    plt.grid(True)

def plot_per_exp(base_dir, file_path):
    # Liste per memorizzare i valori estratti
    accuracy_values = []
    flops_values = []
    params_values = []
    latency_values = []
    total_cost_values = []
    iterations = []
    iteration = 0

    # Seleziona le espressioni regolari in base al modulo
    accuracy_pattern = None
    flops_pattern = None
    params_pattern = None
    latency_pattern = None
    total_cost_pattern = None
    n_plot = 0
    if "accuracy_module" in base_dir:
        accuracy_pattern = re.compile(r"ACCURACY:\s([\d\.]+)")
        n_plot += 1
    if "hardware_module" in base_dir:
        latency_pattern = re.compile(r"LATENCY:\s([\d\.]+)")
        total_cost_pattern = re.compile(r"TOTAL COST:\s([\d\.]+)")
        n_plot += 2
    if "flops_module" in base_dir:
        flops_pattern = re.compile(r"FLOPS:\s(\d+)")
        params_pattern = re.compile(r"PARAMS:\s([\d\.]+)")
        n_plot += 2
    
    with open(os.path.join(base_dir, file_path), "r") as file:
        for line in file:
            if accuracy_pattern:
                accuracy_match = accuracy_pattern.search(line)
                if accuracy_match:
                    accuracy_values.append(float(accuracy_match.group(1)))
            
            if flops_pattern:
                flops_match = flops_pattern.search(line)
                if flops_match:
                    flops_values.append(int(flops_match.group(1)))
            
            if params_pattern:
                params_match = params_pattern.search(line)
                if params_match:
                    params_values.append(float(params_match.group(1)))
            
            if latency_pattern:
                latency_match = latency_pattern.search(line)
                if latency_match:
                    latency_values.append(float(latency_match.group(1)))
            
            if total_cost_pattern:
                total_cost_match = total_cost_pattern.search(line)
                if total_cost_match:
                    total_cost_values.append(float(total_cost_match.group(1)))
            


    # Creazione dei grafici
    plt.figure(figsize=(30, 7*n_plot))
    current_plot = 0
    
    print("------------------- BEST RESULT -------------------")
    if accuracy_pattern:
        print("Total Iterations:", len(accuracy_values) - 1)
    elif flops_pattern:
        print("Total Iterations:", len(flops_values) - 1)
    elif latency_pattern:
        print("Total Iterations:", len(latency_values) - 1)
    if accuracy_values:
        max_accuracy = max(accuracy_values)  # Trova il valore massimo
        max_index = accuracy_values.index(max_accuracy)  # Trova l'indice del valore massimo

        print("Accuracy massima:", max_accuracy, "Iterazione:", max_index + 1)
        current_plot += 1
        iterations = list(range(1, len(accuracy_values) + 1))
        plot_with_trend(iterations, accuracy_values, 'Accuracy', 'b', current_plot, n_plot, best='max')
        

    if flops_values:
        min_flops = min(flops_values)  # Trova il valore minimo
        min_index = flops_values.index(min_flops)  # Trova l'indice del valore minimo   
        print("FLOPS minimo:", min_flops, "Iterazione:", min_index + 1)
        current_plot += 1
        iterations = list(range(1, len(flops_values) + 1))
        plot_with_trend(iterations, flops_values, 'FLOPS', 'r', current_plot, n_plot)

    if params_values:
        min_flops = min(params_values)  # Trova il valore minimo
        min_index = params_values.index(min_flops)  # Trova l'indice del valore minimo
        print("Params minimo:", min_flops, "Iterazione:", min_index + 1)
        current_plot += 1
        iterations = list(range(1, len(params_values) + 1))
        plot_with_trend(iterations, params_values, 'Params', 'g', current_plot, n_plot)
        
    if latency_values:
        min_latency = min(latency_values)  # Trova il valore minimo
        min_index = latency_values.index(min_latency)  # Trova l'indice del valore minimo
        print("Latency minima:", min_latency, "Iterazione:", min_index + 1)
        current_plot += 1
        iterations = list(range(1, len(latency_values) + 1))
        plot_with_trend(iterations, latency_values, 'Latency', 'y', current_plot, n_plot)

    if total_cost_values:
        min_total_cost = min(total_cost_values)  # Trova il valore minimo
        min_index = total_cost_values.index(min_total_cost)  # Trova l'indice del valore minimo
        print("Total Cost minimo:", min_total_cost, "Iterazione:", min_index + 1)
        current_plot += 1
        iterations = list(range(1, len(total_cost_values) + 1))
        plot_with_trend(iterations, total_cost_values, 'Total Cost', 'm', current_plot, n_plot)

    print("--------------------------------------------------")    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "output_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    for root, dirs, files in os.walk('/hpc/home/bzzlca/Symbolic_DNN-Tuner/'):
        if os.path.basename(root).startswith("25_02_") and os.path.basename(root).find("fwdPass") >= 1:
            for file in files:
                if "old_exp" not in root:
                    if file.endswith(".out") and file.startswith("25_02"):
                        print("Dir:", root)
                        plot_per_exp(root, file)
