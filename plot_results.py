import re
import matplotlib.pyplot as plt
import os
import numpy as np
from components.neural_network import evaluate_net


def plot_with_trend(x, y, label, color, subplot_index, total_plots, best='min', restart_points=None):
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
        
    # Aggiungi marcatori per i punti di riavvio della BO
    if restart_points:
        restart_x = [x[i] for i in restart_points]
        restart_y = [y[i] for i in restart_points]
        # restart_x =  [x[i] for i in range(len(x)) if i not in restart_points]
        # restart_y  = [y[i] for i in range(len(y)) if i not in restart_points]

        
        plt.scatter(restart_x, restart_y, color='red', marker='X', s=300, label="BO Restart")

 

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
    restart_indices = []
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
    
    bo_restart_detected = False
    with open(os.path.join(base_dir, file_path), "r") as file:
        for line in file:
            if "Restarting BO" in line:
                # if "Inside BO" in line:
                # print("BO Restart detected")
                bo_restart_detected = True
            if accuracy_pattern:
                accuracy_match = accuracy_pattern.search(line)
                if accuracy_match:
                    accuracy_values.append(float(accuracy_match.group(1)))
                    if bo_restart_detected:
                        restart_indices.append(len(accuracy_values) - 1)
                        bo_restart_detected = False  # Resetta il flag fino al prossimo riavvio

            
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
    
    print("------------------- BEST RESULT -------------------", flush=True)
    
    if accuracy_pattern:
        print("Total Iterations:", len(accuracy_values) - 1, flush=True)
    elif flops_pattern:
        print("Total Iterations:", len(flops_values) - 1)
    elif latency_pattern:
        print("Total Iterations:", len(latency_values) - 1)
    print("Total restarts:", len(restart_indices), flush=True)
    if accuracy_values:
        max_accuracy = max(accuracy_values)  # Trova il valore massimo
        max_index = accuracy_values.index(max_accuracy)  # Trova l'indice del valore massimo

        print("Accuracy massima:", max_accuracy, "Iterazione:", max_index, flush=True)
        current_plot += 1
        iterations = list(range(1, len(accuracy_values) + 1))
        plot_with_trend(iterations, accuracy_values, 'Accuracy', 'b', current_plot, n_plot, best='max', restart_points=restart_indices)
        

    if flops_values:
        min_flops = min(flops_values)  # Trova il valore minimo
        min_index = flops_values.index(min_flops)  # Trova l'indice del valore minimo   
        print("FLOPS minimo:", min_flops, "Iterazione:", min_index)
        current_plot += 1
        iterations = list(range(1, len(flops_values) + 1))
        plot_with_trend(iterations, flops_values, 'FLOPS', 'r', current_plot, n_plot)

    if params_values:
        min_flops = min(params_values)  # Trova il valore minimo
        min_index = params_values.index(min_flops)  # Trova l'indice del valore minimo
        print("Params minimo:", min_flops, "Iterazione:", min_index)
        current_plot += 1
        iterations = list(range(1, len(params_values) + 1))
        plot_with_trend(iterations, params_values, 'Params', 'g', current_plot, n_plot)
        
    if latency_values:
        min_latency = min(latency_values)  # Trova il valore minimo
        min_index = latency_values.index(min_latency)  # Trova l'indice del valore minimo
        print("Latency minima:", min_latency, "Iterazione:", min_index)
        current_plot += 1
        iterations = list(range(1, len(latency_values) + 1))
        plot_with_trend(iterations, latency_values, 'Latency', 'y', current_plot, n_plot)

    if total_cost_values:
        min_total_cost = min(total_cost_values)  # Trova il valore minimo
        min_index = total_cost_values.index(min_total_cost)  # Trova l'indice del valore minimo
        print("Total Cost minimo:", min_total_cost, "Iterazione:", min_index)
        current_plot += 1
        iterations = list(range(1, len(total_cost_values) + 1))
        plot_with_trend(iterations, total_cost_values, 'Total Cost', 'm', current_plot, n_plot)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "output_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    try:
        print(base_dir.split("_")[6], base_dir.split("_")[-3], base_dir.split("_")[-2], base_dir.split("_")[-1])
        # time, score, flops, nparams = evaluate_net(base_dir + "/Model/best-model.keras", base_dir.split("_")[6], base_dir.split("_")[-3], base_dir.split("_")[-2], base_dir.split("_")[-1])
        print(f"Time GPU: {time[0]}, Time CPU: {time[1]}, Loss: {score[0]:.4f}, Acc: {score[1]:.4f}, FLOPS: {flops:,}, Params: {nparams:,}", flush=True)
    except Exception as e:
        print(f"Time GPU: 0:00:00.00, Time CPU: 0:00:00.00, Loss: 0.00, Acc: 0.00, FLOPS: 0.00, Params: 0.00", flush=True)

    print("--------------------------------------------------", flush=True)    
    
if __name__ == "__main__":
    for root, dirs, files in os.walk('/hpc/home/bzzlca/Symbolic_DNN-Tuner'):
        if os.path.basename(root).startswith("25_06_17"): # or os.path.basename(root).startswith("25_06_05"):# and os.path.basename(root).find("cifar") >= 1:
            for file in files:
                if "old_exp" not in root and "results" not in root: # and "25_05" not in root:
                    if file.endswith(".out")  and file.startswith("25_"):
                        split_path = root.split("_")
                        print(f"\n{root}\nExp: Mode {split_path[6]} - Frames {split_path[-3]} - channel {split_path[-2]} - Polarity {split_path[-1]}")
                        # print(root)
                        plot_per_exp(root, file)
