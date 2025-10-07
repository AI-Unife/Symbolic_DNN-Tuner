import pandas as pd
import os
import re
import sys
import ast
import matplotlib.pyplot as plt



from datetime import datetime
import numpy as np


def extract_metrics_from_out(file_path):
    patterns = {
        "iteration": r"ITERATION\s*[:=]?\s*(\d+)",
        "accuracy": r"ACCURACY\s*[:=]?\s*([0-9]*\.?[0-9]+)",
        "flops": r"FLOPS\s*[:=]?\s*([0-9]*\.?[0-9]+)",
        "params": r"PARAMS\s*[:=]?\s*([0-9]*\.?[0-9]+)",
        "latency": r"LATENCY\s*[:=]?\s*([0-9]*\.?[0-9]+)",
        "total_cost": r"TOTAL COST\s*[:=]?\s*([0-9]*\.?[0-9]+)",
        "score": r"score\s*[:=]?\s*([0-9]*\.?[0-9]+)",
    }
    rows = []
    current = {key: None for key in patterns.keys()}
    previous = {key: None for key in patterns.keys()}
    with open(file_path, "r") as file:
        for line in file:
            for key, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    # print(f"Found {key} in line: {line.strip()}")
                    current[key] = float(match.group(1)) if key != "iteration" else int(match.group(1))
                    # print(f"Current {key}: {current[key]}")
            if current["iteration"] is not None and current["score"] is not None:
                # print(f"Iteration: {current['iteration']}, Accuracy: {current['accuracy']:.4f}, "
                #       f"FLOPS: {current['flops']}, Params: {current['params']}, "
                #       f"Latency: {current['latency']}, Total Cost: {current['total_cost']}")
                if "ITERATION" in line:
                    rows.append(current.copy())
                    current = {key: None for key in patterns.keys()}
    return pd.DataFrame(rows)


def extract_optimization_info(file_path):
    best_opt = None
    best_acc = float('-inf')
    current_opt = None
    da = False
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            # detect the "punto scelto" line (case‑insensitive)
            if 'data_augmentation' in line.lower():
                # check if data augmentation is enabled
                da = 'True'
            if 'punto scelto' in line.lower():
                # store everything that appears after the first colon
                current_opt = line.split(':', 1)[1].strip()
            # detect the ACCURACY line that always follows the chosen point
            elif current_opt is not None and line.startswith('ACCURACY:'):
                # extract the first float in the line
                match = re.search(r'([-+]?\d*\.\d+|\d+)', line)
                if match:
                    acc_val = float(match.group(0))
                    # update the best pair found so far
                    if acc_val > best_acc:
                        best_acc = acc_val
                        best_opt = current_opt
                # reset for the next block
                current_opt = None

        print("Miglior 'punto scelto' trovato con ACCURACY massima:")
        print("-" * 80)
        print(best_opt)
        print(f"\nACCURACY massima: {best_acc:.6f}")
        d = ast.literal_eval(best_opt)

        # Estrae l'ottimizzatore
        optimizer = d.get('optimizer')
        print(f"Optimizer estratto: {optimizer}")
        batch_size = d.get('batch_size', 32)  # Default batch size if not specified
        lr = d.get('learning_rate', '0.001')  # Default learning rate if not specified
        return optimizer, lr, batch_size, da

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
        

    plt.xlabel('Iterazione')
    plt.ylabel(label)
    plt.title(f'Andamento di {label}')
    plt.legend()
    plt.grid(True)
    
def plot_metrics_and_print_bests(csv_path: str,
                                 dir: str) -> None:
    """
    Plotta accuracy (e altre metriche se specificate) vs iteration
    e stampa le righe 'best' per ogni parametro.

    Parameters
    ----------
    csv_path : str
        Percorso del file .csv da analizzare.
    """
    add_metrics = ["flops", "params", "latency", "total_cost"]
    df = pd.read_csv(csv_path)

    # --------- PLOT ---------------------------------------------------------
    metrics_to_plot = ['accuracy'] + [m for m in add_metrics if m in df.columns]
    plt.figure(figsize=(30, 7*len(metrics_to_plot)))
    for n, m in enumerate(metrics_to_plot):
        best = 'min' if m != 'accuracy' else 'max'
        plot_with_trend(df['iteration'], df[m], m, 'b', n+1, len(metrics_to_plot)+1, best=best)

    plt.tight_layout()
    plt.savefig(os.path.join(dir, "output_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --------- BEST ROWS ----------------------------------------------------
    best_rows = {}

    # accuracy -> valore massimo
    if 'accuracy' in df.columns:
        best_rows['accuracy'] = df.loc[df['accuracy'].idxmax()]

    # altre metriche -> valore minimo (più basso è meglio)
    for m in ['flops', 'params', 'latency', 'total_cost']:
        if m in df.columns and not df[m].isnull().all():
            best_rows[m] = df.loc[df[m].idxmin()]

    # stampa ordinata
    for metric, row in best_rows.items():
        print(f"\n=== Best {metric.upper()} ===")
        print("Iteration:{}/{}".format(row['iteration'], len(df)))
        print("Accuracy:", row['accuracy'])
        for m in add_metrics:
            if m in row:
                print(f"{m.capitalize()}: {row[m]:,.5f}")
        

def evaluate_net(path, file):
    from flops import flops_calculator as fc
    DATA_NAME = path.split('_')[6]
    if DATA_NAME == "MNIST":
        X_train, X_test, Y_train, Y_test, n_classes = mnist()
    elif DATA_NAME == "CIFAR10":
        X_train, X_test, Y_train, Y_test, n_classes = cifar_data()
    elif DATA_NAME == "CIFAR100":
        X_train, X_test, Y_train, Y_test, n_classes = cifar_data_100()
    elif DATA_NAME == "gesture":
        print("ERRORE: gesture dataset not supported in continue training")
    else:
        print(colors.FAIL, "|  ----------- DATASET NOT FOUND ----------  |\n", colors.ENDC)
        sys.exit()
    
    opt = extract_optimization_info(path, file)
    if not opt:
        print(colors.FAIL, "No optimization info found in file:", file, colors.ENDC)
        return
    model = tf.keras.models.load_model(os.path.join(path, "Model", "best-model.keras"))
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    import io
    summary_str = io.StringIO()
    model.summary(print_fn=lambda x: summary_str.write(x + '\n'))
    # print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    
    score = model.evaluate(X_test, Y_test, verbose=0)

    gpus = tf.config.list_physical_devices('GPU')
    time_gpu = "0:00:00.0"
    if len(gpus) > 0:
        start = datetime.now()
        score = model.evaluate(X_test, Y_test, verbose=0)
        time_gpu = (datetime.now() - start) / X_test.shape[0]
    with tf.device('/CPU:0'):
        start = datetime.now()
        score = model.evaluate(X_test, Y_test, verbose=0)
        time_cpu = (datetime.now() - start) / X_test.shape[0]
    flops, r_dict = fc.analyze_model(model)
    trainableParams = np.sum([np.prod(v.shape)for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.shape)for v in model.non_trainable_weights])
    nparams = trainableParams + nonTrainableParams
    return [time_gpu, time_cpu], score, flops.total_float_ops, nparams