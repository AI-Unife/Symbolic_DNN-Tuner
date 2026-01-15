import sys
import os
from pathlib import Path
import questionary

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import components.neural_network
from components.colors import colors
from exp_config import load_cfg
from components.dataset import get_datasets
from components.custom_train import eval_model

def fine_tune_model(model_path):
    """Fine-tune a pre-trained model"""

    cfg = load_cfg()

    X_train, Y_train, X_test, Y_test, n_classes = get_datasets(cfg.dataset)

    model = tf.keras.models.load_model(model_path)

    def do_eval(tag: str):
        if (cfg.mode == "fwdPass" or cfg.mode == "hybrid") and cfg.dataset == "gesture":
            score = eval_model(model, X_test, Y_test)
        else:
            score = model.evaluate(X_test, Y_test, verbose=0)
        print(f"{tag} - Loss: {score[0]:.4f}, Accuracy: {score[1]:.4f}")
        return score

    print(model.optimizer.get_config())

    es1 = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=1,
                            mode="min", restore_best_weights=True)
    es2 = EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=10, verbose=1,
                            mode="max", restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, verbose=1,
                            min_lr=1e-5)
    
    callbacks = [es1, es2, reduce_lr]

    params = {'batch_size': 32}

    history = None

    before = do_eval("Before fine-tuning")

    if (cfg.mode == "fwdPass" or cfg.mode == "hybrid") and cfg.dataset == "gesture":
        print(colors.OKGREEN + "Modalità 'hybrid' o 'fwdPass' rilevata. Uso train_model custom." + colors.ENDC)

        opt = model.optimizer

        if opt is None:
            print(colors.FAIL + "Nessun ottimizzatore trovato nel modello caricato." + colors.ENDC)
            return


        if hasattr(opt, '_learning_rate') and opt._learning_rate is None:
            try:
                # Extract the LR (as a numpy value) from the internal optimizer
                # to prevent crashes in apply_gradients().
                inner_lr_value = opt._optimizer.learning_rate.numpy()
                
                print(colors.WARNING + f"ATTENZIONE: opt._learning_rate (wrapper) è None." + colors.ENDC)
                print(colors.WARNING + f"Ripristino il valore leggendolo dall'ottimizzatore interno (Adam): {inner_lr_value}" + colors.ENDC)
                
                opt._learning_rate = inner_lr_value
                
            except Exception as e:
                print(colors.FAIL + f"Errore nel tentativo di patchare il LR. Uso un default. Errore: {e}" + colors.ENDC)
                return
        history = components.neural_network.train_model(model, opt, X_train, Y_train, X_test, Y_test, cfg.epochs, params, callbacks)
    else:
        print("Modalità standard rilevata. Uso model.fit di Keras.")

        opt = model.optimizer

        if opt is None:
            print(colors.FAIL + "Nessun ottimizzatore trovato nel modello caricato." + colors.ENDC)
            return

        if hasattr(opt, '_learning_rate') and opt._learning_rate is None:
            try:
                # Extract the LR (as a numpy value) from the internal optimizer
                # to prevent crashes in apply_gradients().
                inner_lr_value = opt._optimizer.learning_rate.numpy()
                
                print(colors.WARNING + f"ATTENZIONE: opt._learning_rate (wrapper) è None." + colors.ENDC)
                print(colors.WARNING + f"Ripristino il valore leggendolo dall'ottimizzatore interno (Adam): {inner_lr_value}" + colors.ENDC)
                
                opt._learning_rate = inner_lr_value
                
            except Exception as e:
                print(colors.FAIL + f"Errore nel tentativo di patchare il LR. Uso un default. Errore: {e}" + colors.ENDC)
                return

        history = model.fit(X_train, Y_train,
                            epochs=cfg.epochs,
                            batch_size=int(params['batch_size']),
                            verbose=1,
                            validation_data=(X_test, Y_test),
                            callbacks=callbacks).history
        
    print("Fine-tuning completed.")

    after = do_eval("After fine-tuning")
    print(f"Delta acc: {after[1] - before[1]:+.4f}")
    print("Last val_acc:", history.get("val_accuracy", [])[-1])
    print("Last val_loss:", history.get("val_loss", [])[-1])

    out_dir = Path(cfg.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        model.save(str(out_dir / "best-model-finetuned.keras"))
    except:
        pass

    # Save the training history if available
    if history is not None:
        import json

        hystory_clean = {}
        h_dict = history if isinstance(history, dict) else history.history
        for key, val in h_dict.items():
            hystory_clean[key] = [float(v) for v in val]
            
        hystory_path = out_dir / "training_history.json"
        with open(hystory_path, "w") as f:
            json.dump(hystory_clean, f, indent=2)
    print(f"Model and training history saved to {out_dir}")