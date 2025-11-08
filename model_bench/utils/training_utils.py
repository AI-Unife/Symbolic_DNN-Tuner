import sys
import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import components.neural_network
from exp_config import load_cfg
from components.dataset import get_datasets

def fine_tune_model(model_path):
    """Fine-tune a pre-trained model"""

    cfg = load_cfg()
    print("Fine-tuning model with the following configuration:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")

    dataset_name = cfg.dataset.strip().lower().replace("-", "")

    X_train, Y_train, X_test, Y_test, n_classes = get_datasets(dataset_name)

    model = tf.keras.models.load_model('/Users/osamaabdouh/Downloads/results/25_10_21_12_49650_gesture_filtered_100_30_fwdPass_32_2_2/Model/best-model.keras')

    print(model.optimizer.get_config())

    es1 = EarlyStopping(monitor="val_loss", min_delta=0.005, patience=20, verbose=1,
                            mode="min", restore_best_weights=True)
    es2 = EarlyStopping(monitor="val_accuracy", min_delta=0.005, patience=20, verbose=1,
                            mode="max", restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=30, verbose=1,
                            min_lr=1e-4)
    
    callbacks = [es1, es2, reduce_lr]

    params = {'batch_size': 32}

    history = None
    epochs = 200

    if (cfg.mode == "fwdPass" or cfg.mode == "hybrid") and cfg.dataset == "gesture":
        print("Modalità 'hybrid' o 'fwdPass' rilevata. Uso train_model custom.")

        opt = model.optimizer

        if opt is None:
            print("Nessun ottimizzatore trovato nel modello caricato.")
            return
        
        # Il problema è che opt._learning_rate (del wrapper) è None dopo il caricamento.
        # Il learning rate che vogliamo (0.000211...) è salvato dentro 
        # l'ottimizzatore *interno* (Adam).
        
        if hasattr(opt, '_learning_rate') and opt._learning_rate is None:
            try:
                # Estraiamo il LR (come valore numpy) dall'ottimizzatore interno
                inner_lr_value = opt._optimizer.learning_rate.numpy()
                
                print(f"ATTENZIONE: opt._learning_rate (wrapper) è None.")
                print(f"Ripristino il valore leggendolo dall'ottimizzatore interno (Adam): {inner_lr_value}")
                
                # Assegniamo questo valore al campo _learning_rate del wrapper.
                # Questo previene il crash in apply_gradients.
                opt._learning_rate = inner_lr_value
                
            except Exception as e:
                print(f"Errore nel tentativo di patchare il LR. Uso un default. Errore: {e}")
                opt._learning_rate = 1e-4 # Fallback di emergenza

        history = components.neural_network.train_model(model, opt, X_train, Y_train, X_test, Y_test, cfg.epochs, params, callbacks)
    else:
        print("Modalità standard rilevata. Uso model.fit di Keras.")

        history = model.fit(X_train, Y_train,
                            epochs=cfg.epochs,
                            batch_size=int(params['batch_size']),
                            verbose=1,
                            validation_data=(X_test, Y_test),
                            callbacks=callbacks).history
        
    print("Fine-tuning completed.")

    # --- 8. VALUTAZIONE FINALE (Come suggerito dall'email) ---
    print("Valutazione del modello sul set di test...")
    
    # Assicurati di aver importato 'eval_model' da 'components.custom_train'
    test_loss, test_acc = components.neural_network.eval_model(model, X_test, Y_test)

    print(f"--- RISULTATI FINALI SUL TEST SET ---")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")


    model.save("fine-tuning/best-model-finetuned.keras")


    input("Press Enter to continue...")