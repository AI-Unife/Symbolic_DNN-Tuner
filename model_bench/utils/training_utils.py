"""
Training utility function for model fine-tuning.
"""
from pathlib import Path
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import components.neural_network
from utils.model_utils import load_model_dataset
from components.colors import colors
from exp_config import load_cfg
from components.custom_train import eval_model

def fine_tune_model(model_path):
    """Fine-tune a pre-trained model"""

    cfg = load_cfg()

    model, X_train, Y_train, X_test, Y_test, n_classes = load_model_dataset(model_path)

    # Ensure that label are one-hot encoded for evaluation for gestture dataset in depth mode
    if cfg.dataset == "gesture" and cfg.mode == "depth":
        if len(Y_train.shape) == 1 or (len(Y_train.shape) == 2 and Y_train.shape[1] == 1):
            import tensorflow as tf
            print(colors.WARNING + "Converting labels to one-hot encoding..." + colors.ENDC)
            Y_train = tf.keras.utils.to_categorical(Y_train, n_classes)
            Y_test = tf.keras.utils.to_categorical(Y_test, n_classes)

    def do_eval(tag: str):
        if (cfg.mode == "fwdPass" or cfg.mode == "hybrid") and cfg.dataset == "gesture":
            score = eval_model(model, X_test, Y_test)
        else:
            score = model.evaluate(X_test, Y_test, verbose=0)
        print(f"\n{tag} - Loss: {score[0]:.4f}, Accuracy: {score[1]:.4f}")
        return score

    es1 = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=10, verbose=1,
                            mode="min", restore_best_weights=True)
    es2 = EarlyStopping(monitor="val_accuracy", min_delta=0.001, patience=10, verbose=1,
                            mode="max", restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, verbose=1,
                            min_lr=1e-5)
    
    callbacks = [es1, es2, reduce_lr]
    params = {'batch_size': cfg.batch_size}
    history = None
    
    print(colors.OKBLUE + "|  --------- START FINE-TUNING --------  |" + colors.ENDC)
    before = do_eval("Before fine-tuning")

    opt = model.optimizer
    if opt is None:
        print(colors.FAIL + "No optimizer found in the loaded model." + colors.ENDC)
        return  
    # workaround for custom optimizer wrapper issue
    if hasattr(opt, '_learning_rate') and opt._learning_rate is None:
        try:
            # Extract the LR (as a numpy value) from the internal optimizer
            # to prevent crashes in apply_gradients().
            inner_lr_value = opt._optimizer.learning_rate.numpy()
            
            print(colors.WARNING + f"WARNING: opt._learning_rate (wrapper) is None." + colors.ENDC)
            print(colors.WARNING + f"Restoring value by reading from internal optimizer (Adam): {inner_lr_value}\n" + colors.ENDC)
            
            opt._learning_rate = inner_lr_value
        except Exception as e:
            print(colors.FAIL + f"Error attempting to patch LR. Error: {e}" + colors.ENDC)
            return

    if (cfg.mode == "fwdPass" or cfg.mode == "hybrid") and cfg.dataset == "gesture":
        print("mode 'hybrid' or 'fwdPass' detected. Using custom train_model.")
        history = components.neural_network.train_model(model, opt, X_train, Y_train, X_test, Y_test, cfg.epochs, params, callbacks)
    else:
        print("Standard mode detected. Using Keras model.fit.")
        history = model.fit(X_train, Y_train,
                            epochs=cfg.epochs,
                            batch_size=int(params['batch_size']),
                            verbose=1,
                            validation_data=(X_test, Y_test),
                            callbacks=callbacks).history
        
    print(colors.OKGREEN + "Fine-tuning completed.\n" + colors.ENDC)
    
    after = do_eval("After fine-tuning")
    print(f"Delta acc: {after[1] - before[1]:+.4f}")
    print("Last val_acc:", history.get("val_accuracy", [])[-1])
    print("Last val_loss:", history.get("val_loss", [])[-1])

    print(colors.OKBLUE + "|  ------------------------------------  |" + colors.ENDC)


    timestamp = datetime.now().strftime("%Y_%m_%d__%H%M")
    base_dir = Path(__file__).parent.parent / "exports" / "fine_tuned_models"
    out_dir = base_dir / f"{timestamp}_{cfg.name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        model.save(str(out_dir / "finetuned-model.keras"))
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
    print(colors.OKGREEN + f"\nModel and training history saved to:\n" + colors.ENDC + f"{out_dir}")
