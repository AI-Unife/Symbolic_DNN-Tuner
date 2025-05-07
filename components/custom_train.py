import tensorflow as tf
import scipy
from tqdm import tqdm

def accuracy(outputs, targets):
    """
    Computes classification accuracy by selecting the most frequent predicted class.
    """
    output_frames = tf.argmax(outputs, axis=2).numpy()
    mode, _ = scipy.stats.mode(output_frames, axis=1, keepdims=False)
    output_preds = tf.cast(mode, tf.int64)

    target_frames = tf.argmax(targets, axis=2).numpy()
    target_mode, _ = scipy.stats.mode(target_frames, axis=1, keepdims=False)

    return tf.reduce_mean(tf.cast(output_preds == target_mode, tf.float64)).numpy()


def train_model(model, optimizer, train_data, train_labels, test_data, test_labels, epochs, params, callbacks):
    """
    Custom training function replacing model.fit(), but keeping callbacks and returning history.
    """
    # Prepara il data generator
    train_loader = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(params['batch_size'])#.shuffle(100).prefetch(tf.data.AUTOTUNE)

    # datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    # train_loader = datagen.flow(train_data, train_labels, batch_size=params['batch_size'])

    # Definizione della loss e dell'ottimizzatore
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    # optimizer = tf.keras.optimizers.Adam()

    # Inizializzazione history
    history = {key: [] for key in ["loss", "accuracy", "val_loss", "val_accuracy"]}

    # Inizializzazione delle callback
    for callback in callbacks:
        callback.set_model(model)
        callback.on_train_begin()

    num_epochs = epochs

    for epoch in range(num_epochs):
        epoch_loss, epoch_acc, n_batches = 0, 0, 0

        # Avvio callbacks per l'epoca
        for callback in callbacks:
            callback.on_epoch_begin(epoch)
        print(f"Epoch {epoch + 1}/{num_epochs}")
        # Training loop
        for data, targets in tqdm(train_loader):
            batch_size, time_steps, *input_shape = data.shape
            outputs_list = []

            with tf.GradientTape() as tape:
                for t in range(time_steps):
                    frame_outputs = model(data[:, t], training=True)
                    outputs_list.append(frame_outputs)

                outputs = tf.stack(outputs_list, axis=1)
                # print("target shape: ", targets.shape)
                # print("output shape: ", outputs.shape)
                loss = loss_fn(targets, outputs)

            # Calcolo gradienti e aggiornamento pesi
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Aggiorna metriche
            epoch_loss += loss.numpy()
            epoch_acc += accuracy(outputs, targets)
            n_batches += 1

        # Calcola loss e accuracy medie dell'epoca
        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches
        history["loss"].append(avg_loss)
        history["accuracy"].append(avg_acc)

        # Validazione
        val_outputs_list = []
        for t in range(test_data.shape[1]):
            val_frame_outputs = model(test_data[:, t], training=False)
            val_outputs_list.append(val_frame_outputs)

        val_outputs = tf.stack(val_outputs_list, axis=1)
        val_loss = loss_fn(test_labels, val_outputs).numpy()
        val_acc = accuracy(val_outputs, test_labels)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        # Logging
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {avg_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        # Avvio callbacks per la fine dell'epoca
        logs = {"loss": avg_loss, "accuracy": avg_acc, "val_loss": val_loss, "val_accuracy": val_acc}
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs)

    # Callback per la fine dell'addestramento
    for callback in callbacks:
        callback.on_train_end()

    return history  # Stesso formato di model.fit().history


def eval_model(model, X, y):
    val_outputs_list = []
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    for t in range(X.shape[1]):
        val_frame_outputs = model(X[:, t], training=False)
        val_outputs_list.append(val_frame_outputs)

    val_outputs = tf.stack(val_outputs_list, axis=1)
    val_loss = loss_fn(y, val_outputs).numpy()
    val_acc = accuracy(val_outputs, y)
    
    return val_loss, val_acc

        