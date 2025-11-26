import tensorflow as tf
from components.dataset import get_datasets
import components.neural_network

# Carica i modelli
model_original = tf.keras.models.load_model('/Users/osamaabdouh/Downloads/results/25_10_21_12_49650_gesture_filtered_100_30_fwdPass_32_2_2/Model/best-model.keras')
model_finetuned = tf.keras.models.load_model('best-model-finetuned.keras')

# Carica il dataset
dataset_name = "gesture"
X_train, Y_train, X_test, Y_test, n_classes = get_datasets(dataset_name)

# Confronta le performance
print("="*60)
print("CONFRONTO PRE vs POST FINE-TUNING")
print("="*60)

# Modello originale
loss_orig, acc_orig = components.neural_network.eval_model(model_original, X_test, Y_test)
print(f"\n📊 MODELLO ORIGINALE:")
print(f"   Loss: {loss_orig:.4f}")
print(f"   Accuracy: {acc_orig:.4f}")

# Modello fine-tuned
loss_ft, acc_ft = components.neural_network.eval_model(model_finetuned, X_test, Y_test)
print(f"\n📊 MODELLO FINE-TUNED:")
print(f"   Loss: {loss_ft:.4f}")
print(f"   Accuracy: {acc_ft:.4f}")

# Differenze
print(f"\n📈 MIGLIORAMENTO:")
print(f"   Δ Loss: {loss_ft - loss_orig:+.4f}")
print(f"   Δ Accuracy: {acc_ft - acc_orig:+.4f} ({(acc_ft - acc_orig)/acc_orig*100:+.2f}%)")

# Verifica che i pesi siano diversi
print(f"\n🔍 VERIFICA PESI:")
weights_changed = False
for layer_orig, layer_ft in zip(model_original.layers, model_finetuned.layers):
    if len(layer_orig.get_weights()) > 0:
        w_orig = layer_orig.get_weights()[0]
        w_ft = layer_ft.get_weights()[0]
        if not np.allclose(w_orig, w_ft):
            weights_changed = True
            print(f"   ✓ Layer '{layer_orig.name}': pesi modificati")
            break

if weights_changed:
    print("   ✅ I pesi sono stati aggiornati dal fine-tuning")
else:
    print("   ⚠️  ATTENZIONE: I pesi sembrano identici!")

print("="*60)