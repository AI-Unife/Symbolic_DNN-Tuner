import json
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import os
from components.gesture_dataset import gesture_data

base_path = "25_02_25_11_04_GESTURE_accuracy_module_hardware_module_flops_module_500_30/"

model_path = base_path + "Model/best-model.json"
weights_path = base_path + "dashboard/model/model.keras" #Weights/best-weights.h5"

# # Controlla se il file esiste e non è vuoto
# if not os.path.exists(model_path) or os.stat(model_path).st_size == 0:
#     raise ValueError("Errore: Il file JSON è vuoto o non esiste.")

# # Legge il file JSON
# with open(model_path, "r") as f:
#     json_data = f.read()

# # Verifica che il JSON sia valido
# try:
#     model_config = json.loads(json_data)  # Decodifica JSON
# except json.JSONDecodeError as e:
#     raise ValueError(f"Errore nel parsing JSON: {e}")

# # Carica il modello da JSON
# model = model_from_json(json_data)
# print(model.summary())
# import h5py


# # with h5py.File(weights_path, "r") as f:
# #     print("Chiavi nel file:", list(f.keys()))
# #     for k in f.keys():
# #         print(f[k])
# #         for e in f[k]:
# #             print(e)
model = tf.keras.models.load_model(weights_path, compile=False)
print("Modello caricato con successo!")
_, X_test, _, Y_test, n_classes = gesture_data()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.evaluate(X_test, Y_test, verbose=1))
print("Fine")
