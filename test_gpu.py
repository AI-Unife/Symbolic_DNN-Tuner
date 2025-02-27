import tensorflow as tf

# Verifica della disponibilità della GPU
device_name = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
print(f"Using device: {device_name}")
