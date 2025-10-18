import os

from tensorflow.keras import backend as K

from components.dataset import cifar_data
from tensorflow_implementation import module_backend, neural_network
from tuner import Tuner, TunerConfig

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def build_config() -> TunerConfig:
    return TunerConfig(
        neural_network_cls=neural_network.NeuralNetwork,
        module_backend_cls=module_backend.ModuleBackend,
        dataset_loader=cifar_data,
        clear_session_callback=K.clear_session,
    )


def main():
    tuner = Tuner(build_config())
    tuner.run()


if __name__ == "__main__":
    main()
