import os
import sys

# Add project root to sys.path so package imports work when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tensorflow.keras import backend as K
from components.dataset import TunerDataset
from tensorflow_implementation import module_backend, neural_network
from components.tuner import Tuner, TunerConfig


os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def build_config() -> TunerConfig:
    dataset = TunerDataset()
    dataset.cifar_data()

    return TunerConfig(
        neural_network_cls=neural_network.NeuralNetwork,
        module_backend_cls=module_backend.ModuleBackend,
        dataset=dataset,
        clear_session_callback=K.clear_session
    )


def main():
    tuner = Tuner(build_config())
    tuner.run()


if __name__ == "__main__":
    main()
