import os
import sys

# Add project root to sys.path so package imports work when running as a script
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pytorch_implementation import module_backend, neural_network
from components.tuner import Tuner, TunerConfig
from components.dataset import TunerDataset


def main():

    dataset = TunerDataset()
    dataset.cifar_data()

    config = TunerConfig(
        neural_network_cls=neural_network.NeuralNetwork,
        module_backend_cls=module_backend.ModuleBackend,
        dataset=dataset
    )
    tuner = Tuner(config)
    tuner.run()


if __name__ == "__main__":
    main()
