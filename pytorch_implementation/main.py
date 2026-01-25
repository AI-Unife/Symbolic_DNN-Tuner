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
    dataset.load_cifar_10()

    config = TunerConfig(
        neural_network_cls=neural_network.NeuralNetwork,
        module_backend_cls=module_backend.ModuleBackend,
        dataset=dataset,
        fixed_hyperparams={'unit_c1': 64, 'dr1_2': 0.16765775684603082, 'unit_c2': 91, 'unit_d': 490, 'dr_f': 0.27073931298122383, 'learning_rate': 0.0004832002326171084, 'batch_size': 197, 'optimizer': 'Adamax', 'activation': 'relu'}
    )
    tuner = Tuner(config)
    tuner.run()


if __name__ == "__main__":
    main()
