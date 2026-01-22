import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Type, Union

from skopt import gp_minimize, load
from skopt.space import Integer, Real
from skopt.callbacks import CheckpointSaver

from components.colors import colors
from components.controller import controller
from components.search_space import search_space as SearchSpace
from components.neural_network import NeuralNetwork
from components.backend_interface import BackendInterface
from components.dataset import TunerDataset


def _default_search_space() -> Sequence[Any]:
    return SearchSpace().search_sp()


def _default_folders() -> List[str]:
    return [
        "Model",
        "Weights",
        "database",
        "checkpoints",
        "log_folder",
        "algorithm_logs",
        "dashboard/model",
    ]


def _default_gp_kwargs() -> Dict[str, Any]:
    return {
        "acq_func": "EI",
        "n_calls": 1,
        "n_random_starts": 1,
    }


@dataclass
class TunerConfig:
    """
    Configuration payload used to bootstrap the tuner with backend specific components.
    """
    neural_network_cls: Type[NeuralNetwork]
    module_backend_cls: Type[BackendInterface]
    dataset: TunerDataset
    search_space: Optional[Sequence[Any]] = None
    max_evals: int = 1
    checkpoint_path: str = "checkpoints/checkpoint.pkl"
    objective_log_path: str = "algorithm_logs/hyper-neural.txt"
    clear_session_callback: Optional[Callable[[], None]] = None
    gp_minimize_kwargs: Dict[str, Any] = field(default_factory=_default_gp_kwargs)
    folders_to_create: Iterable[str] = field(default_factory=_default_folders)
    fixed_hyperparams: Optional[Dict[str, Any]] = None


class Tuner:
    """
    Framework agnostic tuner orchestrator. Backend specific projects
    should instantiate this class, provide their neural network and module backend
    implementations, and call `run`.
    """

    def __init__(self, config: TunerConfig):
        self.config = config
        self._ensure_directories()

        self.dataset = self.config.dataset

        self.neural_network = self.config.neural_network_cls(self.dataset)
        self.module_backend = self.config.module_backend_cls()

        self.controller = controller(
            self.neural_network,
            self.module_backend,
            self.dataset,
            clear_session_callback=self.config.clear_session_callback,
        )

        self.gp_kwargs = {**_default_gp_kwargs(), **self.config.gp_minimize_kwargs}
        self._start_time: Optional[float] = None

        if self.config.search_space is not None:
            self.search_space = self.config.search_space
        else:
            self.search_space = _default_search_space()

        self.current_search_space = self.search_space

    def run(self, max_evals: Optional[int] = None):
        """
        Execute Bayesian optimisation and symbolic diagnosis loop.
        :param max_evals: optional override for number of diagnosis iterations.
        :return: skopt optimization result.
        """
        iterations = self.config.max_evals if max_evals is None else max_evals
        print(colors.OKGREEN, "\nSTART ALGORITHM \n", colors.ENDC)

        self._start_time = time.time()
        result = self._start(self.current_search_space, iterations)

        print(result)
        print(colors.OKGREEN, "\nEND ALGORITHM \n", colors.ENDC)

        if self._start_time is not None:
            elapsed = time.time() - self._start_time
            print(colors.CYAN, "\nTIME --------> \n", elapsed, colors.ENDC)

        self.controller.plotting_obj_function()
        self.controller.save_experience()

        return result

    def _ensure_directories(self):
        for folder in self.config.folders_to_create:
            try:
                os.makedirs(folder, exist_ok=True)
            except OSError:
                print(colors.FAIL, "|  ----------- FAILED TO CREATE FOLDER ----------  |\n", colors.ENDC)

    def _objective(self, params):
        space = {}
        for dim, value in zip(self.current_search_space, params):
            if isinstance(dim, Integer):
                space[dim.name] = int(value)
            elif isinstance(dim, Real):
                space[dim.name] = float(value)
            else:  # Categorical, etc.
                space[dim.name] = value

        if self.config.fixed_hyperparams:
            space.update(self.config.fixed_hyperparams)

        print(space)
        with open(self.config.objective_log_path, "a") as handle:
            handle.write(str(space) + "\n")

        to_optimize = self.controller.training(space)

        if self.config.clear_session_callback:
            self.config.clear_session_callback()

        return to_optimize

    def _start_analysis(self):
        """
        Run symbolic diagnosis and return the updated search space with the corresponding value.
        """
        return self.controller.diagnosis()

    def _check_continuing_bo(self, new_space, x_iters, func_vals):
        x_iters = list(x_iters)
        func_vals = func_vals.tolist()
        for x in list(x_iters):
            for dim, value in zip(new_space, x):
                if value < dim.low or value > dim.high:
                    index = x_iters.index(x)
                    func_vals.pop(index)
                    x_iters.remove(x)
                    break
        return x_iters, func_vals

    def _start(self, search_space, iterations):
        """
        Execute the optimisation loop.
        """
        print(colors.MAGENTA, "|  ----------- START BAYESIAN OPTIMIZATION ----------  |\n", colors.ENDC)

        checkpoint_saver = CheckpointSaver(self.config.checkpoint_path, compress=9)

        self.controller.set_case(False)
        self.current_search_space = search_space

        search_res = gp_minimize(
            self._objective,
            search_space,
            callback=[checkpoint_saver],
            **self.gp_kwargs,
        )

        new_space, to_optimize = self._start_analysis()

        for _ in range(iterations):
            if len(new_space) == len(self.current_search_space):
                res = load(self.config.checkpoint_path)

                try:
                    search_res = gp_minimize(
                        self._objective,
                        new_space,
                        x0=res.x_iters,
                        y0=res.func_vals,
                        callback=[checkpoint_saver],
                        acq_func=self.gp_kwargs.get("acq_func", "EI"),
                        n_calls=self.gp_kwargs.get("n_calls", 1),
                        n_random_starts=0,
                    )
                    print(colors.WARNING, "-----------------------------------------------------", colors.ENDC)
                    print(colors.FAIL, "Inside BO", colors.ENDC)
                    print(colors.WARNING, "-----------------------------------------------------", colors.ENDC)
                except Exception:
                    x_iters, func_vals = self._check_continuing_bo(new_space, res.x_iters, res.func_vals)
                    search_res = gp_minimize(
                        self._objective,
                        new_space,
                        y0=func_vals,
                        callback=[checkpoint_saver],
                        acq_func=self.gp_kwargs.get("acq_func", "EI"),
                        n_calls=self.gp_kwargs.get("n_calls", 1),
                        n_random_starts=self.gp_kwargs.get("n_random_starts", 1),
                    )
                    print(colors.FAIL, "-----------------------------------------------------", colors.ENDC)
                    print(colors.WARNING, "Other BO", colors.ENDC)
                    print(colors.FAIL, "-----------------------------------------------------", colors.ENDC)
            else:
                self.current_search_space = new_space
                search_res = gp_minimize(
                    self._objective,
                    new_space,
                    callback=[checkpoint_saver],
                    acq_func=self.gp_kwargs.get("acq_func", "EI"),
                    n_calls=self.gp_kwargs.get("n_calls", 1),
                    n_random_starts=self.gp_kwargs.get("n_random_starts", 1),
                )

            new_space, to_optimize = self._start_analysis()

        return search_res
