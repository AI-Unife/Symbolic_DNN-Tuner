from __future__ import annotations
from typing import Callable, Optional, Sequence
import numpy as np
from skopt.space import Space
from skopt.utils import eval_callbacks, create_result


class RandomSearch():
    """
    Perform a random search over the given search space.

    Parameters:
    - search_space: The search space to explore.
    - objective_fn: The function to minimize.
    - n_calls: Number of evaluations to perform.
    - random_state: Random state for reproducibility.
    - callback: Optional callback function to evaluate after each call.

    Returns:
    - The best score found during the search.
    """

    def __init__(self, random_state=None, total_iter=1):
        self.random_state = random_state
        self.Xi = []
        self.Yi = []
        self.actual_iter = 0
        self.total_iter = total_iter + 1

    def __call__(self, objective_fn, search_space, callback=None):
        # rng = np.random.default_rng(random_state)
        best_score = float("inf")
        best_params = None
        params = None

        # Generate a random set of parameters from the search space
        params = search_space.rvs(n_samples=self.total_iter, random_state=self.random_state)[self.actual_iter]
        score = objective_fn(params)
        self.actual_iter += 1

        if score < best_score:
            best_score = score
            best_params = params

        self.Xi.extend([best_params])
        self.Yi.extend([best_score])

        res = create_result(
            Xi=self.Xi,
            yi=self.Yi,
            space=search_space,
            models=None,
            rng=self.random_state
        )

        if callback:
            if eval_callbacks(callback, res):
                return res
        return res