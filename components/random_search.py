from __future__ import annotations
from typing import Callable, Optional, Sequence
import numpy as np
from skopt.space import Space
from skopt.utils import eval_callbacks, create_result

class RandomSearch:
    """
    Minimal random search with persistent RNG, de-duplication and skopt-compatible result.
    """
    def __init__(self, random_state: Optional[int] = None, total_iter: int = 1):
        self.random_state = random_state
        self._rng = (np.random.RandomState(random_state)
                     if not isinstance(random_state, np.random.RandomState)
                     else random_state)
        self.Xi, self.Yi = [], []
        self._seen = set()           # keep hashes of sampled points to avoid duplicates
        self.actual_iter = 0
        self.total_iter = int(total_iter)

    def __call__(self,
                 objective_fn: Callable[[list], float],
                 search_space: Space,
                 callback: Optional[Sequence[Callable]] = None):

        # Stop if we've hit the quota
        if self.actual_iter >= self.total_iter:
            res = create_result(self.Xi, self.Yi, search_space, models=None, rng=self.random_state)
            if callback and eval_callbacks(callback, res):
                return res
            return res

        # Sample until we get a point not seen before (bounded tries)
        max_tries = 50
        for _ in range(max_tries):
            # IMPORTANT: use the persistent RNG; do NOT pass random_state each time
            params = search_space.rvs(n_samples=1, random_state=self._rng)[0]
            key = tuple(params) if not isinstance(params, dict) else tuple(params[k.name] for k in search_space.dimensions)
            if key not in self._seen:
                self._seen.add(key)
                break
        else:
            # If we keep hitting duplicates, just proceed with the last sampled params
            pass

        # Evaluate
        try:
            score = float(objective_fn(params))
        except Exception:
            score = float("inf")

        # Update history with the CURRENT trial (not only the best)
        self.Xi.append(params)
        self.Yi.append(score)
        self.actual_iter += 1

        # Build skopt-like result
        res = create_result(self.Xi, self.Yi, search_space, models=None, rng=self.random_state)

        # Callbacks
        if callback and eval_callbacks(callback, res):
            return res
        return res
