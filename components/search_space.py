from __future__ import annotations

from typing import Dict, Iterable, Tuple, Optional, List, Any
import copy
import numpy as np
from skopt.space import Integer, Real, Categorical, Space


class search_space:
    """
    Define and manage the hyperparameter search space used during iterative tuning.

    Notes:
    - Public API preserved (same class/method names & signatures).
    - We operate on `self.search_space` (an skopt Space) and manipulate dimensions by name.
    """

    def __init__(self) -> None:
        """
        Initialize helper epsilons used for building bounds of newly inserted dimensions.
        These are kept to preserve your original semantics.
        """
        self.epsilon_r1 = 10 ** -3
        self.epsilon_r2 = 10 ** 2
        self.epsilon_i = 2
        self.epsilon_d = 4

        # Will be set by search_sp(); defined here to be explicit.
        self.search_space: Space

    # ------------------------------------------------------------------ base --

    def search_sp(self) -> Space:
        """
        Define the base search space reflecting the backbone of the neural network.
        Returns the created `skopt.space.Space`.
        """
        self.search_space = Space([
            Integer(16, 64,  name='unit_c1'),
            Real(0.002, 0.3, name='dr1_2'),
            Integer(64, 128, name='unit_c2'),
            Integer(0, 2048, name='unit_d'),
            Real(0.03, 0.5,  name='dr_f'),
            Real(1e-4, 1e-3, name='learning_rate'),
            Integer(16, 64,  name='batch_size'),
            Categorical(['Adam', 'Adamax'], name='optimizer'),  # kept your reduced set
            Categorical(['relu', 'elu', 'selu', 'swish'], name='activation'),
        ])
        return self.search_space

    # -------------------------------------------------------------- counters --

    def count_initial_layers(self, params: Iterable[Any]) -> Tuple[int, int]:
        """
        Count the number of convolutional and dense layers implied by the *current* space.

        Args:
            params: any iterable of existing hyperparameter dimensions (kept for API;
                    we iterate their `.name` fields like your original code).

        Returns:
            (tot_conv, tot_fc) where:
                tot_conv counts conv sections inferred from 'unit_c*' (×2 as in your code),
                tot_fc counts dense layers inferred from 'unit_d'.
        """
        tot_conv = 0
        tot_fc = 0
        for p in params:
            # p may be a skopt Dimension; we only rely on `name`
            name = getattr(p, "name", str(p))
            if 'unit_c' in name:
                tot_conv += 2
            elif 'unit_d' in name:
                tot_fc += 1

        print("tot_conv: ", tot_conv)
        print("tot_fc: ", tot_fc)
        return tot_conv, tot_fc

    # --------------------------------------------------------- add / remove --

    def add_params(self, params: Dict[str, Any]) -> Space:
        """
        Add new hyperparameters to the search space.

        Args:
            params: dict of {param_name: exemplar_value}. The exemplar defines the type.

        Returns:
            Updated `skopt.space.Space` with appended dimensions.

        Behavior:
            - float exemplar -> Real(0, 0.1)
            - int exemplar:
                * name contains 'new_fc'  -> Integer(0, 2048)
                * name contains 'new_conv'-> Integer(0, 512)
                * otherwise               -> Integer(0, exemplar + epsilon_i)
            - If a param already exists by name, it won't be duplicated.
        """
        # Existing names to avoid duplicates
        existing = {dim.name for dim in self.search_space.dimensions}
        new_dims: List = []

        for name, value in params.items():
            if name in existing:
                # Skip silently to preserve original behavior (no duplicate dims)
                continue

            if isinstance(value, float):
                new_dim = Real(0, 0.1, name=name)
            elif isinstance(value, int):
                if 'new_fc' in name:
                    new_dim = Integer(0, 2048, name=name)
                elif 'new_conv' in name:
                    new_dim = Integer(0, 512, name=name)
                else:
                    new_dim = Integer(0, int(value) + int(self.epsilon_i), name=name)
            else:
                # Fallback: try to coerce numeric strings; otherwise skip
                try:
                    as_float = float(value)
                    new_dim = Real(0, 0.1, name=name) if (as_float % 1) else Integer(0, int(as_float) + int(self.epsilon_i), name=name)
                except Exception:
                    # Unsupported exemplar type; ignore to keep API non-breaking
                    continue

            new_dims.append(new_dim)

        if new_dims:
            self.search_space = Space(self.search_space.dimensions + new_dims)
        return self.search_space

    def remove_params(self, params: Dict[str, Any]) -> Space:
        """
        Remove hyperparameters from the search space by name.

        Args:
            params: dict where keys are the names to remove (values are ignored).

        Returns:
            Updated `skopt.space.Space` with the specified dimensions removed.
        """
        target_names = set(params.keys())
        kept_dims = [dim for dim in self.search_space.dimensions if dim.name not in target_names]
        self.search_space = Space(kept_dims)
        return self.search_space

    # ----------------------------------------------------------- inspectors --

    def count_layer(self, type: str) -> int:
        """
        Count how many dimensions whose name contains `type`.
        E.g., `count_layer('new_conv')`.
        """
        return sum(1 for hp in self.search_space.dimensions if type in hp.name)

    # -------------------------------------------------------------- copies --

    def get_copy(self, space: Space, constrain_fn=None):
        """
        Return a **copy of the dimension list** from `space`.

        Notes:
            - Kept signature & return type to match your usage.
            - This returns the list of dimensions (not a Space), exactly like your original.
        """
        self.search_space = copy.deepcopy(space.dimensions)
        return self.search_space

    def reset_space(self, space: Space) -> Space:
        """
        Reset the current search space by ensuring all dimensions present in `space`
        are included here (union by name, preserving existing ones).

        Args:
            space: reference Space whose dimensions should be present.

        Returns:
            The updated `self.search_space` (an skopt Space).
        """
        current_names = {dim.name for dim in self.search_space.dimensions}
        new_dims = list(self.search_space.dimensions)  # start with current

        for dim in space.dimensions:
            if dim.name not in current_names:
                new_dims.append(copy.deepcopy(dim))

        self.search_space = Space(new_dims)
        return self.search_space


# ------------------------------ quick check ----------------------------------

if __name__ == '__main__':
    ss = search_space()
    sp = ss.search_sp()

    dtest = {'reg': 1e-4}
    res_final = ss.add_params(dtest)

    print(sp)
    print("-----------------------")
    print(res_final)