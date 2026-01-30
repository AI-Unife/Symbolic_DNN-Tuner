import sys
from typing import List
from skopt.space import Space, Categorical, Integer, Real
from exp_config import load_cfg

class ConstraintsWrapper:
    def __init__(self, space: Space):
        self.space = space
        self.exp_cfg = load_cfg()

    def apply_constraints(self, params: List) -> bool:
        """
        Check whether a sampled parameter vector fits within the (possibly updated) space.

        When running RS variants we skip constraints (returns True).
        """
        if "RS" in self.exp_cfg.opt:
            return True

        for i, dim in enumerate(self.space.dimensions):
            val = params[i]
            if isinstance(dim, Categorical):
                if val not in dim.categories:
                    return False
            elif isinstance(dim, (Integer, Real)):
                if not (dim.low <= val <= dim.high):
                    return False
            else:
                ### IMPROVEMENT: Do not sys.exit(1) from a helper class. ###
                # Raise an error that the main program can catch if needed.
                raise TypeError(f"Type space dimension {dim} - {type(dim)} not valid")
        return True
