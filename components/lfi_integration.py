from __future__ import annotations

from problog.program import PrologString
from problog.logic import Term
from problog.learning import lfi

import os
from typing import Any, Iterable, List, Sequence, Tuple

import config as cfg


class LfiIntegration:
    """
    Bridge between the training loop and ProbLog's LFI (Learning From Interpretations).

    Responsibilities:
      - Turn (tuning action, diagnosis, improvement flag) triplets into ProbLog evidence.
      - Keep an in-memory "experience" (list of evidences over time).
      - Read a ProbLog program (actions + rules) and run LFI to learn weights.
    """

    def __init__(self, db):
        """
        Initialize the evidence buffer and store a DB handle for persistence/logging.

        Args:
            db: user-provided storage object expected to expose `insert_evidence(evidence: tuple)`.
        """
        self.db = db
        # Experience is a list of evidence-sets; each element corresponds to one call to `evidence(...)`
        # Format expected by ProbLog LFI: a list of lists of tuples [(atom, truth_value), ...]
        self.experience: List[List[Tuple[Any, bool]]] = []

    # ------------------------------ Helpers ----------------------------------

    def get_str(self, s, before, after):
        """
        Extract substrings between `before` and `after` delimiters.
        NOTE: This is a generator; behavior unchanged from the original.
        """
        return (i.split(after)[0] for i in s.split(before)[1:] if after in i)

    # ----------------------------- Evidence API ------------------------------

    def create_evidence(self, t, d, bool):
        """
        Create a pair of ProbLog evidence atoms from a tuning action and its diagnosis.

        Args:
            t: tuning rule name (string-like or Term-compatible)
            d: diagnosis name (string-like)
            bool: Python bool indicating whether applying `t` improved the metric

        Returns:
            ( (action(t, d), bool), (prob(d_prefix), bool) )

        Notes:
            - The second atom uses the first 3 chars of `d` as a predicate name (legacy behavior).
            - We keep the signature as-is to avoid breaking callers, even though `bool` shadows
              the built-in name; internally we alias it to `success`.
        """
        success = bool  # avoid shadowing the built-in name elsewhere
        t1 = Term(str(t))
        t2 = Term(str(d))

        # Legacy: diagnosis prefix (first 3 chars) as a probability predicate, e.g., 'ove' for 'overfitting'
        prob = Term(d[:3])

        action = Term('action', t1, t2)
        evidence1 = (action, success)
        evidence2 = (prob, success)
        return evidence1, evidence2

    def evidence(self, improve, tuning, diagnosis):
        """
        Build a set of evidence pairs for the current iteration and persist them.

        Args:
            improve (bool): whether the last round produced an improvement.
            tuning (Iterable[str]): list of tuning action names applied.
            diagnosis (Iterable[str]): list of diagnoses aligned with `tuning`.

        Returns:
            List of evidence tuples acceptable by ProbLog LFI:
              [(Term('action', ...), bool), ...]
            (We keep only `e1` entries to preserve original behavior.)
        """
        # Defensive: handle empty input gracefully
        tuning = list(tuning or [])
        diagnosis = list(diagnosis or [])

        evidence: List[Tuple[Any, bool]] = []
        for t, d in zip(tuning, diagnosis):
            e1, e2 = self.create_evidence(t, d, improve)
            evidence.append(e1)
            # Original code kept e2 commented out; we preserve that behavior:
            # evidence.append(e2)

        # Ensure log directory exists
        log_dir = os.path.join(cfg.NAME_EXP, "algorithm_logs")
        os.makedirs(log_dir, exist_ok=True)

        # Append to evidence log and DB (only first item, as per original)
        log_path = os.path.join(log_dir, "evidence.txt")
        with open(log_path, "a") as efile:
            efile.write(str(evidence) + "\n")

        if evidence:
            try:
                self.db.insert_evidence(evidence[0])
            except Exception:
                # Best-effort: don't crash LFI if DB write fails
                pass

        return evidence

    # ------------------------------- Learning --------------------------------

    def learning(self, improve, tuning, diagnosis, actions):
        """
        Run ProbLog LFI to learn action probabilities from accumulated evidence.

        Args:
            improve (bool): improvement flag for the most recent iteration.
            tuning (Iterable[str]): tuning actions just applied.
            diagnosis (Iterable[str]): diagnoses corresponding to `tuning`.
            actions (str): ProbLog clauses defining actions; these are concatenated to the base file.

        Returns:
            (weights, lfi_problem)
              - weights: learned weights from ProbLog
              - lfi_problem: the ProbLog problem/model object (for downstream use)
        """
        # 1) Add current evidence to the in-memory experience
        current = self.evidence(improve, tuning, diagnosis)
        if "X" not in curent:
            self.experience.append(current)

        # 2) Load the base program from disk and append dynamic actions
        sym_dir = os.path.join(cfg.NAME_EXP, "symbolic")
        os.makedirs(sym_dir, exist_ok=True)
        base_path = os.path.join(sym_dir, "lfi.pl")

        try:
            with open(base_path, "r") as f:
                to_learn = f.read()
        except FileNotFoundError:
            # Fallback: allow empty base if file is missing; training may still work with just `actions`
            to_learn = ""

        if actions:
            to_learn += actions

        # 3) Run LFI. `self.experience` must be a list of interpretations (lists of (atom, bool)).
        try:
            # print(self.experience)
            # print(to_learn)
            _, weights, _, _, lfi_problem = lfi.run_lfi(PrologString(to_learn), self.experience)
        except Exception as e:
            # Surface a clearer error while keeping the same return contract (raise is better here)
            raise RuntimeError(f"ProbLog LFI failed: {e}")

        return weights, lfi_problem
