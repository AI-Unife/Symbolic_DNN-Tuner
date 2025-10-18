from tuner import Tuner, TunerConfig

__all__ = ["Tuner", "TunerConfig"]


def create_tuner(config: TunerConfig) -> Tuner:
    """
    Convenience factory that mirrors the previous entrypoint semantics.
    """
    return Tuner(config)


if __name__ == "__main__":
    raise SystemExit(
        "Symbolic DNN Tuner is now framework agnostic. "
        "Please run a framework specific launcher (e.g. pytorch_implementation/main.py)."
    )
