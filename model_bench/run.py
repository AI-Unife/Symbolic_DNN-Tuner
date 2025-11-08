"""
Entry point for the api
"""
import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from exp_config import set_active_config, load_cfg

config_path = Path(__file__).parent / "config.yaml"

set_active_config(config_path)
from model_bench.menus.main_menu import MainMenu

def main():

    cfg = load_cfg()
    print(f"[ModelBench] Usando configurazione: {cfg.name}")
    print(f"[ModelBench] Dataset: {cfg.dataset}")
    print()

    """Avvia l'applicazione"""
    app = MainMenu()
    app.run()

if __name__ == "__main__":
    main()