"""
Entry point for the api
"""
import os
import sys
from pathlib import Path
import yaml

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from exp_config import set_active_config, load_cfg, ConfigSchema
from model_bench.menus.main_menu import MainMenu

def main():
    config_path = Path(__file__).parent / "config.yaml"
    
    if not config_path.exists():
        print(f"[ModelBench] Config file non trovato in {config_path}.")
        print("[ModelBench] Creazione config con valori di default...")
        
        # Crea manualmente il config con i default
        schema = ConfigSchema()
        default_config = {
            "eval": schema.eval,
            "epochs": schema.epochs,
            "mod_list": schema.mod_list or [],
            "dataset": schema.dataset,
            "name": schema.name,
            "frames": schema.frames,
            "mode": schema.mode,
            "channels": schema.channels,
            "polarity": schema.polarity,
            "seed": schema.seed,
            "opt": schema.opt,
        }
        
        # Salva il file
        with open(config_path, 'w') as f:
            yaml.safe_dump(default_config, f, sort_keys=False, allow_unicode=True)
        
        print(f"[ModelBench] Config creato con successo in {config_path}!")

    set_active_config(config_path)

    cfg = load_cfg()
    print(f"[ModelBench] Usando configurazione: {cfg.name}")
    print(f"[ModelBench] Dataset: {cfg.dataset}")
    print()

    """Avvia l'applicazione"""
    app = MainMenu()
    app.run()

if __name__ == "__main__":
    main()