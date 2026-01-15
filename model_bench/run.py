"""
Entry point for the api
"""
import sys
from pathlib import Path
import yaml

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))
from components.colors import colors
from exp_config import set_active_config, load_cfg, ConfigSchema
from model_bench.menus.main_menu import MainMenu

def ensure_directories():
    """Create al necessary directories for the API"""
    base_dir = Path(__file__).parent

    directories = [
        base_dir / "exports" / "latency",
        base_dir / "exports" / "flops",
        base_dir / "fine_tuned",
        base_dir / "config_presets",
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print(colors.OKGREEN + "[ModelBench] Directory strucuture verified" + colors.ENDC)

def ensure_config():
    """Ensure config.yaml exists, create with defaults if missing"""
    config_path = Path(__file__).parent / "config.yaml"
    
    if config_path.exists():
        return config_path
    
    print(colors.WARNING + f"[ModelBench] Config file not found at {config_path}." + colors.ENDC  )
    print("[ModelBench] Creating config with default values...")
    
    # Create config with defaults
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
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(default_config, f, sort_keys=False, allow_unicode=True)
    
    print(colors.OKGREEN + f"[ModelBench] Config created successfully at {config_path}!" + colors.ENDC)
    
    return config_path

def main():
    """Main entry point for ModelBench"""
    try:
        print("=" * 60)
        print("ModelBench API - Initializing...")
        print("=" * 60)

        ensure_directories()

        config_path = ensure_config()
        set_active_config(config_path)

        cfg = load_cfg()
        print(f"[ModelBench] Using configuration: {cfg.name}")
        print(f"[ModelBench] Dataset: {cfg.dataset}")
        print(f"[ModelBench] Mode: {cfg.mode}")
        print("=" * 60)
        print()
        
        app = MainMenu()
        app.run()
    
    except KeyboardInterrupt:
        print("\n" + colors.WARNING + "[ModelBench] Application interrupted by user" + colors.ENDC)
        sys.exit(0)
    except Exception as e:
        print("\n" + colors.FAIL + f"[ModelBench] ✗ Fatal error: {e}" + colors.ENDC)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()