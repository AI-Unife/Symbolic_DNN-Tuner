import questionary
import os
from components.colors import colors
from utils.model_utils import load_trained_model, check_if_path_is_model
from utils.hardware_utils import select_hw_config
from tqdm import tqdm


class ModelTestMenu:

    def __init__(self, hw_mod=None):
        self.hw_mod = hw_mod

    def display_header(self):
        print(colors.CYAN + "+-----------------------------------+")
        print("|         Model Test Menu           |")
        print("+-----------------------------------+" + colors.ENDC)

    def single_model_test(self):
        while True:
            model_path=questionary.path("Enter the path of the pre-trained model").ask()
            
            if not model_path:
                model_path = "gesture/dashboard/model/model.keras" #Default path if not provided
                print("Using default model path:", model_path)
                break

        model=load_trained_model(model_path)
        print(colors.OKGREEN, "Model loaded successfully\n", colors.ENDC)

        hw_choose = select_hw_config(self.hw_mod)
        print(colors.OKGREEN, "|  ---------------------- TEST ----------------------  |", colors.ENDC)
        hw_test(self.hw_mod, model, hw_choose)
        print(colors.OKGREEN, "|  --------------------------------------------------  |\n", colors.ENDC)

    def multi_model_test(self):
        while True:
            models_path = questionary.path("Enter the directory path containing multiple models").ask()
            if not models_path: 
                print(colors.FAIL, "Directory path cannot be empty. Please try again.", colors.ENDC)
                continue
            
            models_path = os.path.abspath(os.path.expanduser(models_path))

            if os.path.exists(models_path) and os.path.isdir(models_path):
                break
            print(colors.FAIL, "Invalid directory. Please try again.", colors.ENDC)

        hw_choose = select_hw_config(self.hw_mod)
       
        model_files = []
        allowed_extensions = (".keras", ".h5")

        print(colors.OKGREEN, f"Scanning directory: {models_path}", colors.ENDC)
        for root, _, files in os.walk(models_path):
            for file in files:
                if file.endswith(allowed_extensions):
                    model_files.append(os.path.join(root, file))
        print(colors.OKGREEN, f"Found {len(model_files)} model(s) in the directory.", colors.ENDC)

        if not model_files:
            print(colors.FAIL, "No models found in the specified directory.", colors.ENDC)
            return
        else:
            confirm = questionary.confirm(f"Proceed to test {len(model_files)} model(s)?", default=True).ask()
            if not confirm:
                print(colors.FAIL, "Operation cancelled by user.", colors.ENDC)
                return
        

        results = []

        with tqdm(total=len(model_files), desc="Testing models", unit="model") as pbar:
            for file in model_files:
                relative_path = os.path.relpath(file, models_path)
                if check_if_path_is_model(file):
                    model = load_trained_model(file, show_info=False)
                    for hw_c in hw_choose:
                        if hw_c in self.hw_mod.nvdla:
                            latency = self.hw_mod.get_model_latency(model, self.hw_mod.nvdla[hw_c]['path'])
                            results.append({
                                "model": relative_path,
                                "HW config": hw_c,
                                "latency(s)": f"{latency/10**9:.6f}"
                            })
                            print(f"Model: {os.path.basename(file)} | Configuration: {hw_c} | Latency: {latency/10**9:.6f} seconds")
                pbar.update(1)  # Aggiorna la barra di caricamento
        print(colors.OKGREEN, f"Testing completed. Results for {len(results)} tests:", colors.ENDC)
        for res in results:
            print(f"Model: {res['model']} | Config: {res['HW config']} | Latency: {res['latency(s)']} seconds")

            """
            for config_name, config in self.hw_mod.nvdla.items():
                latency = self.hw_mod.get_model_latency(model, config['path'])
                risultati.append({
                    "modello": relative_path,
                    "nome configurazione HW": config_name,
                    "latenza": latency / (10**9)
                })
            """  





        print(colors.OKGREEN, "|  ---------------------- TEST ----------------------  |", colors.ENDC)        
        
        for file in model_files:
            relative_path = os.path.relpath(file, models_path)
            print(colors.MAGENTA, f"Testing model: {relative_path}\n", colors.ENDC)
            if check_if_path_is_model(file):
                model = load_trained_model(file, False)
                hw_test(self.hw_mod, model, hw_choose)
        print(colors.OKGREEN, "|  --------------------------------------------------  |\n", colors.ENDC)


    """
    with tqdm(total=len(model_files), desc="Testing models", unit="model") as pbar:
        for file in model_files:
            relative_path = os.path.relpath(file, models_path)
            pbar.set_postfix({"Current Model": relative_path})  # Mostra il nome del modello
            #print(f"Checking model at: {path}")
            if check_model_path(file):
                model = load_trained_model(file, show_info=False)
                for config_name, config in hw_mod.nvdla.items():
                    latency = hw_mod.get_model_latency(model, config['path'])
                    risultati.append({
                        "modello": relative_path,
                        "nome configurazione HW": config_name,
                        "latenza": latency / (10**9)
                    })
                    # print(f"Configuration: {config_name} | Latency: {latency/(10**9):.6f} secondi")
            pbar.update(1)  # Aggiorna la barra di caricamento

    save_results_to_excel(risultati, "risultati_latenza.xlsx")
        print("Multi-model testing is not yet implemented.")
    """
    
    def run(self):
        self.display_header()
        while True:
            choice = questionary.select(
                "Select an option:",
                choices=[
                    "1: Test single model",
                    "2: Test multiple models",
                    "3: Back to Latency Menu"
                ]
            ).ask()

            if not choice:
                break

            choice_num = choice[0] if choice else ""

            if choice_num == "1":
                self.single_model_test()
            elif choice_num == "2":
                self.multi_model_test()
            elif choice_num == "3":
                break

