import questionary
import os
from components.colors import colors
from modules.loss.hardware_module import hardware_module
from menus.hardware_menu import HardwareMenu
from utils.hardware_utils import select_hw_config
from utils.model_utils import check_if_path_is_model
from utils.testing_utils import single_model_test, multi_model_test
from utils.flops_utils import calculate_model_flops, calculate_multiple_models_flops
from utils.export_utils import export_latency_results, export_flops_results

class TestingMenu:
    def __init__(self):
        self.hw_mod = hardware_module()
        self.hardware_menu = HardwareMenu(self.hw_mod)

    def display_header(self):
        os.system('clear' if os.name == 'posix' else 'cls')
        print(colors.CYAN + "+-----------------------------------+")
        print("|        Model Testing Menu         |")
        print("+-----------------------------------+" + colors.ENDC)

    def print_grouped_results(self, results):
        """Print grouped latency results by model"""
        grouped = {}
        for res in results:
            model = res.get("model", "")
            if model not in grouped:
                grouped[model] = []
            grouped[model].append(res)

        print(colors.OKGREEN, "\n|  ---------------- LATENCY ANALYSIS ----------------  |", colors.ENDC)
        for model, items in grouped.items():
            print(colors.MAGENTA, f"Model: {model}", colors.ENDC)
            for r in items:
                hw = r.get("HW config")
                lat = r.get("latency(s)")
                print(f"Configuration: {hw} | Latency: {lat:.6f} seconds")
        print(colors.OKGREEN, "\n|  --------------------------------------------------  |\n", colors.ENDC)

    def print_flops_results(self, results):
        """Print FLOPs results in a formatted way"""
        if not results:
            print(colors.FAIL, "No FLOPs results to display.", colors.ENDC)
            return
        
        print(colors.OKGREEN + "\n|  ---------------------- FLOPS ANALYSIS ----------------------  |" + colors.ENDC)
        
        for result in results:
            print(colors.MAGENTA + f"\nModel: {result['model']}" + colors.ENDC)
            print(f"  Path: {result['path']}")
            print(f"  Total FLOPs: {result['total_flops']:,}")
            print(f"  GFLOPs: {result['gflops']:.4f}")
            print(f"  MFLOPs: {result['mflops']:.4f}")
        
        print(colors.OKGREEN + "\n|  ------------------------------------------------------------  |\n" + colors.ENDC)

    def test_hw_latency(self):
        """Test hardware latency for single or multiple models"""
        hw_choose = select_hw_config(self.hw_mod)
        if not hw_choose:
            return

        results = []

        while True:
            models_path=questionary.path(
                "Enter the model path or directory path containing multiple models:"
                ).ask()

            if not models_path:
                models_path = "gesture/model" #Default path if not provided
                print("Using default model(s) path:", models_path)
            models_path = os.path.abspath(os.path.expanduser(models_path))
            
            # Check if it's a single model file
            if check_if_path_is_model(models_path):
                result = single_model_test(self.hw_mod, hw_choose, models_path)
                if result:
                    results = [result]
                break
            # Check if it's a directory with multiple models
            elif os.path.isdir(models_path):
                recursive = questionary.confirm(
                    "Search recursively in subdirectories?",
                    default=True
                ).ask()
                results = multi_model_test(self.hw_mod, hw_choose, models_path, recursive=recursive)
                break
            else:
                print(colors.FAIL, "Invalid file/directory. Please try again.", colors.ENDC)
                continue
        
        if results is None:
            print(colors.FAIL, "No results returned from model test.", colors.ENDC)
            return
        #display grouped results
        self.print_grouped_results(results)

        export_choice = questionary.confirm(
            "Do you want to export the latency results to a CSV file?",
            default=True
        ).ask()
        if export_choice:
            export_latency_results(results)

        questionary.press_any_key_to_continue().ask()

    def test_model_flops(self):
        """Test FLOPs for single or multiple models"""
        results = []
        
        while True:
            models_path = questionary.path(
                "Enter the model path or directory path containing multiple models:"
            ).ask()

            if not models_path:
                print(colors.WARNING, "No path provided. Using default.", colors.ENDC)
                models_path = "gesture/model"
                print("Using default model(s) path:", models_path)

            
            models_path = os.path.abspath(os.path.expanduser(models_path))

            # Check if it's a single model file
            if check_if_path_is_model(models_path):
                print(f"\n{colors.CYAN}Analyzing single model...{colors.ENDC}")
                result = calculate_model_flops(models_path)
                if result:
                    results = [result]
                break
            
            # Check if it's a directory with multiple models
            elif os.path.isdir(models_path):
                recursive = questionary.confirm(
                    "Search recursively in subdirectories?",
                    default=True
                ).ask()
                print(f"\n{colors.CYAN}Analyzing multiple models in directory...{colors.ENDC}")
                results = calculate_multiple_models_flops(models_path, recursive=recursive)
                break
            
            else:
                print(colors.FAIL, "Invalid file/directory. Please try again.", colors.ENDC)
                continue

        if not results:
            print(colors.FAIL, "No FLOPs results obtained.", colors.ENDC)
            return
        
        # Display results
        self.print_flops_results(results)
        
        # Ask if user wants to see detailed breakdown
        if len(results) == 1:
            show_details = questionary.confirm(
                "Do you want to see detailed operation breakdown?"
            ).ask()
            
            if show_details and results[0].get('details'):
                print("\n" + colors.CYAN + "Detailed Operations:" + colors.ENDC)
                print(results[0]['details'])

        export_choice = questionary.confirm(
            "Do you want to export the FLOPs results to a CSV file?",
            default=True
        ).ask()
        if export_choice:
            export_flops_results(results)

        questionary.press_any_key_to_continue().ask()


    def run(self):
        while True:
            if not self.hw_mod:
                print(colors.FAIL, "No hardware configurations available. Please manage hardware first.", colors.ENDC)
                self.hardware_menu = HardwareMenu(self.hw_mod)  
                
            self.display_header()
            choice = questionary.select(
                "Select an option:",
                choices=[
                    "1: Manage Hardware Configurations",
                    "2: Test Hardware Latency",
                    "3: Test Model Flops",
                    "4: Back to Main Menu"
                ]
            ).ask()

            if not choice:
                break

            choice_num = choice[0] if choice else ""

            if choice_num == "1":
                self.hardware_menu.run()
                self.hw_mod = self.hardware_menu.hw_mod
            elif choice_num == "2":
                self.test_hw_latency()
            elif choice_num == "3":
                self.test_model_flops()
            elif choice_num == "4":
                break