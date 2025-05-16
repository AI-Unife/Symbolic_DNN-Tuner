import netron
import Dashboard
from multiprocessing import Process

# Attenzione, se si modifica la porta di netron, e' necessario aggiornare il dato anche nell'iframe in Dashboard.py
netron.start('/hpc/home/bzzlca/Symbolic_DNN-Tuner/25_02_21_11_12_CIFAR_accuracy_module_flops_module_200_30/dashboard/model/model.keras', browse=False)
proc = Process(target=Dashboard.app.run_server(port=8082))
