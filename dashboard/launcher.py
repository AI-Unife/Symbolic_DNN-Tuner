import netron
import Dashboard
from multiprocessing import Process

# Attenzione, se si modifica la porta di netron, e' necessario aggiornare il dato anche nell'iframe in Dashboard.py
netron.start('model/model.h5', log=False, browse=False)

def run_dashboard():
    Dashboard.app.run_server(port=8082)

if __name__ == "__main__":
    proc = Process(target=run_dashboard)
    proc.start()
    try:
        proc.join()
    except KeyboardInterrupt:
        proc.terminate()
        proc.join()
