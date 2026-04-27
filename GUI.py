from graficifun import *
from analisi_out import *
import streamlit as st
import tkinter as tk
from tkinter import filedialog

@st.cache_data          # Cache dei dati in modo tale che a ogni iterazione con la pagina non venga ricaricato tutto da zero
def caricamento():
    return analyze_all_experiments(st.session_state.cartellaInput, st.session_state.cartellaOutput) #carico nome degli esperimenti e analizer corrispondenti alla cartella selezionata

@st.cache_data          # Cache dei dati in modo tale che a ogni iterazione con la pagina non venga ricaricato tutto da zero
def caricamento2():
    return find_out(st.session_state.cartellaInput) #carico nome degli esperimenti e analizer corrispondenti alla cartella selezionata


def fase_selezione_cartella():
    root = tk.Tk()   # Creo una finestra di tkinter per poter utilizzare il filedialog: seleziona cartella 
    root.withdraw()
    root.attributes('-topmost', True)

    st.empty()      # Pulisco il placeholder per rimuovere eventuali elementi di diverse fasi
    st.title("📈 - ANALYZER Symbolic DNN Tuner")
    
    # Sezione per la cartella da analizzare
    st.subheader("📂 - Selettore cartella da analizzare")   #sottotitolo
    cola, colb = st.columns([2,5])         #creo due colonne e adatto il bottone alla prima colonna per farlo uguale a quello sottostante
    with cola:
        if st.button('Seleziona cartella da analizzare ->',use_container_width=True):   #quando viene premuto il bottone
            # Forza il sistema a processare gli eventi grafici
            root.update()
            cartella_scelta = filedialog.askdirectory(master=root)  #si apre il filedialog per selezionare la cartella da analizzare
            if cartella_scelta:
                st.session_state.cartellaInput = cartella_scelta    #la salvo in una variabile di stato per poterla usare nelle fasi successive
                st.rerun()          # Ricarico la pagina
            
    if 'cartellaInput' in st.session_state:     #se esiste una cartella da analizzare, mostro un messaggio con il percorso della cartella selezionata
        st.info(f"Cartella selezionata: **{st.session_state.cartellaInput}**")

    #selezione opzionale per la cartella di output
    st.subheader("📂 - Selettore cartella di salvataggio [Opzionale]")
    col1, col2, col_vuota1 = st.columns([2, 1, 4]) 
    with col1:
        if st.button('Seleziona cartella di salvataggio ->',use_container_width=True):
            # Forza il sistema a processare gli eventi grafici
            root.update()
            cartella_scelta = filedialog.askdirectory(master=root)
            if cartella_scelta:
                st.session_state.cartellaOutput = cartella_scelta
                st.rerun()
    with col2:
        if st.button('Nessuna selezione',use_container_width=True):
            st.session_state.cartellaOutput = None
            st.rerun()

    if 'cartellaOutput' in st.session_state:
        st.info(f"Cartella di salvataggio selezionata: **{st.session_state.cartellaOutput}**")

    #Bottone vai alla fase successiva
    col_vuota2, col3= st.columns([6, 1]) 
    with col3:
        if st.button('Continua',use_container_width=True):      #quando viene premuto il bottone continua si passa alla fase successiva a patto che sia stata selezionata la cartella di analisi
            if 'cartellaInput' in st.session_state:
                if 'cartellaOutput' not in st.session_state:
                    st.session_state.cartellaOutput = None  # Se non è stata selezionata una cartella di output, la imposto a None

                st.session_state.fase = 'caricamento'
                root.destroy()
                st.rerun()  # Ricarica la pagina per passare alla fase successiva
            else:
                st.error("Per favore, seleziona una cartella da analizzare prima di continuare.", icon="❌")

def fase_caricamento():
    st.empty()            
    st.title("⚙️ - Caricamento dei dati")
    
    if 'stato_bottone_singoli' not in st.session_state:     #creo una variabile di stato per tenere traccia se è stato premuto il bottone per la selezione dei grafici, in modo tale che possa vedere sezione sotto bottone tutto il tempo
        st.session_state.stato_bottone_singoli = False
    if 'stato_bottone_confronto' not in st.session_state:
        st.session_state.stato_bottone_confronto = False
    if 'stato_bottone_spazio_ricerca' not in st.session_state:
        st.session_state.stato_bottone_spazio_ricerca = False

    with st.spinner('Caricando i vari esperimenti...'):     # Mostro uno spinner mentre vengono caricati i dati, in modo da far capire all'utente che l'applicazione sta lavorando e non è bloccata
        lista_esperimenti, lista_analizer = caricamento ()  # Carico i dati degli esperimenti e gli analizer corrispondenti alla cartella selezionata, usando la funzione caricamento che è stata decorata con st.cache_data per evitare di ricaricare tutto da zero a ogni iterazione con la pagina
        if lista_esperimenti and lista_analizer:            #ho bisogno di far diventare queste due liste variabili di stato per poterle usare nelle fasi successive
            st.session_state.lista_esperimenti = lista_esperimenti
            st.session_state.lista_analizer = lista_analizer

        lista_out_files = caricamento2()     #restituise una lista di Spazi di ricerca che sono composti a loro volta di unalista di dizionari e il nome dell'esperimento
        if lista_out_files:
            st.session_state.lista_out_files = lista_out_files

    if not lista_esperimenti:       #mostro un messaggio di warning se non sono ststi trovati esperimenti
        st.warning("❌ - Non sono stati trovati esperimenti nella cartella selezionata. Per favore, verifica che la cartella contenga i dati corretti e riprova.")
        
    if 'lista_grafici_singoli' not in st.session_state:
        st.session_state.lista_grafici_singoli = None   #creo una variabile di stato per tenere traccia dei grafici da fare per ogni esperimento, in modo tale che se dovessi tornare indietro e poi tornare a questa fase non perdo l'informazione sui grafici da fare
    if 'lista_grafici_confronto' not in st.session_state:
        st.session_state.lista_grafici_confronto = None   #creo una variabile di stato per tenere traccia dei grafici di confronto da fare, in modo tale che se dovessi tornare indietro e poi tornare a questa fase non perdo l'informazione sui grafici di confronto da fare

    if lista_esperimenti:
        st.write(f"Trovati {len(lista_esperimenti)} esperimenti nella cartella selezionata: ")      #mostro cosa ho trovato
        st.write("Seleziona esperimenti da analizzare:")
        esp_analizzare=st.multiselect("Esperimenti da analizzare:", [exp.name for exp in lista_esperimenti])        #chiedo quali esperimenti analizzare 

        col1,col2,col3= st.columns([2, 2, 2]) 
        with col1:
            if esp_analizzare and st.button('Crea grafici per singoli esperimenti'):
                st.session_state.stato_bottone_singoli = True
                st.rerun()
        with col2:
            if esp_analizzare and st.button('Grafici di confronto'):
                st.session_state.stato_bottone_confronto = True
                st.rerun()
        with col3:
            if esp_analizzare and st.button('Spazio di Ricerca'):
                st.session_state.stato_bottone_spazio_ricerca = True
                st.rerun()

        if esp_analizzare and st.session_state.stato_bottone_singoli:
            st.subheader("Hai selezionato di visualizzare i grafici per singoli esperimenti.")

            if 'scelta_comune' not in st.session_state:     # Creo una variabile di stato per memorizzare una scelta comune in modo tale che se avessi molti esperimenti la scelta dei grafici da visulizzare possa essere più veloce 
                st.session_state.scelta_comune = None

            grafici_da_fare_singoli = []        # mi devo tenere traccia per ogni esperimento cosa graficare
            opzioni_grafici = [         #grafici possibili
                "LinePlot di Accuracy, Score e Params", 
                "BarPlot efficacia azioni", 
                "BarPlot diagnosi", 
                "BarPlot tuning", 
                "ScatterPlot Timeline tuning"]

            for i, exp in enumerate(esp_analizzare):
                # Se è stata impostata una scelta comune, la usiamo per tutti gli esperimenti senza chiedere
                if st.session_state.scelta_comune is not None:
                    grafici = st.session_state.scelta_comune
                    st.write(f"✅ {exp}: Applicata scelta comune: {', '.join(grafici)}")
                else:
                    # Altrimenti chiediamo all'utente
                    grafici = st.multiselect(       # Apro un menu di selezione multipla
                        f"Quali grafici devo generare per {exp}?",
                        opzioni_grafici,
                        key=f"ms_{exp}" #importante mettere una chiave diversa per ogni multiselect in modo tale che Streamlit riesca a distinguere i diversi menu
                    )
                    
                    # Mostriamo il bottone applica a tutti solo se l'utente ha selezionato qualcosa nel primo esperimento. Gli altri non hanno bottone
                    if i == 0 and grafici:                    
                        if st.button(f"Applica a tutti"):
                            st.session_state.scelta_comune = grafici
                            st.rerun()

                grafici_da_fare_singoli.append(grafici)     # tengo traccia per ogni esperimento (indice) quali grafici fare 
            
            st.session_state.lista_grafici_singoli = grafici_da_fare_singoli   #me lo devo salvare in una variabile di stato sennò nella fase di creazione dei grafici non riesco più a reperire l'informazione

        if esp_analizzare and st.session_state.stato_bottone_confronto:
            st.subheader("Hai selezionato di visualizzare i grafici di confronto.")

            opzioni_grafici = [         #grafici possibili
                "LinePlot di Accuracy, Score e Params", 
                "BarPlot efficacia azioni", 
                "BarPlot diagnosi", 
                "BarPlot tuning", 
                "ScatterPlot Timeline tuning"]
    
            grafici = st.multiselect(f"Quali grafici devo generare?", opzioni_grafici)       # Apro un menu di selezione multipla
            
            st.session_state.lista_grafici_confronto = grafici

        if st.session_state.stato_bottone_spazio_ricerca:
            st.subheader("Hai selezionato di visualizzare i grafici di Spazio di Ricerca.")

    colno,colok= st.columns([6, 2]) 
    with colok:
        if (st.session_state.lista_grafici_singoli or st.session_state.lista_grafici_confronto or st.session_state.stato_bottone_spazio_ricerca) and st.button('Crea grafici'):  # se ho dei grafici da creare mostro il bottone per andare alla fase successiva

            st.session_state.lista_esperimenti=[exp for exp in lista_esperimenti if exp.name in esp_analizzare]   #filtro la lista degli esperimenti in modo tale che nella fase di creazione dei grafici mi rimangano solo quelli che voglio analizzare

            st.session_state.fase = 'analisi'
            st.rerun() 
    with colno:
        if st.button('Annulla'):        # se mi dovessi accorgere che ho fatto una scelta sbagliata posso tornare indietro e ripartire da zero (cancello tutta la cache e le variabili di stato)
            st.session_state.fase = 'selezione_cartella'
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.clear()
            st.query_params.clear()
            st.rerun()

@st.fragment
def render_chart(exp, analizer):
    fig = plotaccuracy(analizer, exp)
    chart_key = f"chart_{exp.name}"
    
    # Render del grafico
    event = st.plotly_chart(fig,on_select="rerun",selection_mode="points", key=chart_key)

    # Verifica selezione
    if event and event.get("selection", {}).get("points"):
        points = event["selection"]["points"][0]
        # Recupero sicuro dell'indice
        custom_data = points.get("customdata")
        iter_idx = custom_data[0] if isinstance(custom_data, list) else custom_data
        
        if iter_idx is not None:
            mostra_dettaglio(exp, iter_idx, f"{chart_key}_{iter_idx}")

@st.dialog("📈 Dettaglio Trend")
def mostra_dettaglio(esperimento, iter_idx, chart_key):
    st.write(f" Dettaglio per Iterazione {iter_idx} di {esperimento.name}")
    
    # Logica recupero dati
    found = False
    for esp in st.session_state.lista_out_files:
        if esp.exp_name == esperimento.name:
            st.plotly_chart(esp.grafici_trend(iter_idx))
            found = True
            break
    
    if not found:
        st.error("Dati non trovati.")

def fase_analisi():
    st.empty()
    st.title("🔍 - Analisi dei risultati")

    if 'salvato_tutto' not in st.session_state:
        st.session_state.salvato_tutto = False

    col1,col2= st.columns([8, 2]) 
    with col1:
        if st.session_state.cartellaOutput and st.button('Salva tutto'):    # se è stata selezionata una cartella di output mostro un bottone per salvare tutti i grafici in un colpo solo, altrimenti non mostro questo bottone
            st.session_state.salvato_tutto = True
            st.rerun()  # Ricarico la pagina per passare alla fase successiva
            
    with col2:              # se una volta visti non voglio fare altre azioni ho la possibilità di andare alla fase successiva
        if st.button('Fine'):
            st.session_state.fase = 'fine'
            st.rerun()  # Ricarica la pagina per passare alla fase successiva

    if st.session_state.stato_bottone_singoli and st.session_state.salvato_tutto:     # se sono nella modalità singoli grafici, salvo solo quelli selezionati per ogni esperimento
        for i in range(len(st.session_state.lista_grafici_singoli)):
            grafici = st.session_state.lista_grafici_singoli[i]
            exp = st.session_state.lista_esperimenti[i]
            if grafici:
                for grafico in grafici:
                    analizer = st.session_state.lista_analizer[i]

                    if grafico == "LinePlot di Accuracy, Score e Params":
                        plotaccuracy(analizer,exp, st.session_state.cartellaOutput)
                    elif grafico == "BarPlot efficacia azioni":
                        plotevidence(analizer,exp, st.session_state.cartellaOutput)
                    elif grafico == "BarPlot diagnosi":
                        plotdiagnosis(analizer,exp, st.session_state.cartellaOutput)
                    elif grafico == "BarPlot tuning":
                        plottuning(analizer,exp, st.session_state.cartellaOutput)
                    elif grafico == "ScatterPlot Timeline tuning":
                        plottimeline(analizer,exp, st.session_state.cartellaOutput)

        st.success("✅ - Tutti i grafici singoli sono stati salvati con successo nella cartella di output selezionata!")

    if st.session_state.stato_bottone_confronto and st.session_state.salvato_tutto:     # se sono nella modalità grafici di confronto, salvo tutti i grafici di confronto.
        #seleziono solo gli analizer degli esperimenti che voglio analizzare per poterli passare alle funzioni di confronto
        nomi=[exp.name for exp in st.session_state.lista_esperimenti]
        analizer_confronto = [analizer for analizer in st.session_state.lista_analizer if analizer.exp_name in nomi]
        for grafico in st.session_state.lista_grafici_confronto:
            if grafico == "LinePlot di Accuracy, Score e Params":
                plotaccuracy_confronto(analizer_confronto, st.session_state.cartellaOutput)
            elif grafico == "BarPlot efficacia azioni":
                plotevidence_confronto(analizer_confronto, st.session_state.cartellaOutput)
            elif grafico == "BarPlot diagnosi":
                plotdiagnosis_confronto(analizer_confronto, st.session_state.cartellaOutput)
            elif grafico == "BarPlot tuning":
                plottuning_confronto(analizer_confronto, st.session_state.cartellaOutput)
            elif grafico == "ScatterPlot Timeline tuning":
                plottimeline_confronto(analizer_confronto, st.session_state.cartellaOutput)

        st.success("✅ - Tutti i grafici di confronto sono stati salvati con successo nella cartella di output selezionata!")

    if st.session_state.stato_bottone_spazio_ricerca and st.session_state.salvato_tutto:     # se sono nella modalità spazio di ricerca, salvo tutti i grafici di spazio di ricerca degli esperimenti selezionati.
        for spazio_ricerca in st.session_state.lista_out_files:
            if spazio_ricerca and spazio_ricerca.exp_name in [exp.name for exp in st.session_state.lista_esperimenti]:   # se sono nella modalità spazio di ricerca, salvo tutti i grafici di spazio di ricerca degli esperimenti selezionati.
                spazio_ricerca.grafici_spazio_ricerca(st.session_state.cartellaOutput)
        st.success("✅ - Tutti i grafici di Spazio di Ricerca sono stati salvati con successo nella cartella di output selezionata!")
    
    st.session_state.salvato_tutto = False   # resetto la variabile di stato in modo tale che se dovessi tornare indietro e poi tornare a questa fase non mi rimane salvato tutto e non mi salva tutto senza volerlo

    if st.session_state.stato_bottone_singoli:     # se sono nella modalità singoli grafici, mostro i grafici uno alla volta con i relativi bottoni di salvataggio    
        st.subheader("Grafici per singoli esperimenti")
        with st.spinner('Generando i grafici...'):      # generazione vera e propria dei grafici
            for i in range(len(st.session_state.lista_grafici_singoli)):    # corro con ciclo 
                grafici = st.session_state.lista_grafici_singoli[i]     #capisco cosa devo graficare
                exp = st.session_state.lista_esperimenti[i]     # per quale esperimento
                if grafici:                                     # se ho grafici
                    st.write(f"Analizzo {exp.name}")            #analizzo
                    for grafico in grafici:
                        analizer = st.session_state.lista_analizer[i]

                        if grafico == "LinePlot di Accuracy, Score e Params":       # serie di if-elif per capire quale grafico fare 
                            st.write(f"- Genero {grafico}")
                            if st.session_state.cartellaOutput and st.button(f"Salva {grafico} per {exp.name}", key=f"salva_{i}_accuracy"):         # creo un bottone che mi possa salvare quel grafico specifico
                                plotaccuracy(analizer,exp, st.session_state.cartellaOutput)     #ricreo il grafico salvandolo questa volta
                                st.success(f"✅ - {grafico} per {exp.name} salvato con successo!")      # mostro un messaggio che mi indica che il grafico è stato salvato bene
                            else:
                                render_chart(exp, analizer)     # chiamo una funzione apposita per fare il render del grafico in modo tale da poter gestire meglio la selezione dei punti e la visualizzazione del dettaglio

                        elif grafico == "BarPlot efficacia azioni":
                            col1, col2, col3 = st.columns([1, 3, 1])
                            with col2:
                                st.write(f"- Genero {grafico} ")
                                if st.session_state.cartellaOutput and st.button(f"Salva {grafico} per {exp.name}", key=f"salva_{i}_evidence"):
                                    plotevidence(analizer,exp, st.session_state.cartellaOutput)
                                    st.success(f"✅ - {grafico} per {exp.name} salvato con successo!")
                                else:
                                    fig=plotevidence(analizer,exp, None)
                                    st.plotly_chart(fig)

                        elif grafico == "BarPlot diagnosi":
                            st.write(f"- Genero {grafico} ")
                            if st.session_state.cartellaOutput and st.button(f"Salva {grafico} per {exp.name}", key=f"salva_{i}_diagnosis"):
                                plotdiagnosis(analizer,exp, st.session_state.cartellaOutput)
                                st.success(f"✅ - {grafico} per {exp.name} salvato con successo!")
                            else:
                                fig=plotdiagnosis(analizer,exp, None)
                                st.plotly_chart(fig)

                        elif grafico == "BarPlot tuning":
                            st.write(f"- Genero {grafico} ")
                            if st.session_state.cartellaOutput and st.button(f"Salva {grafico} per {exp.name}", key=f"salva_{i}_tuning"):
                                plottuning(analizer,exp, st.session_state.cartellaOutput)
                                st.success(f"✅ - {grafico} per {exp.name} salvato con successo!")
                            else:
                                fig=plottuning(analizer,exp,None)
                                st.plotly_chart(fig)

                        elif grafico == "ScatterPlot Timeline tuning":
                            col1, col2, col3 = st.columns([1, 3, 1])
                            with col2:
                                st.write(f"- Genero {grafico} ")
                                if st.session_state.cartellaOutput and st.button(f"Salva {grafico} per {exp.name}", key=f"salva_{i}_timeline"):
                                    plottimeline(analizer,exp, st.session_state.cartellaOutput)
                                    st.success(f"✅ - {grafico} per {exp.name} salvato con successo!")
                                else:
                                    fig=plottimeline(analizer,exp,None)
                                    st.plotly_chart(fig)
                else:
                    st.write(f"Nessun grafico selezionato per {exp.name}, salto l'analisi.")        # avviso che per un esperimento non è ststo selezionato nulla e proseguo

    if st.session_state.stato_bottone_confronto:     # se sono nella modalità grafici di confronto, mostro i grafici di confronto per tutti gli esperimenti selezionati
        st.subheader("Grafici di confronto")
        with st.spinner('Generando i grafici di confronto...'):
            #seleziono solo gli analizer degli esperimenti che voglio analizzare per poterli passare alle funzioni di confronto
            nomi=[exp.name for exp in st.session_state.lista_esperimenti]
            analizer_confronto = [analizer for analizer in st.session_state.lista_analizer if analizer.exp_name in nomi]
            
            for grafico in st.session_state.lista_grafici_confronto:
                if grafico == "LinePlot di Accuracy, Score e Params":
                    st.write(f"Genero {grafico}")
                    if st.session_state.cartellaOutput and st.button(f"Salva {grafico}", key=f"salva_confronto_accuracy"):
                        plotaccuracy_confronto(analizer_confronto, st.session_state.cartellaOutput)
                        st.success(f"✅ - {grafico} di confronto salvato con successo!")
                    fig=plotaccuracy_confronto(analizer_confronto)
                    st.plotly_chart(fig)

                elif grafico == "BarPlot efficacia azioni":
                    st.write(f"Genero {grafico}")
                    if st.session_state.cartellaOutput and st.button(f"Salva {grafico}", key=f"salva_confronto_evidence"):
                        plotevidence_confronto(analizer_confronto, st.session_state.cartellaOutput)
                        st.success(f"✅ - {grafico} di confronto salvato con successo!")
                    fig=plotevidence_confronto(analizer_confronto)
                    st.plotly_chart(fig)

                elif grafico == "BarPlot diagnosi":
                    cola,colb, colc = st.columns([1, 8, 1])
                    with colb:
                        st.write(f"Genero {grafico}")
                        if st.session_state.cartellaOutput and st.button(f"Salva {grafico}", key=f"salva_confronto_diagnosis"):
                            plotdiagnosis_confronto(analizer_confronto, st.session_state.cartellaOutput)
                            st.success(f"✅ - {grafico} di confronto salvato con successo!")
                        fig=plotdiagnosis_confronto(analizer_confronto)
                        st.plotly_chart(fig)

                elif grafico == "BarPlot tuning":
                    cola,colb, colc = st.columns([1, 8, 1])
                    with colb:
                        st.write(f"Genero {grafico}")
                        if st.session_state.cartellaOutput and st.button(f"Salva {grafico}", key=f"salva_confronto_tuning"):
                            plottuning_confronto(analizer_confronto, st.session_state.cartellaOutput)
                            st.success(f"✅ - {grafico} di confronto salvato con successo!")
                        else:
                            fig=plottuning_confronto(analizer_confronto)
                            st.plotly_chart(fig)

                elif grafico == "ScatterPlot Timeline tuning":
                    col4, col2, col3 = st.columns([1, 8, 1])
                    with col2:
                        st.write(f"Genero {grafico}")
                        if st.session_state.cartellaOutput and st.button(f"Salva {grafico}", key=f"salva_confronto_timeline"):
                            plottimeline_confronto(analizer_confronto, st.session_state.cartellaOutput)
                            st.success(f"✅ - {grafico} di confronto salvato con successo!")
                        else:
                            fig=plottimeline_confronto(analizer_confronto)
                            st.plotly_chart(fig)

    if st.session_state.stato_bottone_spazio_ricerca:     # se sono nella modalità spazio di ricerca, mostro i grafici di spazio di ricerca per tutti gli esperimenti selezionati
        st.subheader("Grafici Spazio di Ricerca")
        with st.spinner('Generando i grafici di Spazio di Ricerca...'):
            for esperimento in st.session_state.lista_out_files:
                if esperimento and esperimento.exp_name in [exp.name for exp in st.session_state.lista_esperimenti] and st.session_state.cartellaOutput and st.button(f"Salva grafici Spazio di Ricerca per {esperimento.exp_name}", key=f"salva_spazio_ricerca_{esperimento.exp_name}"):
                    esperimento.grafici_spazio_ricerca(st.session_state.cartellaOutput)
                    st.success(f"✅ - Grafici Spazio di Ricerca per {esperimento.exp_name} salvati con successo!")
                if esperimento and esperimento.exp_name in [exp.name for exp in st.session_state.lista_esperimenti]:   # se sono nella modalità spazio di ricerca, salvo tutti i grafici di spazio di ricerca degli esperimenti selezionati.
                    st.write(f"Analizzo Spazio di Ricerca per {esperimento.exp_name}")
                    fig=esperimento.grafici_spazio_ricerca()
                    col1, col2, col3 = st.columns([1, 8, 1])
                    with col2:
                        st.plotly_chart(fig)
        

def main():
    #imposto nome dell'applicazione icona e layout(su che spazio si sttruttura la pagina)
    st.set_page_config(page_title="ANALYZER Symbolic DNN Tuner", page_icon="📈", layout="wide")
    
    if 'fase' not in st.session_state:                      #creo una variabile di stato per gestire le fasi dell'applicazione
        st.session_state.fase = 'selezione_cartella'        # Setto fase iniziale: selezione cartelle

    placeholder = st.empty()        # Creo un placeholder per gestire dinamicamente il contenuto della pagina in base alla fase corrente, senza che poi si vedano sotto elementi che non devono essere mostrati in quella fase.
    with placeholder.container():
        
        # Fase 1: Selezione cartelle
        if st.session_state.fase == 'selezione_cartella':
            fase_selezione_cartella()
            
        # Fase 2: Caricamento dei dati
        elif st.session_state.fase == 'caricamento':
            fase_caricamento()            

        # Fase 3: Analisi
        elif st.session_state.fase == 'analisi':
            fase_analisi()

        # Fase Ultima
        elif st.session_state.fase == 'fine':
            st.empty()
            st.title("✅ - Analisi completata!")
            st.write("L'analisi dei risultati è stata completata con successo. ")
            st.balloons()

            col_vuota, col= st.columns([5, 2]) 
            with col:                      # se voglio una nuova analisi posso tornare alla fase iniziale e viene cancellato tutto lo stato
                if st.button('Torna alla fase iniziale',use_container_width=True):
                    st.session_state.fase = 'selezione_cartella'
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.session_state.clear()
                    st.query_params.clear()
                    st.rerun()

if __name__ == "__main__":
    main()