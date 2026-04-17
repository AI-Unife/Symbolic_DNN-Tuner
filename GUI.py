from graficifun import *
import streamlit as st
import tkinter as tk
from tkinter import filedialog

@st.cache_data          # Cache dei dati in modo tale che a ogni iterazione con la pagina non venga ricaricato tutto da zero
def caricamento():
    return analyze_all_experiments(st.session_state.cartellaInput, st.session_state.cartellaOutput) #carico nome degli esperimenti e analizer corrispondenti alla cartella selezionata

def main():
    root = tk.Tk()  # Creo una finestra di tkinter per poter utilizzare il filedialog: seleziona cartella 
    root.withdraw() # Nascondo la finesstra
    root.wm_attributes('-topmost', 1)       # Faccio in modo che una volta chiamato filedialog la finestra venga portata in primo piano

    #imposto nome dell'applicazione icona e layout(su che spazio si sttruttura la pagina)
    st.set_page_config(page_title="ANALYZER Symbolic DNN Tuner", page_icon="📈", layout="wide")
    
    if 'fase' not in st.session_state:                      #creo una variabile di stato per gestire le fasi dell'applicazione
        st.session_state.fase = 'selezione_cartella'        # Setto fase iniziale: selezione cartelle

    placeholder = st.empty()        # Creo un placeholder per gestire dinamicamente il contenuto della pagina in base alla fase corrente, senza che poi si vedano sotto elementi che non devono essere mostrati in quella fase.
    with placeholder.container():
        
        # Fase 1: Selezione cartelle
        if st.session_state.fase == 'selezione_cartella':
            st.empty()      # Pulisco il placeholder per rimuovere eventuali elementi di diverse fasi
            st.title("📈 - ANALYZER Symbolic DNN Tuner")
            
            # Sezione per la cartella da analizzare
            st.subheader("📂 - Selettore cartella da analizzare")   #sottotitolo
            cola, colb = st.columns([2,5])         #creo due colonne e adatto il bottone alla prima colonna per farlo uguale a quello sottostante
            with cola:
                if st.button('Seleziona cartella da analizzare ->',use_container_width=True):   #quando viene premuto il bottone
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
                        st.rerun()  # Ricarica la pagina per passare alla fase successiva
                    else:
                        st.error("Per favore, seleziona una cartella da analizzare prima di continuare.", icon="❌")

        
        # Fase 2: Caricamento dei dati
        elif st.session_state.fase == 'caricamento':
            st.empty()            
            st.title("⚙️ - Caricamento dei dati")

            with st.spinner('Caricando i vari esperimenti...'):     # Mostro uno spinner mentre vengono caricati i dati, in modo da far capire all'utente che l'applicazione sta lavorando e non è bloccata
                lista_esperimenti, lista_analizer = caricamento ()  # Carico i dati degli esperimenti e gli analizer corrispondenti alla cartella selezionata, usando la funzione caricamento che è stata decorata con st.cache_data per evitare di ricaricare tutto da zero a ogni iterazione con la pagina
                if lista_esperimenti and lista_analizer:            #ho bisogno di far diventare queste due liste variabili di stato per poterle usare nelle fasi successive
                    st.session_state.lista_esperimenti = lista_esperimenti
                    st.session_state.lista_analizer = lista_analizer

            if not lista_esperimenti:       #mostro un messaggio di warning se non sono ststi trovati esperimenti
                st.warning("❌ - Non sono stati trovati esperimenti nella cartella selezionata. Per favore, verifica che la cartella contenga i dati corretti e riprova.")
                
            if lista_esperimenti:
                st.write(f"Trovati {len(lista_esperimenti)} esperimenti nella cartella selezionata: ")      #mostro cosa ho trovato
                for exp in lista_esperimenti:
                    st.write(f"- {exp.name}")

                if 'scelta_comune' not in st.session_state:     # Creo una variabile di stato per memorizzare una scelta comune in modo tale che se avessi molti esperimenti la scelta dei grafici da visulizzare possa essere più veloce 
                    st.session_state.scelta_comune = None

                grafici_da_fare = []        # mi devo tenere traccia per ogni esperimento cosa graficare
                opzioni_grafici = [         #grafici possibili
                    "LinePlot di Accuracy, Score e Params", 
                    "BarPlot efficacia azioni", 
                    "BarPlot diagnosi", 
                    "BarPlot tuning", 
                    "ScatterPlot Timeline tuning"]

                for i, exp in enumerate(lista_esperimenti):
                    # Se è stata impostata una scelta comune, la usiamo per tutti gli esperimenti senza chiedere
                    if st.session_state.scelta_comune is not None:
                        grafici = st.session_state.scelta_comune
                        st.write(f"✅ {exp.name}: Applicata scelta comune: {', '.join(grafici)}")
                    else:
                        # Altrimenti chiediamo all'utente
                        grafici = st.multiselect(       # Apro un menu di selezione multipla
                            f"Quali grafici devo generare per {exp.name}?",
                            opzioni_grafici,
                            key=f"ms_{exp.name}" #importante mettere una chiave diversa per ogni multiselect in modo tale che Streamlit riesca a distinguere i diversi menu
                        )
                        
                        # Mostriamo il bottone applica a tutti solo se l'utente ha selezionato qualcosa nel primo esperimento. Gli altri non hanno bottone
                        if i == 0 and grafici:                    
                            if st.button(f"Applica a tutti", key=f"btn_{i}"):
                                st.session_state.scelta_comune = grafici
                                st.rerun()

                    grafici_da_fare.append(grafici)     # tengo traccia per ogni esperimento (indice) quali grafici fare 
                
                st.session_state.lista_grafici = grafici_da_fare   #me lo devo salvare in una variabile di stato sennò nella fase di creazione dei grafici non riesco più a reperire l'informazione              

            colno,colok= st.columns([6, 2]) 
            with colok:
                if st.session_state.lista_grafici and st.button('Crea grafici'):  # se ho dei grafici da creare mostro il bottone per andare alla fase successiva 
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

        # Fase 3: Analisi
        elif st.session_state.fase == 'analisi':
            st.empty()
            st.title("🔍 - Analisi dei risultati")

            col1,col2= st.columns([8, 2]) 
            with col1:
                if st.session_state.cartellaOutput and st.button('Salva tutto'):    # se è stata selezionata una cartella di output mostro un bottone per salvare tutti i grafici in un colpo solo, altrimenti non mostro questo bottone
                    for i in range(len(st.session_state.lista_grafici)):
                        grafici = st.session_state.lista_grafici[i]
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

                    st.success("✅ - Tutti i grafici sono stati salvati con successo nella cartella di output selezionata!")
                    st.session_state.cartellaOutput = None  # Resetta la cartella di output dopo il salvataggio, una volta che ho salvato tutto non mi serve salvare ancora 
                    
            with col2:              # se una volta visti non voglio fare altre azioni ho la possibilità di andare alla fase successiva
                if st.button('Fine'):
                    st.session_state.fase = 'fine'
                    st.rerun()  # Ricarica la pagina per passare alla fase successiva
            
            with st.spinner('Generando i grafici...'):      # generazione vera e propria dei grafici
                for i in range(len(st.session_state.lista_grafici)):    # corro con ciclo 

                    grafici = st.session_state.lista_grafici[i]     #capisco cosa devo graficare
                    exp = st.session_state.lista_esperimenti[i]     # per quale esperimento
                    if grafici:                                     # se ho grafici
                        st.write(f"Analizzo {exp.name}")            #analizzo
                        for grafico in grafici:
                            st.write(f"- Genenro {grafico} ")
                            analizer = st.session_state.lista_analizer[i]

                            if grafico == "LinePlot di Accuracy, Score e Params":       # serie di if-elif per capire quale grafico fare 
                                fig=plotaccuracy(analizer,exp, None)                    # creo il grafico senza salvarlo
                                st.pyplot(fig,clear_figure=True)                        # lo mostro a video
                                if st.session_state.cartellaOutput and st.button(f"Salva {grafico} per {exp.name}", key=f"salva_{i}_accuracy"):         # creo un bottone che mi possa salvare quel grafico specifico
                                    plotaccuracy(analizer,exp, st.session_state.cartellaOutput)     #ricreo il grafico salvandolo questa volta
                                    st.success(f"✅ - {grafico} per {exp.name} salvato con successo!")      # mostro un messaggio che mi indica che il grafico è stato salvato bene

                            elif grafico == "BarPlot efficacia azioni":
                                col1, col2, col3 = st.columns([1, 3, 1])
                                with col2:
                                    fig=plotevidence(analizer,exp, None)
                                    st.pyplot(fig,clear_figure=True)
                                    if st.session_state.cartellaOutput and st.button(f"Salva {grafico} per {exp.name}", key=f"salva_{i}_evidence"):
                                        plotevidence(analizer,exp, st.session_state.cartellaOutput)
                                        st.success(f"✅ - {grafico} per {exp.name} salvato con successo!")

                            elif grafico == "BarPlot diagnosi":
                                fig=plotdiagnosis(analizer,exp, None)
                                st.pyplot(fig,clear_figure=True)
                                if st.session_state.cartellaOutput and st.button(f"Salva {grafico} per {exp.name}", key=f"salva_{i}_diagnosis"):
                                    plotdiagnosis(analizer,exp, st.session_state.cartellaOutput)
                                    st.success(f"✅ - {grafico} per {exp.name} salvato con successo!")

                            elif grafico == "BarPlot tuning":
                                fig=plottuning(analizer,exp,None)
                                st.pyplot(fig,clear_figure=True)
                                if st.session_state.cartellaOutput and st.button(f"Salva {grafico} per {exp.name}", key=f"salva_{i}_tuning"):
                                    plottuning(analizer,exp, st.session_state.cartellaOutput)
                                    st.success(f"✅ - {grafico} per {exp.name} salvato con successo!")

                            elif grafico == "ScatterPlot Timeline tuning":
                                col1, col2, col3 = st.columns([1, 3, 1])
                                with col2:
                                    fig=plottimeline(analizer,exp, None)
                                    st.pyplot(fig,clear_figure=True)
                                    if st.session_state.cartellaOutput and st.button(f"Salva {grafico} per {exp.name}", key=f"salva_{i}_timeline"):
                                        plottimeline(analizer,exp, st.session_state.cartellaOutput)
                                        st.success(f"✅ - {grafico} per {exp.name} salvato con successo!")

                    else:
                        st.write(f"Nessun grafico selezionato per {exp.name}, salto l'analisi.")        # avviso che per un esperimento non è ststo selezionato nulla e proseguo


        # Fase Ultima
        elif st.session_state.fase == 'fine':
            st.empty()
            st.title("✅ - Analisi completata!")
            st.write("L'analisi dei risultati è stata completata con successo. ")
            st.balloons()

            col_vuota2, col3= st.columns([5, 2]) 
            with col3:                      # se voglio una nuova analisi posso tornare alla fase iniziale e viene cancellato tutto lo stato
                if st.button('Torna alla fase iniziale',use_container_width=True):
                    st.session_state.fase = 'selezione_cartella'
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.session_state.clear()
                    st.query_params.clear()
                    st.rerun()

if __name__ == "__main__":
    main()
