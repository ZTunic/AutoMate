import tkinter as tk
import pickle
import numpy as np
import pandas as pd

from regressore import realizza_modello

# Chiamiamo la funzione per creare il modello
realizza_modello()

# Carichiamo da disco il regressore
with open("regressor.pickle", "rb") as file:
    reg = pickle.load(file)

# Creiamo il Form
root = tk.Tk()
root.title("AutoMate")
root.geometry("500x350")

label = tk.Label(root, text="COMPILA IL FORM CON LE INFORMAZIONI DELL'AUTO DI CUI VUOI STIMARE IL PREZZO")
label1 = tk.Label(root, text="")

# Creiamo le etichette per i campi del Form
marca_label = tk.Label(root, text="Marchio dell'auto:")
anno_label = tk.Label(root, text="Anno immatricolazione:")
tipo_veicolo_label = tk.Label(root, text="Categoria dell'auto:")
alimentazione_label = tk.Label(root, text="Alimentazione dell'auto:")
trasmissione_label = tk.Label(root, text="Tipo di trasmissione:")
chilometri_label = tk.Label(root, text="Chilometri percorsi (km):")
potenza_label = tk.Label(root, text="Cavalli (CV):")

# Creiamo i campi per l'input
marca_input = tk.OptionMenu(root, tk.StringVar(), "fiat", "hyundai", "renault", "opel", "skoda", "toyota", "citroen",
                            "mazda",
                            "mitsubishi", "smart", "seat", "chrysler", "nissan", "kia", "chevrolet", "daihatsu",
                            "dacia", "daewoo", "lancia", "trabant",
                            "lada", "volvo", "ford", "volkswagen", "peugeot", "alfa_romeo", "mini", "subaru", "honda",
                            "saab", "suzuki", "jeep", "audi",
                            "bmw", "mercedes_benz", "porsche", "land_rover", "rover", "jaguar")

anno_input = tk.OptionMenu(root, tk.StringVar(), "1986", "1987", "1988", "1989", "1990", "1991", "1992",
                           "1993", "1994", "1995", "1996", "1997", "1998", "1999", "2000", "2001", "2002",
                           "2003", "2004", "2005", "2006", "2007", "2008", "2009", "2010", "2011", "2012",
                           "2013", "2014", "2015", "2016")

tipo_veicolo_input = tk.OptionMenu(root, tk.StringVar(), "Suv", "Berlina", "Station Wagon", "Monovolume", "Utilitaria",
                                   "Cabrio", "Coupe")

alimentazione_input = tk.OptionMenu(root, tk.StringVar(), "Diesel", "Benzina", "Elettrica", "Gpl", "Metano")

trasmissione_input = tk.OptionMenu(root, tk.StringVar(), "Manuale", "Automatico")

chilometri_input = tk.OptionMenu(root, tk.StringVar(), "10000", "20000", "30000", "40000", "50000", "60000", "70000",
                                 "80000", "90000", "100000", "110000", "120000", "130000", "140000", "150000", "160000",
                                 "170000", "180000", "190000", "200000", "210000", "220000", "230000", "240000", "250000",
                                 "260000", "270000", "280000", "290000", "300000", "310000", "320000", "330000", "340000", "350000")

potenza_input = tk.Spinbox(root, from_=50, to=1000)

# Posizioniamo i campi e le loro etichette all'interno di una griglia
label.grid(row=0, column=1, columnspan=5)
label1.grid(row=1, column=1, pady=8)
marca_label.grid(row=4, column=1)
marca_input.grid(row=4, column=2, columnspan=2)
anno_label.grid(row=5, column=1)
anno_input.grid(row=5, column=2, columnspan=2)
tipo_veicolo_label.grid(row=6, column=1)
tipo_veicolo_input.grid(row=6, column=2, columnspan=2)
alimentazione_label.grid(row=7, column=1)
alimentazione_input.grid(row=7, column=2, columnspan=2)
trasmissione_label.grid(row=8, column=1)
trasmissione_input.grid(row=8, column=2, columnspan=2)
chilometri_label.grid(row=9, column=1)
chilometri_input.grid(row=9, column=2, columnspan=2)
potenza_label.grid(row=10, column=1)
potenza_input.grid(row=10, column=2, columnspan=2)

# Creiamo il pulsante per effettuare la predizione
predict_button = tk.Button(root, text="Calcola prezzo")
predict_button.grid(row=14, column=1, columnspan=2, pady=3)


def predict():
    #Ricaviamo gli input dell'utente
    chilometri_percorsi = (int(chilometri_input.cget("text")))
    potenza = np.array(int(potenza_input.get()))
    anni = 2016 - int(anno_input.cget("text"))

    # Creiamo le variabili dummies per tutte le caratteristiche descrittive ottenute dal Form
    trasmissione = trasmissione_input.cget("text")
    trasmissione = [1 if x == trasmissione else 0 for x in ['Manuale']]

    tipo_veicolo = tipo_veicolo_input.cget("text")
    tipo_veicolo = [1 if x == tipo_veicolo else 0 for x in
                    ['Cabrio', 'Coupe', 'Monovolume', 'Station_Wagon', 'Suv', 'Utilitaria']]

    marca = marca_input.cget("text")
    marca = [1 if x == marca else 0 for x in ['audi', 'bmw', 'chevrolet',
                                              'chrysler', 'citroen', 'dacia', 'daewoo',
                                              'daihatsu', 'fiat', 'ford', 'honda',
                                              'hyundai', 'jaguar', 'jeep', 'kia',
                                              'lada', 'lancia', 'land_rover', 'mazda',
                                              'mercedes_benz', 'mini', 'mitsubishi', 'nissan',
                                              'opel', 'peugeot', 'porsche', 'renault',
                                              'rover', 'saab', 'seat', 'skoda', 'smart',
                                              'subaru', 'suzuki', 'toyota', 'trabant',
                                              'volkswagen', 'volvo']]

    alimentazione = alimentazione_input.cget("text")
    alimentazione = [1 if x == alimentazione else 0 for x in ["Diesel", "GPL", "Ibrida", "Metano", "Elettrica"]]

    # Otteniamo un array dalla combinazione delle variabili dummies
    input_features = tipo_veicolo + trasmissione + alimentazione + marca

    # Carichiamo la media e la deviazione standard da disco
    means = np.load("means.npy")
    stddevs = np.load("stddevs.npy")

    # Normalizzimo i nuovi dati utilizzando la normalizzazione Z-Score
    potenza = ((potenza - means[0]) / stddevs[0])
    anni = ((anni - means[1]) / stddevs[1])
    chilometri_percorsi = ((chilometri_percorsi - means[2]) / stddevs[2])

    # Costruiamo un array contenente i risultati normalizzati
    arrayDati = np.array([anni, potenza, chilometri_percorsi])

    # Creiamo un array che contiene i nomi delle colonne del DataFrame
    nomi_colonne = ['anni', 'power_ps', 'odometer_km', 'vehicle_type_Cabrio',
                    'vehicle_type_Coupe', 'vehicle_type_Monovolume',
                    'vehicle_type_Station_Wagon', 'vehicle_type_Suv',
                    'vehicle_type_Utilitaria', 'transmission_Manuale', 'fuel_type_Diesel',
                    'fuel_type_Elettrica', 'fuel_type_Gpl', 'fuel_type_Ibrida',
                    'fuel_type_Metano', 'brand_audi', 'brand_bmw', 'brand_chevrolet',
                    'brand_chrysler', 'brand_citroen', 'brand_dacia', 'brand_daewoo',
                    'brand_daihatsu', 'brand_fiat', 'brand_ford', 'brand_honda',
                    'brand_hyundai', 'brand_jaguar', 'brand_jeep', 'brand_kia',
                    'brand_lada', 'brand_lancia', 'brand_land_rover', 'brand_mazda',
                    'brand_mercedes_benz', 'brand_mini', 'brand_mitsubishi', 'brand_nissan',
                    'brand_opel', 'brand_peugeot', 'brand_porsche', 'brand_renault',
                    'brand_rover', 'brand_saab', 'brand_seat', 'brand_skoda', 'brand_smart',
                    'brand_subaru', 'brand_suzuki', 'brand_toyota', 'brand_trabant',
                    'brand_volkswagen', 'brand_volvo']

    # Concateniamo l'array con i dati numerici normalizzati e con le variabili dummy
    dati_predizione = np.concatenate((arrayDati, input_features))

    # Combiniamo i nomi delle colonne con i corrispondenti valori
    colonne = {nomi_colonne[i]: dati_predizione[i] for i in range(len(nomi_colonne))}

    # Creiamo un DataFrame che mantiene i dati da predire
    df = pd.DataFrame(colonne, index=[0])

    predizione = reg.predict(df)

    predizione = format(predizione[0], '.2f')

    str_risultato = "Un'auto con le caratteristiche fornite ha un valore di € " + predizione
    if float(predizione) <= 0.0:
        str_risultato = "Valore auto: € 0,00"

    # Mostriamo il risultato
    result_label = tk.Label(root, text=str_risultato)
    result_label.grid(row=16, column=1, columnspan=3)


# Associamo la funzione predict all'evento "click" del pulsante
predict_button.config(command=predict)
predict_button.grid(row=14, column=4, columnspan=2)

# Avviiamo la GUI
root.mainloop()
