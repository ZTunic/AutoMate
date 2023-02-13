import numpy as np
import pandas as pd
import pickle
from scipy.stats import zscore
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics

def realizza_modello():
    # Importiamo il file .csv contenente i dati
    df = pd.read_csv('../autos_random_50k_cleaned.csv')

    # Diamo un nome alla colonna degli Id
    df.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)

    # Settiamo gli Id del DataFrame considerando la colonna degli Id del Dataset
    df.set_index('Id', inplace=True)

    # Ricaviamo dalla colonna dell'anno di immatricolazione la colonna degli anni dell'auto
    df['registration_year'] = 2016 - df['registration_year']

    # Rinominiamo la colonna dell'anno di immatricolazione
    df.rename(columns={'registration_year': 'anni'}, inplace=True)

    # Rimuoviamo le auto d'epoca
    df = df[(df['anni'] >= 0) & (df['anni'] <= 30)]

    # Traduciamo le stringhe presenti all'interno del Dataset da tedesco a italiano
    traduzione_tipo_veicolo = {
        "kombi": "Station_Wagon",
        "bus": "Monovolume",
        "kleinwagen": "Utilitaria",
        "limousine": "Berlina",
        "cabrio": "Cabrio",
        "coupe": "Coupe",
        "suv": "Suv"
    }

    traduzione_tipo_trasmissione = {
        "manuell": "Manuale",
        "automatik": "Automatica"
    }

    traduzione_tipo_alimentazione = {
        "benzin": "Benzina",
        "diesel": "Diesel",
        "lpg": "Gpl",
        "cng": "Metano",
        "hybrid": "Ibrida",
        "elektro": "Elettrica"
    }

    # Applichiamo le traduzioni alle colonne del Dataset coinvolte
    df["vehicle_type"] = df["vehicle_type"].map(traduzione_tipo_veicolo)
    df["transmission"] = df["transmission"].map(traduzione_tipo_trasmissione)
    df["fuel_type"] = df["fuel_type"].map(traduzione_tipo_alimentazione)

    # Calcoliamo la media e la deviazione standard per le variabili numeriche
    means = np.mean(df[['power_ps', 'anni', 'odometer_km']], axis=0)
    stddevs = np.std(df[['power_ps', 'anni', 'odometer_km']], axis=0)

    # Salviamo la media e la deviazione standard in un file da recuperare per l'app Desktop
    np.save("means.npy", means)
    np.save("stddevs.npy", stddevs)

    # Selezioniamo il sottoinsieme di features da normalizzare
    toNormalize = df[['power_ps', 'anni', 'odometer_km']]

    # Sovrascriviamo le colonne del Dataset originale
    df[['power_ps', 'anni', 'odometer_km']] = toNormalize.apply(zscore)

    # Riduciamo il Dataset alle sole Features selezionate e la variabile Target
    df = df[['price_EUR', 'vehicle_type', 'anni', 'transmission', 'power_ps', 'odometer_km', 'fuel_type', 'brand']]

    # Rimuoviamo le righe del Dataset con valori "Unknown" o "altro (andere/sonstige)" (sconosciuti)
    df = df[df != "Unknown"]
    df = df.dropna(axis=0)
    df = df[df != "andere"]
    df = df.dropna(axis=0)
    df = df[df != "sonstige_autos"]
    df = df.dropna(axis=0)

    # Creiamo un sottoinsieme di variabili qualitative
    var_qualitative = df.select_dtypes(include=['object'])
    # Convertiamole in variabili dummies
    var_dummies = pd.get_dummies(var_qualitative, drop_first=True)
    # Rimuoviamo dal dataset le colonne qualitative originali
    df = df.drop(list(var_qualitative.columns), axis=1)
    # Inseriamo le variabili qualitative all'interno del Dataset
    df = pd.concat([df, var_dummies], axis=1)

    # Selezioniamo il sottoinsieme di features che il modello utilizzerà per le sue predizioni
    x = df.drop(columns="price_EUR", axis=1)

    # Isoliamo la variabile dipendente
    y = df["price_EUR"]

    # Creiamo il modello di regressione lineare
    reg = linear_model.LinearRegression(fit_intercept=True)

    # Inizializziamo la 10-fold cross validation
    ten_fold = model_selection.KFold(n_splits=10, shuffle=True, random_state=0)

    # Creiamo un Array in cui memorizzeremo i MAE ad ogni iterazione
    array_MAE = []

    # Eseguiamo la convalida incrociata
    for train_index, test_index in ten_fold.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Addestriamo il modello con la partizione di training corrente
        reg.fit(x_train, y_train)

        # Effettuiamo la predizione
        y_pred = reg.predict(x_test)

        # Calcoliamo il MAE per questa iterazione
        MAE = metrics.mean_absolute_error(y_test, y_pred)
        array_MAE.append(MAE)

    # Calcoliamo la media dei MAE ottenuti in ogni iterazione
    mean_MAE = sum(array_MAE) / len(array_MAE)

    print("Media MAE:", mean_MAE)

    # Registriamo su disco il regressore
    with open("regressor.pickle", "wb") as file:
        pickle.dump(reg, file)

