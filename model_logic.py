import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
import logging
import time
import os

# Logging Konfiguration
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

def load_data(filepath):
    """Lädt die Bank Note Daten."""
    logger = logging.getLogger()
    try:
        data = pd.read_csv(filepath)
        logger.info(f"Daten erfolgreich von {filepath} geladen.")
        
        X = data.drop('Class', axis=1)
        y = data['Class']
        
        return X, y
    except Exception as e:
        logger.error(f"Fehler beim Laden der Daten: {e}")
        raise

def fit_model(X_train, y_train):
    """Skaliert Daten und trainiert ein Neuronales Netz (MLP)."""
    logger = logging.getLogger()
    start_time = time.time()
    
    logger.info("Starte Daten-Skalierung...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    logger.info("Starte MLPClassifier Training...")
    # Wir nutzen MLPClassifier als moderne/kompatible Alternative zum alten DNNClassifier
    model = MLPClassifier(hidden_layer_sizes=(10, 20, 10), max_iter=500, random_state=101)
    model.fit(X_train_scaled, y_train)
    
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Training beendet in {duration:.4f} Sekunden.")
    
    return model, scaler, duration

def predict_model(model, scaler, X_test):
    """Führt Vorhersagen mit dem skalierten Modell durch."""
    logger = logging.getLogger()
    logger.info("Erstelle Vorhersagen...")
    X_test_scaled = scaler.transform(X_test)
    predictions = model.predict(X_test_scaled)
    return predictions

if __name__ == "__main__":
    X, y = load_data('Bank_Note_Data.csv')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    
    model, scaler, duration = fit_model(X_train, y_train)
    preds = predict_model(model, scaler, X_test)
    
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")
    logging.info(f"Deep Learning Accuracy im Testlauf: {acc}")
