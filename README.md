# Deep Learning Projekt

Dieses Repository enthält ein Deep Learning Projekt mit TensorFlow/Keras als Teil der Angleichungsleistungen im Modul "Data Science und Engineering mit Python".

## Projektüberblick
Das Ziel dieses Projekts ist es, vorherzusagen, ob eine Banknote echt oder eine Fälschung ist, basierend auf Merkmalen, die aus Bildern extrahiert wurden.

## Inhalt
* `Deep_Learning_Solution.ipynb`: Das Haupt-Notebook mit der Implementierung des neuronalen Netzes.
* `bank_note_data.csv`: Der Datensatz, der für Training und Test verwendet wurde.

## Prüfungsaufgabe 2: Automatisierung und Testen

Dieses Projekt wurde gemäß den Anforderungen für Aufgabe 2 refaktoriert und mit automatisierten Tests sowie Logging ausgestattet.

### Struktur
- `model_logic.py`: Enthält die Kernlogik (StandardScaler und MLPClassifier) sowie Logging-Funktionalität.
- `test_model.py`: Führt Unit-Tests zur Validierung der Modellgüte (Accuracy) und der Trainingslaufzeit durch.
- `training.log`: Wird automatisch erstellt und protokolliert Trainingsereignisse.

### Testergebnisse
Die Tests wurden erfolgreich ausgeführt:
```text
[Test DL] Gemessene Accuracy: 1.0
.
[Test Fit] Gemessene Dauer: 0.2203s (Limit: 3.0000s)
.
Ran 2 tests in 0.476s
OK
```

## Nutzung
Das Notebook kann direkt über [myBinder](https://mybinder.org/v2/gh/Johannes-Steinle/6-Deep_Learning/main?filepath=Deep_Learning_Solution.ipynb) ausgeführt werden.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Johannes-Steinle/6-Deep_Learning/main?filepath=Deep_Learning_Solution.ipynb)
