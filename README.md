# Deep Learning Projekt

Dieses Repository enthält ein Deep Learning Projekt mit TensorFlow/Keras als Teil der Angleichungsleistungen im Modul "Data Science und Engineering mit Python".

## Projektüberblick
Das Ziel dieses Projekts ist es, vorherzusagen, ob eine Banknote echt oder eine Fälschung ist, basierend auf Merkmalen, die aus Bildern extrahiert wurden.

## Inhalt
* `Deep_Learning_Solution.ipynb`: Das Haupt-Notebook mit der Implementierung des neuronalen Netzes.
* `bank_note_data.csv`: Der Datensatz, der für Training und Test verwendet wurde.

## Prüfungsaufgabe 2: Automatisierung und Testen

- `model_logic.py`: Enthält die Kernlogik (StandardScaler und MLPClassifier) sowie Logging-Funktionalität.
- `test_model.py`: Führt Unit-Tests zur Validierung der Modellgüte (Accuracy) und der Trainingslaufzeit durch.
- `training.log`: Wird automatisch erstellt und protokolliert Trainingsereignisse.

### Testergebnisse
Die Tests wurden erfolgreich ausgeführt und validieren sowohl die Genauigkeit als auch die Performance:
```text
[Test predict()] Gemessene Accuracy: 1.0000
.
[Test fit()] Gemessene Dauer: 0.1695s (Limit: 0.7500s)
.
----------------------------------------------------------------------
Ran 2 tests in 0.361s

OK
```

## Nutzung
Das Notebook kann direkt über [myBinder](https://mybinder.org/v2/gh/Johannes-Steinle/6-Deep_Learning/main?filepath=Deep_Learning_Solution.ipynb) ausgeführt werden.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Johannes-Steinle/6-Deep_Learning/main?filepath=Deep_Learning_Solution.ipynb)
