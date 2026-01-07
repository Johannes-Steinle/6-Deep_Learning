# Deep Learning Projekt

Meine Umsetzung der Deep Learning Übung aus dem Udemy-Kurs "Python für Data Science, Maschinelles Lernen & Visualization" im Rahmen der Angleichungsleistung.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Johannes-Steinle/6-Deep_Learning/main?filepath=Deep_Learning_Solution.ipynb)

## Überblick
Vorhersage, ob eine Banknote echt oder gefälscht ist, anhand von Merkmalen, die aus Bildern extrahiert wurden. Modell: MLPClassifier / Neuronales Netz (scikit-learn).

## Inhalt
* `Deep_Learning_Solution.ipynb` - Haupt-Notebook mit der Implementierung des neuronalen Netzes
* `bank_note_data.csv` - Datensatz mit Banknoten-Merkmalen (1372 Einträge)

## Ausführung

1. Auf den **Binder-Badge** oben klicken, um das Notebook in myBinder zu starten.
2. Warten, bis die Umgebung geladen ist (kann 1-2 Minuten dauern).
3. `Deep_Learning_Solution.ipynb` öffnen.
4. Alle Zellen nacheinander ausführen (*Run > Run All Cells*).
5. **Erwartete Ergebnisse:**
   - Analyse der Banknoten-Merkmale
   - Feature-Skalierung mit StandardScaler
   - Training des neuronalen Netzes (Multi-Layer Perceptron)
   - Classification Report mit sehr hoher Precision und Recall
   - Accuracy von ca. **0.99 - 1.00**

---

## Prüfungsaufgabe 2: Automatisierung und Testen

Ich habe das Projekt für Aufgabe 2 um Unit-Tests und Logging erweitert, nach dem Ansatz aus dem Artikel "Unit Testing and Logging for Data Science".

### Dateien
| Datei | Beschreibung |
|---|---|
| `model_logic.py` | StandardScaler + MLPClassifier Logik mit `my_logger` und `my_timer` Dekoratoren |
| `test_model.py` | Unit-Tests für `predict()` (Accuracy) und `fit()` (Laufzeit) |
| `generate_test_data.py` | Skript zur Erzeugung der Testdaten |
| `train_data.csv` | Trainingsdaten (960 Zeilen) |
| `test_data.csv` | Testdaten (412 Zeilen) |
| `training.log` | Log-File mit Trainingsereignissen |

### Testfälle

**Testfall 1 - predict():** Das neuronale Netz wird auf `train_data.csv` trainiert und die Accuracy auf `test_data.csv` geprüft. Ziel: Accuracy > 0.95.

**Testfall 2 - fit():** Die Laufzeit der Trainingsfunktion wird gemessen und geprüft, ob sie unter 120% der Normzeit (0.5s) bleibt.

### Testergebnisse
```text
[Test predict()] Gemessene Accuracy: 1.0000
.
[Test fit()] Gemessene Dauer: 0.1607s (Limit: 0.6000s)
.
----------------------------------------------------------------------
Ran 2 tests in 0.330s

OK
```

### Tests ausführen

1. Binder-Umgebung über den Badge oben starten.
2. **Terminal** öffnen (*File > New > Terminal*).
3. Folgenden Befehl ausführen:
   ```bash
   python -m unittest test_model -v
   ```
4. Die Tests laden die Daten aus `test_data.csv` und `train_data.csv`.
5. Beide Tests sollten mit `OK` durchlaufen.

Um die Testdaten neu zu generieren: `python generate_test_data.py`
