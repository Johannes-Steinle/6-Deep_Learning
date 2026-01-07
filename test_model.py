import unittest
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model_logic import load_data, fit_model, predict_model

class TestDeepLearningModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Wird einmal vor allen Tests ausgeführt. Lädt Daten."""
        cls.X, cls.y = load_data('bank_note_data.csv')
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.3, random_state=101
        )
        cls.norm_fit_time = 0.5 # Erwartete Normzeit für dieses Modell

    def test_1_predict_accuracy(self):
        """
        Aufgabe: Test der Vorhersagefunktion (predict).
        Ziel: Accuracy > 0.95.
        """
        model, scaler = fit_model(self.X_train, self.y_train)
        predictions = predict_model(model, scaler, self.X_test)
        
        acc = accuracy_score(self.y_test, predictions)
        
        print(f"\n[Test predict()] Gemessene Accuracy: {acc:.4f}")
        self.assertGreater(acc, 0.95, f"Accuracy zu niedrig: {acc} < 0.95")

    def test_2_fit_runtime(self):
        """
        Aufgabe: Überprüfung der Laufzeit der Trainingsfunktion (fit).
        Ziel: Laufzeit < 150% der Normzeit.
        """
        start_time = time.time()
        fit_model(self.X_train, self.y_train)
        duration = time.time() - start_time
        
        limit = self.norm_fit_time * 1.5 
        print(f"\n[Test fit()] Gemessene Dauer: {duration:.4f}s (Limit: {limit:.4f}s)")
        
        self.assertLess(duration, limit, f"Training dauerte zu lange: {duration:.4f}s > {limit:.4f}s")

if __name__ == '__main__':
    unittest.main()
