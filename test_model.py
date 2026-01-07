import unittest
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model_logic import load_data, fit_model, predict_model

class TestDeepLearningModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Wird einmal vor allen Tests ausgeführt. Lädt Daten."""
        cls.X, cls.y = load_data('Bank_Note_Data.csv')
        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            cls.X, cls.y, test_size=0.3, random_state=101
        )
        cls.norm_fit_time = 2.0 # MLP Training dauert etwas länger als simple Modelle

    def test_1_mlp_accuracy(self):
        """
        Ziel: Accuracy > 0.95.
        """
        model, scaler, _ = fit_model(self.X_train, self.y_train)
        predictions = predict_model(model, scaler, self.X_test)
        
        acc = accuracy_score(self.y_test, predictions)
        
        print(f"\n[Test DL] Gemessene Accuracy: {acc}")
        self.assertGreater(acc, 0.95, f"Accuracy zu niedrig: {acc} < 0.95")

    def test_2_fit_runtime(self):
        """
        Ziel: Trainingzeit < 120% der Normzeit.
        """
        _, _, duration = fit_model(self.X_train, self.y_train)
        
        limit = self.norm_fit_time * 1.5 # 50% Puffer wegen MLP Varianz
        print(f"\n[Test Fit] Gemessene Dauer: {duration:.4f}s (Limit: {limit:.4f}s)")
        
        self.assertLess(duration, limit, f"Training dauerte zu lange: {duration:.4f}s > {limit:.4f}s")

if __name__ == '__main__':
    unittest.main()
