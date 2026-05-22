"""
Tests unitarios para el proyecto de detección de anomalías.
"""

import unittest
import numpy as np
import pandas as pd
import os
import sys

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessor import DataPreprocessor, generate_sample_network_data
from src.models import AnomalyDetectionModels
from src.evaluator import ModelEvaluator


class TestDataPreprocessor(unittest.TestCase):
    """Tests para el módulo de preprocesamiento."""
    
    def setUp(self):
        """Configurar datos de prueba."""
        self.df = generate_sample_network_data(n_samples=100)
        self.preprocessor = DataPreprocessor()
    
    def test_generate_sample_data(self):
        """Test para generación de datos sintéticos."""
        df = generate_sample_network_data(n_samples=500)
        self.assertEqual(len(df), 500)
        self.assertIn('label', df.columns)
        self.assertGreater(df['label'].sum(), 0)  # Debe haber anomalías
    
    def test_clean_data(self):
        """Test para limpieza de datos."""
        df_clean = self.preprocessor.clean_data(self.df.copy())
        self.assertEqual(df_clean.isnull().sum().sum(), 0)
    
    def test_encode_categorical(self):
        """Test para codificación de variables categóricas."""
        df_encoded = self.preprocessor.encode_categorical(self.df.copy())
        
        # Verificar que las columnas categóricas fueron codificadas
        categorical_cols = ['protocol_type', 'service', 'flag']
        for col in categorical_cols:
            if col in df_encoded.columns:
                self.assertTrue(df_encoded[col].dtype in [np.int64, np.float64])
    
    def test_scale_features(self):
        """Test para escalado de características."""
        X = np.random.rand(100, 10)
        X_scaled = self.preprocessor.scale_features(X, fit=True)
        
        # Verificar que los datos están escalados (media ~0, std ~1 para StandardScaler)
        self.assertAlmostEqual(np.mean(X_scaled), 0, places=5)
        self.assertAlmostEqual(np.std(X_scaled), 1, places=5)
    
    def test_preprocess(self):
        """Test para preprocesamiento completo."""
        X, y, feature_names = self.preprocessor.preprocess(self.df, target_col='label')
        
        self.assertEqual(X.shape[0], len(self.df))
        self.assertEqual(len(y), len(self.df))
        self.assertGreater(len(feature_names), 0)
        self.assertEqual(len(feature_names), X.shape[1])
    
    def test_save_load_preprocessor(self):
        """Test para guardar y cargar preprocesador."""
        self.preprocessor.preprocess(self.df, target_col='label')
        
        # Guardar
        self.preprocessor.save_preprocessor('test_preprocessor.pkl')
        
        # Cargar
        loaded_preprocessor = DataPreprocessor.load_preprocessor('test_preprocessor.pkl')
        
        # Verificar
        self.assertIsNotNone(loaded_preprocessor.scaler)
        self.assertEqual(len(loaded_preprocessor.feature_names), len(self.preprocessor.feature_names))
        
        # Limpiar
        os.remove('test_preprocessor.pkl')


class TestAnomalyDetectionModels(unittest.TestCase):
    """Tests para el módulo de modelos."""
    
    def setUp(self):
        """Configurar datos de prueba."""
        df = generate_sample_network_data(n_samples=200)
        preprocessor = DataPreprocessor()
        self.X, self.y, _ = preprocessor.preprocess(df, target_col='label')
    
    def test_create_isolation_forest(self):
        """Test para creación de Isolation Forest."""
        detector = AnomalyDetectionModels()
        model = detector.create_isolation_forest(n_estimators=50, contamination=0.1)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.n_estimators, 50)
        self.assertEqual(model.contamination, 0.1)
    
    def test_create_lof(self):
        """Test para creación de LOF."""
        detector = AnomalyDetectionModels()
        model = detector.create_lof(n_neighbors=15, contamination=0.1)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.n_neighbors, 15)
    
    def test_create_elliptic_envelope(self):
        """Test para creación de Elliptic Envelope."""
        detector = AnomalyDetectionModels()
        model = detector.create_elliptic_envelope(contamination=0.1)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.contamination, 0.1)
    
    def test_train_and_predict(self):
        """Test para entrenamiento y predicción."""
        detector = AnomalyDetectionModels()
        detector.create_isolation_forest(contamination=0.1)
        detector.train(self.X)
        
        predictions = detector.predict(self.X)
        
        self.assertEqual(len(predictions), len(self.X))
        # Las predicciones son -1 (anomalía) o 1 (normal)
        self.assertTrue(all(p in [-1, 1] for p in predictions))
    
    def test_get_anomaly_scores(self):
        """Test para obtención de scores de anomalía."""
        detector = AnomalyDetectionModels()
        detector.create_isolation_forest(contamination=0.1)
        detector.train(self.X)
        
        scores = detector.get_anomaly_scores(self.X)
        
        self.assertEqual(len(scores), len(self.X))
        self.assertIsInstance(scores, np.ndarray)
    
    def test_save_load_model(self):
        """Test para guardar y cargar modelo."""
        detector = AnomalyDetectionModels()
        detector.create_isolation_forest(contamination=0.1)
        detector.train(self.X)
        
        # Guardar
        detector.save_model('test_model.pkl')
        
        # Cargar
        loaded_detector = AnomalyDetectionModels.load_model('test_model.pkl', 'isolation_forest')
        
        # Verificar que produce las mismas predicciones
        predictions_original = detector.predict(self.X[:10])
        predictions_loaded = loaded_detector.predict(self.X[:10])
        
        np.testing.assert_array_equal(predictions_original, predictions_loaded)
        
        # Limpiar
        os.remove('test_model.pkl')


class TestModelEvaluator(unittest.TestCase):
    """Tests para el módulo de evaluación."""
    
    def setUp(self):
        """Configurar datos de prueba."""
        np.random.seed(42)
        self.y_true = np.random.randint(0, 2, 100)
        self.y_pred = np.random.randint(0, 2, 100)
        self.y_scores = np.random.rand(100)
        self.evaluator = ModelEvaluator()
    
    def test_evaluate(self):
        """Test para evaluación básica."""
        metrics = self.evaluator.evaluate(self.y_true, self.y_pred)
        
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('confusion_matrix', metrics)
        
        # Verificar rangos válidos
        self.assertGreaterEqual(metrics['accuracy'], 0)
        self.assertLessEqual(metrics['accuracy'], 1)
    
    def test_evaluate_with_scores(self):
        """Test para evaluación con scores."""
        metrics = self.evaluator.evaluate(self.y_true, self.y_pred, self.y_scores)
        
        self.assertIn('roc_auc', metrics)
        self.assertIn('average_precision', metrics)
    
    def test_plot_confusion_matrix(self):
        """Test para graficar matriz de confusión."""
        self.evaluator.evaluate(self.y_true, self.y_pred)
        
        # Graficar sin mostrar (guardar en memoria)
        import matplotlib
        matplotlib.use('Agg')  # Backend no interactivo
        
        try:
            self.evaluator.plot_confusion_matrix(save_path='test_cm.png')
            self.assertTrue(os.path.exists('test_cm.png'))
            os.remove('test_cm.png')
        except Exception as e:
            self.fail(f"plot_confusion_matrix falló: {e}")
    
    def test_get_classification_report(self):
        """Test para obtener reporte de clasificación."""
        self.evaluator.evaluate(self.y_true, self.y_pred)
        report = self.evaluator.get_classification_report()
        
        self.assertIsInstance(report, str)
        self.assertIn('Normal', report)
        self.assertIn('Anomalía', report)
    
    def test_save_results(self):
        """Test para guardar resultados."""
        self.evaluator.evaluate(self.y_true, self.y_pred)
        
        self.evaluator.save_results('test_results.csv')
        self.assertTrue(os.path.exists('test_results.csv'))
        
        # Verificar contenido
        df = pd.read_csv('test_results.csv')
        self.assertIn('Metric', df.columns)
        self.assertIn('Value', df.columns)
        
        os.remove('test_results.csv')


class TestIntegration(unittest.TestCase):
    """Tests de integración para el flujo completo."""
    
    def test_full_pipeline(self):
        """Test para el pipeline completo de entrenamiento y evaluación."""
        # Generar datos
        df = generate_sample_network_data(n_samples=500)
        
        # Preprocesar
        preprocessor = DataPreprocessor()
        X, y, feature_names = preprocessor.preprocess(df, target_col='label')
        
        # Dividir
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Entrenar
        detector = AnomalyDetectionModels()
        detector.create_isolation_forest(contamination=0.1)
        detector.train(X_train)
        
        # Predecir
        predictions = detector.predict(X_test)
        predictions_binary = (predictions == -1).astype(int)
        
        # Evaluar
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_test, predictions_binary)
        
        # Verificar
        self.assertGreater(metrics['accuracy'], 0.5)
        self.assertGreaterEqual(metrics['f1'], 0)
        self.assertLessEqual(metrics['f1'], 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
