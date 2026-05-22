"""
Módulo de modelos para detección de anomalías en red.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from typing import Dict, Optional, Tuple, Union
import joblib


class AnomalyDetectionModels:
    """
    Clase que implementa múltiples algoritmos de detección de anomalías.
    
    Algoritmos soportados:
    - Isolation Forest
    - One-Class SVM
    - Local Outlier Factor (LOF)
    - Elliptic Envelope
    - Autoencoder (Keras/TensorFlow)
    """
    
    def __init__(self):
        """Inicializa el diccionario de modelos."""
        self.models = {}
        self.model_type = None
        
    def create_isolation_forest(self, n_estimators: int = 100, 
                                 contamination: float = 0.1,
                                 random_state: int = 42,
                                 **kwargs) -> IsolationForest:
        """
        Crea un modelo Isolation Forest.
        
        Args:
            n_estimators: Número de árboles
            contamination: Proporción esperada de anomalías
            random_state: Semilla para reproducibilidad
            **kwargs: Argumentos adicionales para IsolationForest
            
        Returns:
            Modelo IsolationForest configurado
        """
        model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            **kwargs
        )
        self.models['isolation_forest'] = model
        self.model_type = 'isolation_forest'
        print(f"Isolation Forest creado con {n_estimators} estimadores")
        return model
    
    def create_one_class_svm(self, kernel: str = 'rbf', 
                              nu: float = 0.1,
                              gamma: str = 'scale',
                              **kwargs) -> OneClassSVM:
        """
        Crea un modelo One-Class SVM.
        
        Args:
            kernel: Tipo de kernel ('linear', 'poly', 'rbf', 'sigmoid')
            nu: Parámetro de suavizado
            gamma: Parámetro del kernel
            **kwargs: Argumentos adicionales para OneClassSVM
            
        Returns:
            Modelo OneClassSVM configurado
        """
        model = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma,
            **kwargs
        )
        self.models['one_class_svm'] = model
        self.model_type = 'one_class_svm'
        print(f"One-Class SVM creado con kernel={kernel}, nu={nu}")
        return model
    
    def create_lof(self, n_neighbors: int = 20,
                   contamination: float = 0.1,
                   metric: str = 'minkowski',
                   **kwargs) -> LocalOutlierFactor:
        """
        Crea un modelo Local Outlier Factor.
        
        Args:
            n_neighbors: Número de vecinos
            contamination: Proporción esperada de anomalías
            metric: Métrica de distancia
            **kwargs: Argumentos adicionales para LOF
            
        Returns:
            Modelo LOF configurado
        """
        model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            metric=metric,
            novelty=True,  # Para poder predecir en nuevos datos
            **kwargs
        )
        self.models['lof'] = model
        self.model_type = 'lof'
        print(f"LOF creado con {n_neighbors} vecinos")
        return model
    
    def create_elliptic_envelope(self, contamination: float = 0.1,
                                  random_state: int = 42,
                                  **kwargs) -> EllipticEnvelope:
        """
        Crea un modelo Elliptic Envelope.
        
        Args:
            contamination: Proporción esperada de anomalías
            random_state: Semilla para reproducibilidad
            **kwargs: Argumentos adicionales para EllipticEnvelope
            
        Returns:
            Modelo EllipticEnvelope configurado
        """
        model = EllipticEnvelope(
            contamination=contamination,
            random_state=random_state,
            **kwargs
        )
        self.models['elliptic_envelope'] = model
        self.model_type = 'elliptic_envelope'
        print(f"Elliptic Envelope creado con contamination={contamination}")
        return model
    
    def train(self, X: np.ndarray, model_name: Optional[str] = None) -> object:
        """
        Entrena el modelo seleccionado.
        
        Args:
            X: Datos de entrenamiento (features escaladas)
            model_name: Nombre del modelo a usar (si None, usa el último creado)
            
        Returns:
            Modelo entrenado
        """
        if model_name is None:
            if not self.models:
                raise ValueError("No hay modelos creados. Crea un modelo primero.")
            model_name = self.model_type
        
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' no encontrado. Modelos disponibles: {list(self.models.keys())}")
        
        model = self.models[model_name]
        print(f"Entrenando {model_name} con {X.shape[0]} muestras...")
        
        # Ajustar el modelo
        model.fit(X)
        
        print(f"Modelo {model_name} entrenado exitosamente")
        return model
    
    def predict(self, X: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """
        Predice anomalías en los datos.
        
        Args:
            X: Datos a evaluar (features escaladas)
            model_name: Nombre del modelo a usar
            
        Returns:
            Array con predicciones (-1: anomalía, 1: normal)
        """
        if model_name is None:
            model_name = self.model_type
        
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' no encontrado")
        
        model = self.models[model_name]
        predictions = model.predict(X)
        
        return predictions
    
    def get_anomaly_scores(self, X: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """
        Obtiene scores de anomalía para los datos.
        
        Args:
            X: Datos a evaluar
            model_name: Nombre del modelo a usar
            
        Returns:
            Array con scores de anomalía (valores más bajos = más anómalos)
        """
        if model_name is None:
            model_name = self.model_type
        
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' no encontrado")
        
        model = self.models[model_name]
        
        # Diferentes modelos tienen diferentes métodos para scores
        if hasattr(model, 'decision_function'):
            scores = model.decision_function(X)
        elif hasattr(model, 'score_samples'):
            scores = model.score_samples(X)
        else:
            # Para LOF y otros, usar negative_outlier_factor_
            if model_name == 'lof':
                # Necesitamos ajustar temporalmente para obtener scores
                model_copy = LocalOutlierFactor(
                    n_neighbors=model.n_neighbors,
                    contamination=model.contamination,
                    novelty=False
                )
                model_copy.fit_predict(X)
                scores = model_copy.negative_outlier_factor_
            else:
                scores = -model.predict(X)  # Convertir a score continuo
        
        return scores
    
    def save_model(self, path: str, model_name: Optional[str] = None):
        """
        Guarda un modelo entrenado.
        
        Args:
            path: Ruta donde guardar el modelo
            model_name: Nombre del modelo a guardar
        """
        if model_name is None:
            model_name = self.model_type
        
        if model_name not in self.models:
            raise ValueError(f"Modelo '{model_name}' no encontrado")
        
        joblib.dump(self.models[model_name], path)
        print(f"Modelo {model_name} guardado en {path}")
    
    @classmethod
    def load_model(cls, path: str, model_name: str) -> 'AnomalyDetectionModels':
        """
        Carga un modelo guardado.
        
        Args:
            path: Ruta del modelo guardado
            model_name: Nombre para identificar el modelo
            
        Returns:
            Instancia de AnomalyDetectionModels con el modelo cargado
        """
        instance = cls()
        instance.models[model_name] = joblib.load(path)
        instance.model_type = model_name
        print(f"Modelo {model_name} cargado desde {path}")
        return instance


def create_autoencoder(input_dim: int, 
                       encoding_dim: int = 32,
                       learning_rate: float = 0.001) -> 'tf.keras.Model':
    """
    Crea un autoencoder para detección de anomalías.
    
    Args:
        input_dim: Dimensión de entrada
        encoding_dim: Dimensión del espacio latente
        learning_rate: Tasa de aprendizaje
        
    Returns:
        Modelo autoencoder compilado
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers, models
    except ImportError:
        raise ImportError("TensorFlow/Keras es necesario para usar autoencoders. "
                         "Instala con: pip install tensorflow")
    
    # Encoder
    encoder_input = keras.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    encoded = layers.Dense(encoding_dim, activation='relu')(x)
    
    # Decoder
    x = layers.Dense(64, activation='relu')(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    decoded = layers.Dense(input_dim, activation='linear')(x)
    
    # Autoencoder completo
    autoencoder = models.Model(encoder_input, decoded)
    autoencoder.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse'
    )
    
    print(f"Autoencoder creado: {input_dim} -> {encoding_dim} -> {input_dim}")
    
    return autoencoder


def train_autoencoder(X_train: np.ndarray,
                      X_val: np.ndarray,
                      epochs: int = 50,
                      batch_size: int = 32,
                      encoding_dim: int = 32,
                      learning_rate: float = 0.001,
                      early_stopping_patience: int = 10) -> Tuple['tf.keras.Model', dict]:
    """
    Entrena un autoencoder para detección de anomalías.
    
    Args:
        X_train: Datos de entrenamiento
        X_val: Datos de validación
        epochs: Número de épocas
        batch_size: Tamaño de batch
        encoding_dim: Dimensión del espacio latente
        learning_rate: Tasa de aprendizaje
        early_stopping_patience: Paciencia para early stopping
        
    Returns:
        Tuple con (modelo entrenado, historial de entrenamiento)
    """
    from tensorflow import keras
    
    # Crear modelo
    model = create_autoencoder(
        input_dim=X_train.shape[1],
        encoding_dim=encoding_dim,
        learning_rate=learning_rate
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7
        )
    ]
    
    # Entrenar
    print(f"Entrenando autoencoder por {epochs} épocas...")
    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print("Autoencoder entrenado exitosamente")
    
    return model, history.history


def detect_anomalies_autoencoder(model: 'tf.keras.Model',
                                  X: np.ndarray,
                                  threshold_percentile: float = 95) -> np.ndarray:
    """
    Detecta anomalías usando un autoencoder entrenado.
    
    Args:
        model: Autoencoder entrenado
        X: Datos a evaluar
        threshold_percentile: Percentil para determinar el threshold
        
    Returns:
        Array con predicciones (1: anomalía, 0: normal)
    """
    # Reconstruir datos
    reconstructions = model.predict(X, verbose=0)
    
    # Calcular error de reconstrucción
    reconstruction_errors = np.mean(np.square(X - reconstructions), axis=1)
    
    # Determinar threshold
    threshold = np.percentile(reconstruction_errors, threshold_percentile)
    
    # Predecir anomalías
    predictions = (reconstruction_errors > threshold).astype(int)
    
    print(f"Threshold: {threshold:.4f}, Anomalías detectadas: {predictions.sum()} ({predictions.mean()*100:.2f}%)")
    
    return predictions


if __name__ == '__main__':
    # Ejemplo de uso
    from src.preprocessor import generate_sample_network_data, DataPreprocessor
    
    print("Generando datos de prueba...")
    df = generate_sample_network_data(1000)
    
    print("\nPreprocesando datos...")
    preprocessor = DataPreprocessor()
    X, y, _ = preprocessor.preprocess(df, target_col='label')
    
    # Dividir en train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print("\n=== Probando Isolation Forest ===")
    detector = AnomalyDetectionModels()
    model_if = detector.create_isolation_forest(contamination=0.1)
    detector.train(X_train)
    predictions_if = detector.predict(X_test)
    
    # Convertir predicciones a binario (1: anomalía, 0: normal)
    predictions_binary = (predictions_if == -1).astype(int)
    
    # Evaluar
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nReporte de clasificación:")
    print(classification_report(y_test, predictions_binary, target_names=['Normal', 'Anomalía']))
    
    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, predictions_binary))
    
    print("\n¡Modelo de detección de anomalías probado exitosamente!")
