"""
Módulo de preprocesamiento de datos para detección de anomalías en red.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Optional, List


class DataPreprocessor:
    """
    Clase para preprocesar datos de tráfico de red.
    
    Maneja:
    - Limpieza de datos (valores nulos, duplicados)
    - Codificación de variables categóricas
    - Escalado de características
    - División train/test
    """
    
    def __init__(self, scaler_type: str = 'standard'):
        """
        Inicializa el preprocesador.
        
        Args:
            scaler_type: Tipo de escalador ('standard' o 'minmax')
        """
        self.scaler_type = scaler_type
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Carga datos desde un archivo CSV.
        
        Args:
            file_path: Ruta al archivo CSV
            
        Returns:
            DataFrame con los datos cargados
        """
        df = pd.read_csv(file_path)
        print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia el DataFrame eliminando valores nulos y duplicados.
        
        Args:
            df: DataFrame a limpiar
            
        Returns:
            DataFrame limpio
        """
        # Eliminar duplicados
        initial_rows = len(df)
        df = df.drop_duplicates()
        print(f"Duplicados eliminados: {initial_rows - len(df)}")
        
        # Imputar valores nulos numéricos con la mediana
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            imputer = SimpleImputer(strategy='median')
            df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        
        # Imputar valores nulos categóricos con el valor más frecuente
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'unknown')
        
        print(f"Valores nulos restantes: {df.isnull().sum().sum()}")
        return df
    
    def encode_categorical(self, df: pd.DataFrame, 
                          categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Codifica variables categóricas usando Label Encoding.
        
        Args:
            df: DataFrame con datos
            categorical_cols: Lista de columnas categóricas (si es None, se detectan automáticamente)
            
        Returns:
            DataFrame con variables codificadas
        """
        if categorical_cols is None:
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                le = LabelEncoder()
                # Manejar valores no vistos en test
                df_encoded[col] = df_encoded[col].astype(str)
                unique_values = df_encoded[col].unique()
                le.fit(unique_values)
                df_encoded[col] = le.transform(df_encoded[col])
                self.label_encoders[col] = le
                print(f"Columna '{col}' codificada con {len(unique_values)} categorías únicas")
        
        return df_encoded
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Escala las características.
        
        Args:
            X: Array de características
            fit: Si True, ajusta el escalador; si False, solo transforma
            
        Returns:
            Array escalado
        """
        if fit:
            if self.scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif self.scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f" scaler_type '{self.scaler_type}' no soportado")
            
            X_scaled = self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise ValueError("El escalador no ha sido ajustado. Ejecuta primero con fit=True")
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def preprocess(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Preprocesa completamente el dataset.
        
        Args:
            df: DataFrame con los datos
            target_col: Nombre de la columna objetivo
            
        Returns:
            Tuple con (X escalado, y, nombres de características)
        """
        # Limpiar datos
        df_clean = self.clean_data(df)
        
        # Separar características y target
        if target_col not in df_clean.columns:
            raise ValueError(f"La columna objetivo '{target_col}' no existe en el DataFrame")
        
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        
        # Codificar categóricas
        X_encoded = self.encode_categorical(X)
        
        # Guardar nombres de características
        self.feature_names = X_encoded.columns.tolist()
        
        # Escalar características
        X_scaled = self.scale_features(X_encoded.values, fit=True)
        
        # Codificar target si es categórico
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y.astype(str))
            self.label_encoders['_target'] = le_target
        
        print(f"Preprocesamiento completado. Features: {X_scaled.shape[1]}, Muestras: {X_scaled.shape[0]}")
        
        return X_scaled, y, self.feature_names
    
    def save_preprocessor(self, path: str):
        """
        Guarda el preprocesador para uso futuro.
        
        Args:
            path: Ruta donde guardar el preprocesador
        """
        import joblib
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'scaler_type': self.scaler_type
        }, path)
        print(f"Preprocesador guardado en {path}")
    
    @classmethod
    def load_preprocessor(cls, path: str) -> 'DataPreprocessor':
        """
        Carga un preprocesador guardado.
        
        Args:
            path: Ruta del preprocesador guardado
            
        Returns:
            Instancia de DataPreprocessor cargada
        """
        import joblib
        data = joblib.load(path)
        preprocessor = cls(scaler_type=data['scaler_type'])
        preprocessor.scaler = data['scaler']
        preprocessor.label_encoders = data['label_encoders']
        preprocessor.feature_names = data['feature_names']
        print(f"Preprocesador cargado desde {path}")
        return preprocessor


def generate_sample_network_data(n_samples: int = 1000) -> pd.DataFrame:
    """
    Genera datos sintéticos de tráfico de red para pruebas.
    
    Args:
        n_samples: Número de muestras a generar
        
    Returns:
        DataFrame con datos sintéticos
    """
    np.random.seed(42)
    
    data = {
        'duration': np.random.exponential(scale=100, size=n_samples),
        'src_bytes': np.random.exponential(scale=500, size=n_samples),
        'dst_bytes': np.random.exponential(scale=500, size=n_samples),
        'count': np.random.poisson(lam=50, size=n_samples),
        'srv_count': np.random.poisson(lam=30, size=n_samples),
        'serror_rate': np.random.uniform(0, 1, size=n_samples),
        'srv_serror_rate': np.random.uniform(0, 1, size=n_samples),
        'rerror_rate': np.random.uniform(0, 1, size=n_samples),
        'srv_rerror_rate': np.random.uniform(0, 1, size=n_samples),
        'same_srv_rate': np.random.uniform(0, 1, size=n_samples),
        'diff_srv_rate': np.random.uniform(0, 1, size=n_samples),
        'srv_diff_host_rate': np.random.uniform(0, 1, size=n_samples),
        'dst_host_count': np.random.poisson(lam=20, size=n_samples),
        'dst_host_srv_count': np.random.poisson(lam=15, size=n_samples),
        'dst_host_same_srv_rate': np.random.uniform(0, 1, size=n_samples),
        'dst_host_diff_srv_rate': np.random.uniform(0, 1, size=n_samples),
        'dst_host_same_src_port_rate': np.random.uniform(0, 1, size=n_samples),
        'dst_host_srv_diff_host_rate': np.random.uniform(0, 1, size=n_samples),
        'protocol_type': np.random.choice(['tcp', 'udp', 'icmp'], size=n_samples),
        'service': np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'dns', 'other'], size=n_samples),
        'flag': np.random.choice(['SF', 'S0', 'REJ', 'RSTO', 'SH'], size=n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generar labels (0: normal, 1: anomalía)
    # Las anomalías tendrán valores extremos en algunas características
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    labels = np.zeros(n_samples)
    labels[anomaly_indices] = 1
    
    # Modificar características para las anomalías
    for idx in anomaly_indices:
        if np.random.random() > 0.5:
            df.loc[idx, 'src_bytes'] *= np.random.uniform(5, 10)
            df.loc[idx, 'dst_bytes'] *= np.random.uniform(5, 10)
        else:
            df.loc[idx, 'count'] *= np.random.uniform(5, 10)
            df.loc[idx, 'srv_count'] *= np.random.uniform(5, 10)
    
    df['label'] = labels.astype(int)
    
    print(f"Datos sintéticos generados: {n_samples} muestras, {int(labels.sum())} anomalías ({labels.mean()*100:.1f}%)")
    
    return df


if __name__ == '__main__':
    # Ejemplo de uso
    print("Generando datos de prueba...")
    df = generate_sample_network_data(1000)
    
    print("\nPreprocesando datos...")
    preprocessor = DataPreprocessor(scaler_type='standard')
    X, y, feature_names = preprocessor.preprocess(df, target_col='label')
    
    print(f"\nCaracterísticas: {len(feature_names)}")
    print(f"Muestras: {X.shape[0]}")
    print(f"Distribución de clases: {np.bincount(y)}")
    
    # Guardar preprocesador
    preprocessor.save_preprocessor('preprocessor.pkl')
    
    print("\n¡Preprocesamiento completado exitosamente!")
