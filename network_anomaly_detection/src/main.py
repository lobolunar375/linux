"""
Script principal para entrenar y evaluar modelos de detección de anomalías.
"""

import argparse
import os
import json
from pathlib import Path


def train_model(data_path: str, model_type: str = 'isolation_forest',
                contamination: float = 0.1, output_dir: str = 'models',
                test_size: float = 0.3, random_state: int = 42):
    """
    Entrena un modelo de detección de anomalías.
    
    Args:
        data_path: Ruta al archivo CSV con los datos
        model_type: Tipo de modelo ('isolation_forest', 'lof', 'one_class_svm', 'elliptic_envelope')
        contamination: Proporción esperada de anomalías
        output_dir: Directorio para guardar modelos y resultados
        test_size: Proporción de datos para test
        random_state: Semilla para reproducibilidad
    """
    from src.preprocessor import DataPreprocessor
    from src.models import AnomalyDetectionModels
    from src.evaluator import ModelEvaluator
    from sklearn.model_selection import train_test_split
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("ENTRENAMIENTO DE MODELO DE DETECCIÓN DE ANOMALÍAS")
    print("=" * 60)
    
    # Cargar y preprocesar datos
    print(f"\n[1/5] Cargando datos desde {data_path}...")
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data(data_path)
    
    # Detectar columna objetivo
    target_candidates = ['label', 'Label', 'target', 'class', 'anomaly']
    target_col = None
    for candidate in target_candidates:
        if candidate in df.columns:
            target_col = candidate
            break
    
    if target_col is None:
        raise ValueError(f"No se encontró columna objetivo. Columnas disponibles: {df.columns.tolist()}")
    
    print(f"Columna objetivo detectada: {target_col}")
    
    # Preprocesar
    print(f"\n[2/5] Preprocesando datos...")
    X, y, feature_names = preprocessor.preprocess(df, target_col=target_col)
    
    # Guardar preprocesador
    preprocessor_path = os.path.join(output_dir, 'preprocessor.pkl')
    preprocessor.save_preprocessor(preprocessor_path)
    
    # Dividir datos
    print(f"\n[3/5] Dividiendo datos (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {X_train.shape[0]} muestras, Test: {X_test.shape[0]} muestras")
    print(f"Distribución en train - Normal: {sum(y_train==0)}, Anomalía: {sum(y_train==1)}")
    print(f"Distribución en test - Normal: {sum(y_test==0)}, Anomalía: {sum(y_test==1)}")
    
    # Crear y entrenar modelo
    print(f"\n[4/5] Entrenando modelo {model_type}...")
    detector = AnomalyDetectionModels()
    
    if model_type == 'isolation_forest':
        detector.create_isolation_forest(contamination=contamination, random_state=random_state)
    elif model_type == 'lof':
        detector.create_lof(contamination=contamination)
    elif model_type == 'one_class_svm':
        detector.create_one_class_svm(nu=contamination)
    elif model_type == 'elliptic_envelope':
        detector.create_elliptic_envelope(contamination=contamination, random_state=random_state)
    else:
        raise ValueError(f"Modelo '{model_type}' no soportado. Opciones: isolation_forest, lof, one_class_svm, elliptic_envelope")
    
    detector.train(X_train)
    
    # Guardar modelo
    model_path = os.path.join(output_dir, f'{model_type}_model.pkl')
    detector.save_model(model_path)
    
    # Evaluar
    print(f"\n[5/5] Evaluando modelo...")
    predictions = detector.predict(X_test)
    predictions_binary = (predictions == -1).astype(int)
    
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(y_test, predictions_binary)
    
    # Guardar resultados
    results_path = os.path.join(output_dir, 'evaluation_metrics.json')
    with open(results_path, 'w') as f:
        # Convertir tipos no serializables
        serializable_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str)) or v is None:
                serializable_metrics[k] = v
            elif hasattr(v, 'tolist'):
                serializable_metrics[k] = v.tolist()
            else:
                serializable_metrics[k] = str(v)
        json.dump(serializable_metrics, f, indent=2)
    
    # Generar visualizaciones
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    evaluator.plot_confusion_matrix(save_path=os.path.join(viz_dir, 'confusion_matrix.png'))
    
    print("\n" + "=" * 60)
    print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    print(f"\nArchivos guardados en {output_dir}/:")
    print(f"  - preprocessor.pkl: Preprocesador guardado")
    print(f"  - {model_type}_model.pkl: Modelo entrenado")
    print(f"  - evaluation_metrics.json: Métricas de evaluación")
    print(f"  - visualizations/: Gráficos y visualizaciones")
    
    return detector, preprocessor, metrics


def detect_anomalies(model_path: str, preprocessor_path: str, 
                     data_path: str, output_path: str = 'anomalies.csv'):
    """
    Detecta anomalías en nuevos datos usando un modelo entrenado.
    
    Args:
        model_path: Ruta al modelo entrenado
        preprocessor_path: Ruta al preprocesador guardado
        data_path: Ruta a los nuevos datos
        output_path: Ruta para guardar resultados
    """
    import pandas as pd
    from src.preprocessor import DataPreprocessor
    from src.models import AnomalyDetectionModels
    
    print("=" * 60)
    print("DETECCIÓN DE ANOMALÍAS")
    print("=" * 60)
    
    # Cargar modelo y preprocesador
    print(f"\n[1/3] Cargando modelo y preprocesador...")
    
    # Determinar tipo de modelo desde el nombre del archivo
    model_name = os.path.basename(model_path).replace('_model.pkl', '')
    detector = AnomalyDetectionModels.load_model(model_path, model_name)
    preprocessor = DataPreprocessor.load_preprocessor(preprocessor_path)
    
    # Cargar nuevos datos
    print(f"\n[2/3] Cargando datos desde {data_path}...")
    df_new = pd.read_csv(data_path)
    print(f"Datos cargados: {df_new.shape[0]} filas, {df_new.shape[1]} columnas")
    
    # Preprocesar (usando el preprocesador ya ajustado)
    print(f"\n[3/3] Detectando anomalías...")
    
    # Eliminar columna target si existe
    target_candidates = ['label', 'Label', 'target', 'class', 'anomaly']
    df_for_prediction = df_new.copy()
    for col in target_candidates:
        if col in df_for_prediction.columns:
            df_for_prediction = df_for_prediction.drop(columns=[col])
    
    # Codificar categóricas
    df_encoded = preprocessor.encode_categorical(df_for_prediction)
    
    # Escalar
    X_scaled = preprocessor.scale_features(df_encoded.values, fit=False)
    
    # Predecir
    predictions = detector.predict(X_scaled)
    predictions_binary = (predictions == -1).astype(int)
    
    # Obtener scores
    scores = detector.get_anomaly_scores(X_scaled)
    
    # Guardar resultados
    df_results = df_new.copy()
    df_results['is_anomaly'] = predictions_binary
    df_results['anomaly_score'] = scores
    
    df_results.to_csv(output_path, index=False)
    
    # Estadísticas
    total = len(predictions_binary)
    anomalies = sum(predictions_binary)
    percentage = (anomalies / total) * 100
    
    print("\n" + "=" * 60)
    print("DETECCIÓN COMPLETADA")
    print("=" * 60)
    print(f"\nResultados:")
    print(f"  Total muestras: {total}")
    print(f"  Anomalías detectadas: {anomalies}")
    print(f"  Porcentaje: {percentage:.2f}%")
    print(f"\nResultados guardados en: {output_path}")
    
    return df_results


def main():
    parser = argparse.ArgumentParser(
        description='Detección de anomalías en tráfico de red'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # Comando train
    train_parser = subparsers.add_parser('train', help='Entrenar un modelo')
    train_parser.add_argument('--data', type=str, required=True,
                             help='Ruta al archivo CSV con datos de entrenamiento')
    train_parser.add_argument('--model', type=str, default='isolation_forest',
                             choices=['isolation_forest', 'lof', 'one_class_svm', 'elliptic_envelope'],
                             help='Tipo de modelo a entrenar')
    train_parser.add_argument('--contamination', type=float, default=0.1,
                             help='Proporción esperada de anomalías')
    train_parser.add_argument('--output', type=str, default='models',
                             help='Directorio para guardar el modelo')
    train_parser.add_argument('--test-size', type=float, default=0.3,
                             help='Proporción de datos para test')
    
    # Comando detect
    detect_parser = subparsers.add_parser('detect', help='Detectar anomalías en nuevos datos')
    detect_parser.add_argument('--model', type=str, required=True,
                              help='Ruta al modelo entrenado')
    detect_parser.add_argument('--preprocessor', type=str, required=True,
                              help='Ruta al preprocesador guardado')
    detect_parser.add_argument('--data', type=str, required=True,
                              help='Ruta a los nuevos datos')
    detect_parser.add_argument('--output', type=str, default='anomalies.csv',
                              help='Ruta para guardar resultados')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(
            data_path=args.data,
            model_type=args.model,
            contamination=args.contamination,
            output_dir=args.output,
            test_size=args.test_size
        )
    elif args.command == 'detect':
        detect_anomalies(
            model_path=args.model,
            preprocessor_path=args.preprocessor,
            data_path=args.data,
            output_path=args.output
        )
    else:
        parser.print_help()
        print("\nEjemplos de uso:")
        print("  python main.py train --data data/train.csv --model isolation_forest")
        print("  python main.py detect --model models/isolation_forest_model.pkl --preprocessor models/preprocessor.pkl --data data/new_data.csv")


if __name__ == '__main__':
    main()
