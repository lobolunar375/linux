"""
Notebook de ejemplo para demostración del sistema de detección de anomalías.
Este script ejecuta un flujo completo de entrenamiento y evaluación.
"""

def run_demo():
    """Ejecuta una demostración completa del sistema."""
    
    print("=" * 70)
    print("DEMOSTRACIÓN: SISTEMA DE DETECCIÓN DE ANOMALÍAS EN RED")
    print("=" * 70)
    
    # Paso 1: Generar datos sintéticos
    print("\n" + "=" * 70)
    print("PASO 1: GENERACIÓN DE DATOS SINTÉTICOS")
    print("=" * 70)
    
    from src.preprocessor import generate_sample_network_data
    
    df = generate_sample_network_data(n_samples=5000)
    print(f"\nDatos generados:")
    print(f"  - Total muestras: {len(df)}")
    print(f"  - Características: {df.shape[1] - 1}")
    print(f"  - Anomalías: {df['label'].sum()} ({df['label'].mean()*100:.2f}%)")
    
    # Mostrar primeras filas
    print("\nPrimeras 5 filas del dataset:")
    print(df.head())
    
    # Paso 2: Preprocesamiento
    print("\n" + "=" * 70)
    print("PASO 2: PREPROCESAMIENTO DE DATOS")
    print("=" * 70)
    
    from src.preprocessor import DataPreprocessor
    
    preprocessor = DataPreprocessor(scaler_type='standard')
    X, y, feature_names = preprocessor.preprocess(df, target_col='label')
    
    print(f"\nDespués del preprocesamiento:")
    print(f"  - Features: {len(feature_names)}")
    print(f"  - Muestras: {X.shape[0]}")
    print(f"  - Distribución de clases: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Paso 3: División train/test
    print("\n" + "=" * 70)
    print("PASO 3: DIVISIÓN DE DATOS (TRAIN/TEST)")
    print("=" * 70)
    
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nDatos de entrenamiento: {X_train.shape[0]} muestras")
    print(f"Datos de test: {X_test.shape[0]} muestras")
    print(f"Distribución en train - Normal: {sum(y_train==0)}, Anomalía: {sum(y_train==1)}")
    print(f"Distribución en test - Normal: {sum(y_test==0)}, Anomalía: {sum(y_test==1)}")
    
    # Paso 4: Entrenar múltiples modelos
    print("\n" + "=" * 70)
    print("PASO 4: ENTRENAMIENTO DE MODELOS")
    print("=" * 70)
    
    from src.models import AnomalyDetectionModels
    
    models_config = [
        ('isolation_forest', {'n_estimators': 100, 'contamination': 0.1}),
        ('lof', {'n_neighbors': 20, 'contamination': 0.1}),
        ('elliptic_envelope', {'contamination': 0.1}),
    ]
    
    results = {}
    
    for model_name, params in models_config:
        print(f"\n--- Entrenando {model_name.upper()} ---")
        detector = AnomalyDetectionModels()
        
        if model_name == 'isolation_forest':
            detector.create_isolation_forest(**params)
        elif model_name == 'lof':
            detector.create_lof(**params)
        elif model_name == 'elliptic_envelope':
            detector.create_elliptic_envelope(**params)
        
        detector.train(X_train)
        
        # Predecir
        predictions = detector.predict(X_test)
        predictions_binary = (predictions == -1).astype(int)
        
        results[model_name] = {
            'detector': detector,
            'predictions': predictions_binary
        }
    
    # Paso 5: Evaluar modelos
    print("\n" + "=" * 70)
    print("PASO 5: EVALUACIÓN DE MODELOS")
    print("=" * 70)
    
    from src.evaluator import ModelEvaluator, compare_models
    
    metrics_dict = {}
    
    for model_name, data in results.items():
        print(f"\n{'='*50}")
        print(f"Evaluando {model_name.upper()}")
        print('='*50)
        
        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(y_test, data['predictions'])
        
        metrics_dict[model_name] = metrics
    
    # Comparar modelos
    print("\n" + "=" * 70)
    print("COMPARACIÓN DE MODELOS")
    print("=" * 70)
    
    comparison_df = compare_models(metrics_dict)
    
    # Encontrar el mejor modelo
    best_model = comparison_df['f1'].idxmax()
    best_f1 = comparison_df.loc[best_model, 'f1']
    
    print(f"\n🏆 MEJOR MODELO: {best_model.upper()} con F1-Score = {best_f1:.4f}")
    
    # Paso 6: Guardar el mejor modelo
    print("\n" + "=" * 70)
    print("PASO 6: GUARDAR MODELO Y RESULTADOS")
    print("=" * 70)
    
    import os
    import joblib
    
    output_dir = 'demo_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar preprocesador
    preprocessor.save_preprocessor(os.path.join(output_dir, 'preprocessor.pkl'))
    
    # Guardar mejor modelo
    best_detector = results[best_model]['detector']
    best_detector.save_model(
        os.path.join(output_dir, f'{best_model}_model.pkl'),
        model_name=best_model
    )
    
    # Guardar métricas
    import json
    metrics_path = os.path.join(output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        # Convertir a formato serializable
        serializable_metrics = {}
        for model, mets in metrics_dict.items():
            serializable_metrics[model] = {}
            for k, v in mets.items():
                if isinstance(v, (int, float)) or v is None:
                    serializable_metrics[model][k] = v
                elif hasattr(v, 'tolist'):
                    serializable_metrics[model][k] = v.tolist()
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"\nArchivos guardados en '{output_dir}/':")
    print(f"  ✓ preprocessor.pkl")
    print(f"  ✓ {best_model}_model.pkl")
    print(f"  ✓ metrics.json")
    
    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)
    
    print(f"\n✅ Demostración completada exitosamente!")
    print(f"\nEstadísticas clave:")
    print(f"  • Dataset: {len(df)} muestras, {len(feature_names)} características")
    print(f"  • Train/Test: {len(X_train)}/{len(X_test)} muestras")
    print(f"  • Modelos probados: {len(models_config)}")
    print(f"  • Mejor modelo: {best_model.upper()}")
    print(f"  • F1-Score: {best_f1:.4f}")
    print(f"  • ROC-AUC: {metrics_dict[best_model].get('roc_auc', 'N/A')}")
    
    print("\n" + "=" * 70)
    print("¡GRACIAS POR USAR EL SISTEMA DE DETECCIÓN DE ANOMALÍAS!")
    print("=" * 70)
    
    return {
        'data': df,
        'preprocessor': preprocessor,
        'models': results,
        'metrics': metrics_dict,
        'best_model': best_model
    }


if __name__ == '__main__':
    import numpy as np
    results = run_demo()
