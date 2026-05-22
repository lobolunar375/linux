"""
Módulo de evaluación para modelos de detección de anomalías.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, precision_recall_curve,
    average_precision_score
)
from typing import Tuple, Dict, Optional
import pandas as pd


class ModelEvaluator:
    """
    Clase para evaluar modelos de detección de anomalías.
    
    Proporciona métricas completas y visualizaciones.
    """
    
    def __init__(self):
        """Inicializa el evaluador."""
        self.metrics = {}
        self.y_true = None
        self.y_pred = None
        self.y_scores = None
        
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                 y_scores: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evalúa el modelo con múltiples métricas.
        
        Args:
            y_true: Labels verdaderos (0: normal, 1: anomalía)
            y_pred: Labels predichos (0: normal, 1: anomalía)
            y_scores: Scores de predicción (probabilidades o scores continuos)
            
        Returns:
            Diccionario con todas las métricas
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_scores = y_scores
        
        # Métricas básicas
        self.metrics['accuracy'] = accuracy_score(y_true, y_pred)
        self.metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        self.metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        self.metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # Métricas con scores (si están disponibles)
        if y_scores is not None:
            try:
                self.metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
                self.metrics['average_precision'] = average_precision_score(y_true, y_scores)
            except Exception as e:
                print(f"Advertencia: No se pudo calcular algunas métricas: {e}")
                self.metrics['roc_auc'] = None
                self.metrics['average_precision'] = None
        else:
            self.metrics['roc_auc'] = None
            self.metrics['average_precision'] = None
        
        # Matriz de confusión
        self.metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # Estadísticas adicionales
        self.metrics['total_samples'] = len(y_true)
        self.metrics['true_anomalies'] = int(np.sum(y_true))
        self.metrics['predicted_anomalies'] = int(np.sum(y_pred))
        self.metrics['false_positives'] = int(np.sum((y_pred == 1) & (y_true == 0)))
        self.metrics['false_negatives'] = int(np.sum((y_pred == 0) & (y_true == 1)))
        self.metrics['true_positives'] = int(np.sum((y_pred == 1) & (y_true == 1)))
        self.metrics['true_negatives'] = int(np.sum((y_pred == 0) & (y_true == 0)))
        
        print("=== Métricas de Evaluación ===")
        print(f"Accuracy:           {self.metrics['accuracy']:.4f}")
        print(f"Precision:          {self.metrics['precision']:.4f}")
        print(f"Recall:             {self.metrics['recall']:.4f}")
        print(f"F1-Score:           {self.metrics['f1']:.4f}")
        if self.metrics['roc_auc'] is not None:
            print(f"ROC-AUC:            {self.metrics['roc_auc']:.4f}")
        if self.metrics['average_precision'] is not None:
            print(f"Average Precision:  {self.metrics['average_precision']:.4f}")
        print(f"\nTotal muestras:     {self.metrics['total_samples']}")
        print(f"Anomalías reales:   {self.metrics['true_anomalies']}")
        print(f"Anomalías predichas:{self.metrics['predicted_anomalies']}")
        print(f"Verdaderos positivos: {self.metrics['true_positives']}")
        print(f"Falsos positivos:     {self.metrics['false_positives']}")
        print(f"Falsos negativos:     {self.metrics['false_negatives']}")
        
        return self.metrics
    
    def plot_confusion_matrix(self, save_path: Optional[str] = None,
                             show_labels: bool = True):
        """
        Grafica la matriz de confusión.
        
        Args:
            save_path: Ruta para guardar la figura (si None, muestra en pantalla)
            show_labels: Si True, muestra valores en las celdas
        """
        cm = self.metrics.get('confusion_matrix')
        if cm is None:
            raise ValueError("Primero debes ejecutar evaluate()")
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=show_labels, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Anomalía'],
                   yticklabels=['Normal', 'Anomalía'])
        plt.title('Matriz de Confusión')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Matriz de confusión guardada en {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_roc_curve(self, save_path: Optional[str] = None):
        """
        Grafica la curva ROC.
        
        Args:
            save_path: Ruta para guardar la figura (si None, muestra en pantalla)
        """
        if self.y_scores is None:
            raise ValueError("Se necesitan y_scores para graficar la curva ROC")
        
        try:
            fpr, tpr, _ = roc_curve(self.y_true, self.y_scores)
            roc_auc = self.metrics.get('roc_auc', 0)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2,
                    label=f'Curva ROC (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Tasa de Falsos Positivos')
            plt.ylabel('Tasa de Verdaderos Positivos')
            plt.title('Curva ROC')
            plt.legend(loc='lower right')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Curva ROC guardada en {save_path}")
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            print(f"Error al graficar curva ROC: {e}")
    
    def plot_precision_recall_curve(self, save_path: Optional[str] = None):
        """
        Grafica la curva Precision-Recall.
        
        Args:
            save_path: Ruta para guardar la figura (si None, muestra en pantalla)
        """
        if self.y_scores is None:
            raise ValueError("Se necesitan y_scores para graficar la curva PR")
        
        try:
            precision, recall, _ = precision_recall_curve(self.y_true, self.y_scores)
            avg_precision = self.metrics.get('average_precision', 0)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='blue', lw=2,
                    label=f'Curva PR (AP = {avg_precision:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Curva Precision-Recall')
            plt.legend(loc='lower left')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Curva PR guardada en {save_path}")
            else:
                plt.show()
            
            plt.close()
        except Exception as e:
            print(f"Error al graficar curva PR: {e}")
    
    def plot_all_metrics(self, save_dir: Optional[str] = None):
        """
        Genera todas las visualizaciones.
        
        Args:
            save_dir: Directorio para guardar las figuras
        """
        import os
        
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        print("Generando visualizaciones...")
        
        self.plot_confusion_matrix(
            save_path=os.path.join(save_dir, 'confusion_matrix.png') if save_dir else None
        )
        
        if self.y_scores is not None:
            self.plot_roc_curve(
                save_path=os.path.join(save_dir, 'roc_curve.png') if save_dir else None
            )
            self.plot_precision_recall_curve(
                save_path=os.path.join(save_dir, 'precision_recall_curve.png') if save_dir else None
            )
        
        print("Visualizaciones generadas exitosamente")
    
    def get_classification_report(self) -> str:
        """
        Obtiene un reporte de clasificación detallado.
        
        Returns:
            String con el reporte de clasificación
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("Primero debes ejecutar evaluate()")
        
        report = classification_report(
            self.y_true, self.y_pred,
            target_names=['Normal', 'Anomalía'],
            digits=4
        )
        return report
    
    def save_results(self, path: str):
        """
        Guarda los resultados en un archivo CSV.
        
        Args:
            path: Ruta del archivo CSV
        """
        results_df = pd.DataFrame({
            'Metric': list(self.metrics.keys()),
            'Value': list(self.metrics.values())
        })
        results_df.to_csv(path, index=False)
        print(f"Resultados guardados en {path}")


def compare_models(results_dict: Dict[str, Dict[str, float]],
                  save_path: Optional[str] = None):
    """
    Compara múltiples modelos y genera una tabla comparativa.
    
    Args:
        results_dict: Diccionario con nombre_modelo -> métricas
        save_path: Ruta para guardar la tabla comparativa
    """
    metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    comparison_data = {}
    for model_name, metrics in results_dict.items():
        comparison_data[model_name] = {
            metric: metrics.get(metric, 0) or 0
            for metric in metrics_to_compare
        }
    
    comparison_df = pd.DataFrame(comparison_data).T
    
    print("\n=== Comparación de Modelos ===")
    print(comparison_df.round(4))
    
    # Encontrar el mejor modelo para cada métrica
    print("\n=== Mejores Modelos por Métrica ===")
    for metric in metrics_to_compare:
        best_model = comparison_df[metric].idxmax()
        best_value = comparison_df[metric].max()
        print(f"{metric:15s}: {best_model} ({best_value:.4f})")
    
    if save_path:
        comparison_df.to_csv(save_path)
        print(f"\nTabla comparativa guardada en {save_path}")
    
    return comparison_df


if __name__ == '__main__':
    # Ejemplo de uso
    from src.preprocessor import generate_sample_network_data, DataPreprocessor
    from src.models import AnomalyDetectionModels
    from sklearn.model_selection import train_test_split
    
    print("Generando datos de prueba...")
    df = generate_sample_network_data(2000)
    
    print("\nPreprocesando datos...")
    preprocessor = DataPreprocessor()
    X, y, _ = preprocessor.preprocess(df, target_col='label')
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Entrenar y evaluar Isolation Forest
    print("\n=== Entrenando Isolation Forest ===")
    detector_if = AnomalyDetectionModels()
    detector_if.create_isolation_forest(contamination=0.1)
    detector_if.train(X_train)
    predictions_if = detector_if.predict(X_test)
    predictions_if_binary = (predictions_if == -1).astype(int)
    
    evaluator_if = ModelEvaluator()
    metrics_if = evaluator_if.evaluate(y_test, predictions_if_binary)
    
    # Guardar resultados
    evaluator_if.save_results('evaluation_results.csv')
    evaluator_if.plot_confusion_matrix(save_path='confusion_matrix.png')
    
    print("\n¡Evaluación completada!")
