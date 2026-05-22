# Network Anomaly Detection Project

## Descripción
Proyecto para la detección de anomalías en tráfico de red utilizando técnicas de Machine Learning y Deep Learning.

## Estructura del Proyecto

```
network_anomaly_detection/
├── src/                # Código fuente principal
├── data/               # Datos de entrenamiento y prueba
├── models/             # Modelos guardados
├── utils/              # Utilidades y funciones auxiliares
├── tests/              # Tests unitarios
├── notebooks/          # Notebooks de exploración y experimentación
├── requirements.txt    # Dependencias del proyecto
└── README.md          # Este archivo
```

## Características

- **Detección de anomalías**: Identificación de tráfico malicioso o inusual en la red
- **Múltiples algoritmos**: Implementación de diversos algoritmos de ML (Isolation Forest, Autoencoders, etc.)
- **Preprocesamiento**: Herramientas para limpieza y transformación de datos de red
- **Visualización**: Gráficos y dashboards para análisis de resultados
- **Evaluación**: Métricas completas para evaluar el rendimiento del modelo

## Instalación

1. Clonar el repositorio:
```bash
git clone <repository-url>
cd network_anomaly_detection
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso Básico

### Entrenar un modelo
```python
from src.trainer import train_model

model = train_model(data_path='data/train.csv', model_type='isolation_forest')
```

### Detectar anomalías
```python
from src.detector import detect_anomalies

anomalies = detect_anomalies(model, data_path='data/test.csv')
```

## Datasets Soportados

- NSL-KDD
- CICIDS2017
- UNSW-NB15
- Datos personalizados (formato CSV)

## Métricas de Evaluación

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Matriz de confusión

## Autor

Proyecto creado para detección de anomalías en redes.

## Licencia

MIT License
