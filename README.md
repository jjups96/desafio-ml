# Desafío ML

Este proyecto es una prueba técnica para Mercado Libre que incluye la implementación de modelos de forecasting y clasificación de etiquetas utilizando diversas técnicas de machine learning.

## Estructura del Proyecto

| Carpeta               | Función |
|-----------------------|---------|
| `data/`              | Contiene los datasets. |
| `notebooks/`         | Análisis exploratorio y tuning de modelos. |
| `results/`           | Reportes y visualizaciones generadas. |
| `src/models/`        | Modelos de forecasting y clasificación. |
| `src/pipeline/`      | Entrenamiento y evaluación de modelos. |
| `src/preprocessing/` | Limpieza de datos y generación de features. |
| `src/test/`          | Pruebas unitarias. |
| `src/utils/`         | Funciones auxiliares como visualización, guardado de resultados e interfaces. |

## Instalación

1. Clona el repositorio:
```sh
git clone https://github.com/jjups96/desafio-ml.git
cd desafio_ml
```

2. Instala las dependencias usando Poetry:
```sh
poetry install
```

## Uso

### Forecasting

Para ejecutar el pipeline de forecasting, utiliza el siguiente comando:
```sh
python main_forecasting_mod.py
```

## Clasificación de Etiquetas
Para ejecutar el pipeline de clasificación de etiquetas, utiliza el siguiente comando:

```sh
python main_zero_tags_solid.py
```

## Notebooks
Los notebooks en la carpeta notebook proporcionan análisis exploratorio de datos (EDA) y ajuste de hiperparámetros:
- `forecasting_eda.ipynb:` Análisis exploratorio de datos para forecasting.
- `tuning.ipynb:` Ajuste de hiperparámetros para los modelos de forecasting.

## Resultados

Los resultados de las predicciones y visualizaciones se guardan en la carpeta results:
- `forecasting_plot_errors.png`: Comparación de errores (MAE y RMSE) entre modelos.
- `forecasting_plot_predictions.png`: Predicciones de los modelos vs. ventas históricas.
- `forecasting_results.csv:` Resultados de las predicciones de los modelos.
- `tag_class_zero_shoot_results.csv`: Resultados de la clasificación de etiquetas.
- `tag_classification_heatmap.png`: Heatmap de probabilidades de clasificación por categoría.
