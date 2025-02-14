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
Los notebooks en la carpeta

## Resultados

Los resultados de las predicciones y visualizaciones se guardan en la carpeta
