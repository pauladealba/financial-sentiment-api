# Financial Sentiment Analysis API

Este proyecto es una API local creada con FastAPI para hacer análisis de sentimientos financieros usando un modelo transformer.

## Dataset

Se usó el dataset de Kaggle:

sbhatti/financial-sentiment-analysis

Variable X: sentence  
Variable Y: sentiment

## Archivos principales

- main.py: contiene la API con FastAPI.
- train_model.py: descarga el dataset, entrena el modelo y lo guarda.
- requirements.txt: contiene las librerías necesarias.
- .gitignore: evita subir carpetas pesadas como venv, model y results.

## Instalación

Crear entorno virtual:

python -m venv venv

Activar entorno virtual en Windows:

venv\Scripts\activate

Instalar dependencias:

pip install -r requirements.txt

## Entrenar modelo

python train_model.py

## Ejecutar API

uvicorn main:app --reload

Abrir en el navegador:

http://127.0.0.1:8000/docs

## Endpoint principal

POST /predict

Ejemplo de entrada:

{
  "sentence": "The company reported strong profits and revenue growth this quarter."
}

Ejemplo de salida:

{
  "sentence": "The company reported strong profits and revenue growth this quarter.",
  "sentiment": "positive",
  "confidence": 0.990551233291626
}

## Tecnologías

Python, FastAPI, Uvicorn, Transformers, PyTorch, Pandas, Scikit-learn y KaggleHub.
