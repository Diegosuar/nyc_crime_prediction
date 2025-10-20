# NYC Crime Analytics - Predicción de Crímenes Violentos

Este proyecto implementa un pipeline completo de ciencia de datos para analizar datos de denuncias criminales en Nueva York y predecir si un incidente reportado corresponde a un **crimen violento (Felony) o no violento (Misdemeanor/Violation)**. Utiliza un modelo avanzado de Machine Learning (XGBoost) entrenado con características geoespaciales y temporales, logrando una alta precisión. Los resultados se presentan a través de una aplicación web interactiva construida con Flask.

El flujo de trabajo está orquestado con **Prefect** para garantizar la reproducibilidad y la automatización del preprocesamiento de datos y el entrenamiento del modelo.

---
## 🎯 Objetivo y Pregunta de Investigación

El objetivo principal es responder a la pregunta:

> **"Dadas las características de un incidente reportado en NYC (ubicación, hora, tipo de lugar, tipo de ofensa), ¿podemos predecir con alta precisión si se trata de un crimen violento (Felony)?"**

Este enfoque permite evaluar el riesgo potencial de los incidentes y podría ser útil para la asignación de recursos de seguridad pública.

---
## 🚀 Características Principales

* **Dashboard de Analítica**: Página principal con métricas clave del modelo (Precisión ~94.5%), gráficos sobre la distribución de crímenes originales y análisis de los lugares con mayor incidencia.
* **Herramienta de Predicción**: Permite al usuario ingresar detalles de un incidente hipotético (ubicación, hora, día, mes, tipo de lugar, tipo de ofensa) para obtener una predicción en tiempo real de la **probabilidad de que sea un crimen violento (Felony)**.
* **Mapa de Densidad**: Visualización geoespacial interactiva que muestra las "zonas calientes" de denuncias en la ciudad.
* **Pipeline Automatizado con Prefect**: Proceso ETL robusto que incluye:
    * Ingesta de datos de denuncias (dataset principal).
    * Ingeniería de características avanzada (cíclicas temporales, flags de fin de semana/noche, clustering geoespacial K-Means).
    * Balanceo de clases con SMOTE.
    * Entrenamiento optimizado con XGBoost y RandomizedSearchCV.
* **Alta Precisión**: El modelo final alcanza una **precisión general del 94.50%** en la predicción de crímenes violentos vs. no violentos.

---
## 📊 Datasets Utilizados

* **Dataset Principal:** [NYPD Complaint Data Historic](https://data.cityofnewyork.us/resource/qgea-i56i.csv) - Contiene los registros históricos de denuncias, incluyendo tipo de ofensa, categoría legal (Felony, Misdemeanor, Violation), ubicación y fecha/hora. Es la fuente para las características y la variable objetivo (`is_violent`).
* **(Opcional/Exploratorio):** Se exploró la integración con datos de arrestos y paradas de vehículos, pero no se utilizaron directamente en las características del modelo final debido a la dificultad para establecer una unión fiable o la falta de disponibilidad de datos consistentes.

---
## 📁 Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

```bash
/proyecto_crimen_nyc
|
|-- app/
|   |-- static/   
|  
    |   |-- css/   
|   |-- templates/ 
|   |-- app.py     
|
|-- data/          
|   |-- raw/       
|
|-- models/        
|   |-- crime_predictor_model.joblib
|   |-- scaler.joblib
|   |-- label_encoder.joblib
|
|-- reports/      
|   |-- dashboard_metrics.json
|
|-- src/            
|   |-- data_ingestion.py
|   |-- preprocessing.py 
|   |-- train.py      
|   |-- evaluate.py   
|   |-- pipeline.py   
|
|-- requirements.txt  
|-- run_app.py        
|-- README.md         
```

## 🛠️ Tecnologías Utilizadas

* Backend: Python, Flask
* Orquestación de Datos: Prefect
* Machine Learning: Scikit-learn, XGBoost, Imbalanced-learn (para SMOTE)
* Manipulación de Datos: Pandas, numpy
* Visualización: Matplotlib, Seaborn, Folium
* Entorno: Conda / venv

## ⚙️ Cómo Correr el Proyecto
Sigue estos pasos para configurar y ejecutar el proyecto en tu máquina local.

1. Prerrequisitos
    - Tener Python 3.9 o superior instalado.
    - Se recomienda usar un entorno virtual (como venv o conda) para aislar las dependencias del proyecto.

2. Clonar el Repositorio
    - Abre una terminal y clona este repositorio (o simplemente descarga y descomprime el código en una carpeta).

```Bash 
git clone <URL_DEL_REPOSITORIO>
cd proyecto_crimen_nyc
```

3. Configurar el Entorno Virtual
Crea y activa un entorno virtual.

```Bash

# Usando venv
python -m venv env
source env/bin/activate  # En Windows: env\Scripts\activate

# O usando conda
conda create -n ml-final python=3.9
conda activate ml-final
```

4. Instalar Dependencias
Instala todas las librerías necesarias con el siguiente comando:

```Bash
pip install -r requirements.txt
```

5. Ejecutar el Pipeline de Datos (Paso Crucial)
Antes de poder usar la aplicación web, debes ejecutar el pipeline de Prefect. Este paso se encarga de descargar los datos, limpiarlos y entrenar el modelo de Machine Learning.

Este comando solo necesita ser ejecutado una vez (o cada vez que quieras re-entrenar el modelo con datos nuevos).

```Bash
python -m src.pipeline
Espera a que la terminal muestre que todas las tareas se han completado. Esto creará los archivos necesarios en las carpetas data/processed y models.
```

6. Iniciar la Aplicación Web
Una vez que el pipeline ha terminado, inicia el servidor de Flask:


```Bash
python run_app.py
La terminal te mostrará un mensaje indicando que el servidor está activo y escuchando en una dirección, usualmente http://127.0.0.1:5001.
```

7. Acceder a la Aplicación
Abre tu navegador web y ve a la siguiente dirección:

http://127.0.0.1:5000

¡Listo! Ahora puedes navegar por el dashboard, usar la herramienta de pronóstico y explorar el mapa de calor.