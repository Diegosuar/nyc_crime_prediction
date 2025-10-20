# NYC Crime Analytics - Proyecto de Machine Learning

Este proyecto implementa un pipeline completo de ciencia de datos para analizar y predecir la probabilidad de robos en la ciudad de Nueva York. Utiliza un modelo de Machine Learning entrenado con datos históricos de denuncias y presenta los resultados a través de una aplicación web interactiva construida con Flask.

El flujo de trabajo está orquestado con **Prefect** para garantizar la reproducibilidad y la automatización del preprocesamiento de datos y el entrenamiento del modelo.

---
## 🚀 Características Principales

* **Dashboard de Analítica**: Una página principal con métricas clave del modelo y gráficos visuales sobre la distribución de crímenes, los días más peligrosos y los tipos de lugares con mayor incidencia.
* **Pronóstico Interactivo de Robos**: Una herramienta que permite al usuario configurar condiciones (distrito, día, hora, tipo de lugar) para obtener una predicción en tiempo real de la probabilidad de que un incidente sea un robo.
* **Mapa de Calor de Densidad**: Una visualización geoespacial que muestra las "zonas calientes" de denuncias de crímenes en toda la ciudad, superpuesta con los límites de los distritos (Boroughs) para un mejor contexto.
* **Pipeline Automatizado con Prefect**: Todo el proceso de ETL (Extracción, Transformación y Carga) y entrenamiento del modelo está encapsulado en un flujo de Prefect, lo que facilita su ejecución y mantenimiento.

---
## 📁 Estructura del Proyecto

El proyecto está organizado de la siguiente manera para separar la lógica del pipeline, la aplicación web y los datos:

```bash
/proyecto_crimen_nyc
|
|-- app/              # Contiene la aplicación web Flask
|   |-- static/       # Archivos estáticos (imágenes, GeoJSON)
|   |-- templates/    # Plantillas HTML de la aplicación
|   |-- app.py        # Lógica principal del servidor web
|
|-- data/             # Almacena los datasets
|   |-- raw/          # Datos crudos descargados por el pipeline
|   |-- processed/    # Datos limpios y listos para el modelo
|
|-- models/           # Modelos de ML y preprocesadores entrenados
|
|-- src/              # Contiene el pipeline de datos con Prefect
|   |-- config.py
|   |-- data_ingestion.py
|   |-- preprocessing.py
|   |-- train.py
|   |-- evaluate.py
|   |-- pipeline.py
|
|-- requirements.txt  # Dependencias del proyecto
|-- run_app.py        # Script para iniciar la aplicación web
|-- README.md         # Este archivo
```

## 🛠️ Tecnologías Utilizadas

* Backend: Python, Flask
* Orquestación de Datos: Prefect
* Machine Learning: Scikit-learn
* Manipulación de Datos: Pandas
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

http://127.0.0.1:5001

¡Listo! Ahora puedes navegar por el dashboard, usar la herramienta de pronóstico y explorar el mapa de calor.