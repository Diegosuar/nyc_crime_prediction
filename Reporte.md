# Reporte: Predicción de Crímenes Violentos en NYC usando Machine Learning

*Estudiantes*
* Diego Suárez (0000277124) - Ingeniería Informática
* John Jairo Rojas (0000282452) - Ingeniería Informática
* Giovanni Moreno (0000247483) - Ingeniería Informática

*Universidad de la Sabana*  
*Machine Learning*  

*Profesor:* Hugo Franco, Ph.D.  
*Fecha:* 20 de octubre de 2025

---

## Introducción

El análisis y la predicción de patrones criminales son fundamentales para la seguridad pública en grandes urbes como Nueva York (NYC). Una tarea clave es diferenciar la gravedad de los incidentes, específicamente clasificando las denuncias entre crímenes violentos (usualmente Felonies) y no violentos (Misdemeanors o Violations). Esta clasificación puede informar sobre la respuesta necesaria y el análisis de riesgos.

Este proyecto aborda la clasificación automática de denuncias criminales en NYC como violentas (`is_violent = 1` si Felony) o no violentas (`is_violent = 0`). El objetivo principal fue desarrollar un modelo de Machine Learning con alta precisión utilizando datos públicos, respondiendo a la pregunta: *"Dadas las características de un incidente reportado en NYC (ubicación, hora, tipo de lugar, tipo de ofensa), ¿podemos predecir con alta precisión si se trata de un crimen violento (Felony)?"*. Se hipotetizó que la combinación de características temporales, geoespaciales (mediante clustering K-Means) y contextuales (tipo de lugar y descripción de la ofensa) permitiría una predicción eficaz. Los hallazgos del Análisis Exploratorio de Datos (EDA) respaldaron la elección de una variable objetivo binaria (`is_violent`) y la relevancia de las características seleccionadas.

---

## Métodos

### Fuente y Preparación de Datos

Se utilizó el dataset "NYPD Complaint Data Historic" de NYC OpenData. Los datos se filtraron para incluir únicamente incidentes desde 2020. Se realizó una limpieza inicial eliminando registros con nulos en columnas clave (`cmplnt_fr_dt`, `latitude`, `longitude`, `prem_typ_desc`, `ofns_desc`, `law_cat_cd`, `boro_nm`). La columna `cmplnt_fr_dt` se convirtió a formato datetime. La ingesta de datos fue gestionada por `data_ingestion.py`.

### Variable Objetivo

Se definió la variable binaria `is_violent` (1 si `law_cat_cd`='FELONY', 0 en caso contrario). Esta binarización resultó en una proporción de clases más manejable.

### Ingeniería de Características

Se generaron las siguientes características:
* **Temporales:** Transformaciones cíclicas (seno/coseno) para hora (`hour_sin`, `hour_cos`) y mes (`month_sin`, `month_cos`), y una bandera `is_weekend`. El EDA validó su relevancia.
* **Geoespaciales (Clustering):** Se aplicó K-Means (MacQueen, 1967) sobre `latitude` y `longitude` para crear 50 clusters (`crime_cluster`). El EDA mostró concentraciones geográficas, justificando esta técnica. Las coordenadas originales se mantuvieron.

    ![Mapa de Calor de Densidad de Crímenes en NYC](heatmap.jpg)
    *(Figura 1: Mapa de calor mostrando la concentración de denuncias)*

* **Categóricas:** `prem_typ_desc`, `crime_cluster` y `ofns_desc` se codificaron usando One-Hot (`pd.get_dummies`) con `drop_first=True`. Solo se consideraron `ofns_desc` con más de 10 ocurrencias. El EDA confirmó la variación de violencia según estas variables.

### Manejo del Desbalance y División de Datos

Se aplicó SMOTE (Chawla et al., 2002) al conjunto de entrenamiento para balancearlo. Los datos se dividieron en 80% entrenamiento y 20% prueba con estratificación.

### Escalado

Las características numéricas se escalaron con `StandardScaler`, ajustado solo en el entrenamiento. El scaler y un `LabelEncoder` (`['No Violento', 'Violento']`) se guardaron.

### Modelo y Optimización

Se utilizó `XGBClassifier` (Chen & Guestrin, 2016). Se optimizaron los hiperparámetros mediante `RandomizedSearchCV` (Pedregosa et al., 2011) con 20 iteraciones y validación cruzada de 3 folds, optimizando para *accuracy*. El mejor modelo se guardó.

### Evaluación

El modelo final se evaluó en el conjunto de prueba usando: Accuracy, AUC-ROC, Precision, Recall, F1-Score (por clase y ponderado), matriz de confusión e importancia de características. El EDA justificó el uso de AUC-ROC y F1-Score además de Accuracy.

### Orquestación y Herramientas

El pipeline completo fue orquestado con Prefect. Se usaron Pandas (McKinney, 2010), Scikit-learn (Pedregosa et al., 2011), Imbalanced-learn (Lemaître et al., 2017), XGBoost (Chen & Guestrin, 2016), Matplotlib y Seaborn. La aplicación web se construyó con Flask.

---

## Resultados

### Rendimiento del Modelo

El modelo XGBoost optimizado mostró un alto rendimiento en el conjunto de prueba, como se resume en el dashboard generado:

* **Accuracy General:** 94.50%
* **AUC-ROC:** 0.9873
* **F1-Score (Violento):** 0.9134
* **F1-Score Ponderado:** 0.9444

La matriz de confusión visualizada en el dashboard mostró una clara separación entre clases, con pocos errores.

![Dashboard de Analítica de Crímenes con Métricas y Gráficos](dashboard_metrics.jpg)
*(Figura 2: Dashboard principal mostrando métricas clave, matriz de confusión, importancia de características y análisis descriptivos)*

### Importancia de Características

El análisis de importancia de características (visible en la Figura 2) reveló una dependencia predominante del modelo en las variables derivadas de `ofns_desc`.

### Análisis Descriptivo

El dashboard (Figura 2) también presenta análisis descriptivos:
* **Crímenes Comunes:** "PETIT LARCENY", "HARRASSMENT 2", y "ASSAULT 3 & RELATED OFFENSES" son los más frecuentes.
* **Patrones Temporales:** Los incidentes varían según el día de la semana.
* **Lugares Comunes:** "Street", "RESIDENCE - APT. HOUSE", son los lugares con más denuncias.

### Herramienta de Predicción

La aplicación web permite estimar la probabilidad de violencia. Los ejemplos muestran la fuerte influencia de `ofns_desc`:
* Para "ASSAULT 3 & RELATED OFFENSES" (No Violento), la probabilidad predicha es baja (1.08%).
* Para "ROBBERY" (Violento), la probabilidad predicha es muy alta (95.75%).

---

## Discusión

El modelo XGBoost alcanzó alta precisión métrica (94.50% Accuracy, 0.9444 F1 Ponderado). La metodología (K-Means, SMOTE, XGBoost) fue efectiva métricamente y justificada por el EDA.

Sin embargo, la **fuga de información (target leakage)** es una limitación crucial. La alta dependencia del modelo en `ofns_desc` (Figura 2) indica que aprende la correlación directa entre descripción y categoría legal, más que predecir contextualmente [cite: , ]. Esto se confirma en la herramienta de predicción, donde el tipo de ofensa domina el resultado.

![Ejemplo de Predicción - Baja Probabilidad (Assault 3)](prediction_low_prob.jpg)
*(Figura 3: Predicción para un incidente tipo "Assault 3", resultando en baja probabilidad de ser violento)*

![Ejemplo de Predicción - Alta Probabilidad (Robbery)](prediction_high_prob.jpg)
*(Figura 4: Predicción para un incidente tipo "Robbery", resultando en muy alta probabilidad de ser violento)*

**Otras Limitaciones:**
* **Interpretabilidad:** K-Means y XGBoost son complejos.
* **Sesgos:** Los datos pueden tener sesgos inherentes.
* **Definición Simplificada:** 'Violento' = 'Felony'.

**Implicaciones y Futuro Trabajo:**
El modelo es útil para clasificación *post-hoc*. La aplicación Flask demuestra este uso. Para una predicción *contextual* (sin `ofns_desc`):
1.  Re-entrenar el modelo excluyendo `ofns_desc` y evaluar el rendimiento realista.
2.  Incorporar datos contextuales adicionales (demográficos, etc.).
3.  Explorar alternativas geoespaciales.


---

## Bibliografía (Formato APA 7ma Edición)

1.  Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, *16*, 321–357. https://doi.org/10.1613/jair.953
2.  Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning. *Journal of Machine Learning Research*, *18*(17), 1–5. http://jmlr.org/papers/v18/16-365.html
3.  McKinney, W. (2010). Data structures for statistical computing in Python. In S. van der Walt & J. Millman (Eds.), *Proceedings of the 9th Python in Science Conference* (pp. 56–61). https://doi.org/10.25080/Majora-92bf1922-00a
4.  NYC OpenData. (n.d.). *NYPD Complaint Data Historic*. Retrieved October 20, 2025, from https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i
5.  Prefect Technologies, Inc. (n.d.). *Prefect Documentation*. Retrieved October 20, 2025, from https://docs.prefect.io/ (Nota: Si se encuentra una publicación formal, reemplazar.)