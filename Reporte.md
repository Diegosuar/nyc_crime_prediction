# Reporte: Predicción de Crímenes Violentos en NYC usando Machine Learning

*Estudiantes*
* Diego Suárez (0000277124) - Ingeniería Informática
* John Jairo Rojas (0000282452) - Ingeniería Informática
* Giovanni Moreno (0000247483) - Ingeniería Informática

*Universidad de la Sabana*
*Machine Learning*

*Profesor:* Hugo Franco, Ph.D.
*Fecha:* 20 de octubre de 2025

---

## 1. Introducción

El análisis y la predicción de patrones criminales son actividades cruciales para mejorar la seguridad pública y optimizar la asignación de recursos policiales en entornos urbanos complejos como la ciudad de Nueva York (NYC). Una tarea de particular interés es la clasificación de la gravedad de los incidentes reportados, específicamente la distinción entre crímenes violentos, a menudo catalogados como delitos graves ('Felonies'), y aquellos considerados no violentos ('Misdemeanors' o 'Violations'). Esta diferenciación no solo informa sobre la respuesta inmediata requerida sino que también facilita un análisis de riesgos más granular a nivel temporal y espacial.

Este proyecto se enfoca en el desarrollo y evaluación de un modelo de Machine Learning para la clasificación automática de denuncias criminales en NYC como violentas (`is_violent = 1` si Felony) o no violentas (`is_violent = 0`). El objetivo central es responder a la pregunta de investigación: *"Dadas las características espaciales, temporales y contextuales de un incidente reportado en NYC (incluyendo ubicación, hora, tipo de lugar y tipo específico de ofensa), ¿es posible predecir con precisión si corresponde a un crimen violento (Felony)?"*. La hipótesis subyacente es que una combinación sinérgica de estas características, procesadas mediante técnicas de ingeniería avanzadas como el clustering geoespacial (K-Means) y la codificación cíclica temporal, permitiría una predicción eficaz.

Los hallazgos preliminares del Análisis Exploratorio de Datos (EDA) jugaron un papel fundamental en la definición del problema. Se observó que intentar una clasificación multiclase sobre los 180 tipos distintos de `ofns_desc` resultaba inviable debido al severo desbalance y la granularidad. En contraste, la binarización basada en `law_cat_cd` ('Felony' vs. 'No Felony') ofrecía una proporción de clases más manejable (aproximadamente 33% vs. 67%), justificando la elección de `is_violent` como variable objetivo. Además, el EDA confirmó la existencia de patrones temporales (variaciones horarias y de fin de semana) y geoespaciales (concentraciones en "zonas calientes"), así como diferencias significativas en la tasa de violencia según el tipo de lugar (`prem_typ_desc`) y la descripción de la ofensa (`ofns_desc`), validando así la relevancia de las características seleccionadas para el modelo predictivo. La implementación exitosa de este modelo podría ofrecer una herramienta analítica valiosa para entender los factores asociados a la violencia criminal y apoyar la toma de decisiones estratégicas y operativas.

---

## 2. Métodos

### 2.1 Fuente y Preparación de Datos

El estudio se basó en el conjunto de datos público "NYPD Complaint Data Historic", accesible a través del portal NYC OpenData. La ingesta de datos, gestionada por el script `data_ingestion.py`, incluyó la descarga de datos de denuncias, filtrándolos para incidentes ocurridos a partir del 1 de enero de 2020 para asegurar la relevancia temporal y manejar el volumen de datos. Se estandarizaron los nombres de las columnas a minúsculas. Un paso crucial de limpieza implicó la eliminación de registros con valores nulos en columnas esenciales para el análisis y modelado: `cmplnt_fr_dt`, `latitude`, `longitude`, `prem_typ_desc`, `ofns_desc`, `law_cat_cd`, y `boro_nm`. La columna `cmplnt_fr_dt` se convirtió explícitamente a formato datetime para facilitar la extracción de características temporales. Aunque se consideró la integración con datasets de arrestos y paradas de vehículos durante la fase de ingesta, estos no fueron utilizados en el modelo final debido a desafíos en la unión de datos y errores de disponibilidad (e.g., error 404 para `vehicle_stops`).

### 2.2 Variable Objetivo

La variable objetivo binaria, `is_violent`, se derivó directamente de la columna `law_cat_cd`. Se asignó el valor `1` si `law_cat_cd` era 'FELONY' y `0` si era 'MISDEMEANOR' o 'VIOLATION'. Esta definición operativa simplifica el problema a una clasificación binaria, que, como se mencionó, presentaba un desbalance de clases más manejable que alternativas multiclase.

### 2.3 Ingeniería de Características

El proceso de ingeniería de características, implementado en `preprocessing.py`, se centró en extraer información relevante del tiempo, espacio y contexto del incidente:

* **Características Temporales:** Para capturar patrones cíclicos diarios y anuales sin introducir discontinuidades artificiales, se aplicaron transformaciones seno y coseno a la hora del día (extraída de `cmplnt_fr_dt`) y al mes del año, generando `hour_sin`, `hour_cos`, `month_sin`, `month_cos`. Adicionalmente, se creó una variable indicadora binaria `is_weekend` (1 para sábado/domingo, 0 en caso contrario). El EDA previo había mostrado variaciones significativas en la ocurrencia y tipo de crímenes según estos factores temporales.
* **Características Geoespaciales (Clustering):** Reconociendo la naturaleza espacialmente autocorrelacionada del crimen, se aplicó el algoritmo K-Means (MacQueen, 1967) a las coordenadas `latitude` y `longitude` para agrupar los incidentes en $k=50$ clusters geoespaciales. La asignación de cada incidente a un cluster se almacenó en la característica categórica `crime_cluster`. El número de clusters (50) se eligió como un balance entre granularidad y interpretabilidad. Las coordenadas originales (`latitude`, `longitude`) se conservaron como características numéricas. El EDA visual mediante mapas de calor (Figura 1) había confirmado la existencia de "zonas calientes", justificando el uso de clustering para capturar esta estructura espacial.

    ![Mapa de Calor de Densidad de Crímenes en NYC](heatmap.jpg)
    *(Figura 1: Mapa de calor mostrando la concentración geográfica de denuncias en NYC)*

* **Características Categóricas:** Las columnas `prem_typ_desc` (tipo de lugar), `crime_cluster` (cluster geoespacial) y `ofns_desc` (descripción específica de la ofensa) se trataron como categóricas. Para manejar la alta cardinalidad de `ofns_desc`, solo se consideraron las categorías con más de 10 ocurrencias en el dataset filtrado; las instancias de categorías menos frecuentes fueron efectivamente excluidas del modelado en este paso. Posteriormente, se aplicó codificación One-Hot (`pd.get_dummies`) a las tres variables, utilizando el prefijo `cat_` y eliminando la primera categoría de cada una (`drop_first=True`) para prevenir multicolinealidad perfecta. El EDA había resaltado la fuerte relación entre `prem_typ_desc`, `ofns_desc` y la probabilidad de violencia, validando su inclusión como predictores clave.

### 2.4 Manejo del Desbalance de Clases

La proporción de la clase minoritaria (`is_violent = 1`) fue aproximadamente del 31-33% después de la limpieza inicial. Para abordar este desbalance y prevenir que el modelo se sesgara hacia la clase mayoritaria, se aplicó la técnica de sobremuestreo SMOTE (Synthetic Minority Over-sampling Technique) (Chawla et al., 2002). Es crucial destacar que SMOTE se aplicó **únicamente** al conjunto de datos de entrenamiento después de la división, para evitar la fuga de datos hacia el conjunto de prueba.

### 2.5 División de Datos y Escalado

El conjunto de datos procesado (post-codificación one-hot) se dividió en un conjunto de entrenamiento (80%) y un conjunto de prueba (20%) utilizando `train_test_split` de Scikit-learn (Pedregosa et al., 2011). Se utilizó estratificación (`stratify=y`) para asegurar que la proporción original de clases `is_violent` se mantuviera en ambas divisiones. Las características numéricas (`latitude`, `longitude`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos`) se escalaron utilizando `StandardScaler`. El escalador se ajustó (`fit`) exclusivamente con los datos de entrenamiento (post-SMOTE) y luego se aplicó (`transform`) tanto al conjunto de entrenamiento como al de prueba. El `StandardScaler` ajustado y un `LabelEncoder` para las etiquetas del target (`['No Violento', 'Violento']`) fueron serializados y guardados usando `joblib.dump`.

### 2.6 Modelo de Clasificación y Optimización

Se seleccionó `XGBClassifier` (Chen & Guestrin, 2016), una implementación optimizada del algoritmo de gradient boosting, debido a su robustez y rendimiento demostrado en competiciones y problemas de clasificación tabular. Para encontrar una configuración de hiperparámetros cercana a la óptima, se utilizó `RandomizedSearchCV` (Pedregosa et al., 2011), explorando combinaciones aleatorias dentro de un espacio de búsqueda definido para `n_estimators`, `learning_rate`, `max_depth`, `subsample` y `colsample_bytree`. Se ejecutaron 20 iteraciones (`n_iter=20`) con validación cruzada interna de 3 folds (`cv=3`), utilizando *accuracy* como métrica de puntuación para la selección del mejor modelo. El `RandomizedSearchCV` se ajustó sobre el conjunto de entrenamiento (ya balanceado con SMOTE y escalado). El mejor estimador encontrado fue guardado como `models/crime_predictor_model.joblib`.

### 2.7 Evaluación del Modelo

El rendimiento del mejor modelo XGBoost se evaluó rigurosamente sobre el conjunto de prueba, que no fue utilizado durante el entrenamiento ni la optimización. Se calcularon las siguientes métricas:
* **Accuracy:** Proporción general de predicciones correctas.
* **AUC-ROC:** Área bajo la curva ROC, que mide la capacidad del modelo para discriminar entre las clases positiva y negativa.
* **Informe de Clasificación:** Incluye Precision, Recall y F1-Score para cada clase ('Violento', 'No Violento'), así como promedios macro y ponderados (`weighted avg`). El F1-Score para la clase 'Violento' y el F1-Score ponderado fueron considerados métricas clave dada la relevancia de identificar correctamente los crímenes violentos y el desbalance original.
* **Matriz de Confusión:** Visualización de verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos.
* **Importancia de Características:** Se extrajeron las puntuaciones de importancia (`feature_importances_`) del modelo XGBoost entrenado para identificar los predictores más influyentes.

Los resultados de la evaluación, incluyendo métricas numéricas, análisis descriptivos (conteo de crímenes por lugar y distrito) y rutas a los gráficos generados (matriz de confusión, importancia de características), se compilaron y guardaron en un archivo JSON (`reports/dashboard_metrics.json`).

### 2.8 Orquestación y Herramientas

El flujo completo de trabajo, desde la descarga de datos hasta la evaluación final y el guardado de artefactos, fue encapsulado y orquestado como un pipeline utilizando Prefect, lo que facilita la ejecución automatizada y la reproducibilidad. Las herramientas de software primarias incluyeron Python 3.9+, Pandas (McKinney, 2010), Scikit-learn (Pedregosa et al., 2011), Imbalanced-learn (Lemaître et al., 2017), XGBoost (Chen & Guestrin, 2016), Joblib, Matplotlib y Seaborn. Una aplicación web interactiva para visualizar resultados y realizar predicciones se desarrolló utilizando Flask.

---

## 3. Resultados

### 3.1 Rendimiento Cuantitativo del Modelo

El modelo XGBoost final, tras la optimización de hiperparámetros, exhibió un rendimiento notablemente alto en la clasificación de incidentes violentos versus no violentos sobre el conjunto de prueba (20% de los datos originales, no expuesto durante el entrenamiento). Las métricas clave, extraídas del archivo `dashboard_metrics.json` y visualizadas en el dashboard (Figura 2), fueron las siguientes:

* **Accuracy General:** 94.50%. Indica que el 94.5% de todas las predicciones en el conjunto de prueba fueron correctas.
* **AUC-ROC:** 0.9873. Un valor muy cercano a 1.0, lo que sugiere una excelente capacidad del modelo para distinguir entre las clases 'Violento' y 'No Violento'.
* **F1-Score (Clase 'Violento'):** 0.9134. Representa la media armónica entre la precisión y el recall específicamente para la clase minoritaria 'Violento', indicando un buen equilibrio en la identificación de estos casos.
* **F1-Score Ponderado:** 0.9444. Es el promedio de los F1-Scores de ambas clases, ponderado por el número de instancias verdaderas de cada clase, reflejando el rendimiento general balanceado del modelo en el conjunto de prueba desbalanceado.

La matriz de confusión, también mostrada en el dashboard (Figura 2), corroboró estos resultados, visualizando un número bajo de falsos positivos y falsos negativos en relación con los verdaderos positivos y negativos. La diagonal principal de la matriz contenía la gran mayoría de las predicciones, indicando una alta tasa de aciertos para ambas clases.

![Dashboard de Analítica de Crímenes con Métricas y Gráficos](dashboard_metrics.jpg)
*(Figura 2: Dashboard principal mostrando métricas clave (Accuracy, AUC-ROC, F1), matriz de confusión, importancia de características y análisis descriptivos)*

### 3.2 Importancia de Características

El análisis de la importancia de características (`feature_importances_`) proporcionado por el modelo XGBoost (Figura 2) identificó de manera concluyente que las características más influyentes fueron aquellas derivadas de la codificación one-hot de la variable `ofns_desc` (descripción específica de la ofensa). Variables como `cat_HARASSMENT 2`, `cat_PETIT LARCENY`, `cat_ASSAULT 3 & RELATED OFFENSES`, `cat_GRAND LARCENY`, etc., dominaron los primeros puestos del ranking de importancia. Las características temporales (cíclicas de hora/mes, fin de semana) y geoespaciales (`latitude`, `longitude`, `cat_crime_cluster_*`) mostraron una importancia relativa significativamente menor.

### 3.3 Análisis Descriptivo y Visualizaciones

El pipeline también generó análisis descriptivos y visualizaciones que se presentan en el dashboard:
* **Crímenes Más Comunes:** El análisis de frecuencia de `ofns_desc` en los datos crudos (Figura 2) confirmó que "PETIT LARCENY", "HARRASSMENT 2", y "ASSAULT 3 & RELATED OFFENSES" son consistentemente los tipos de incidentes más reportados.
* **Patrones Temporales:** El gráfico de incidentes por día de la semana (Figura 2) mostró variaciones en la actividad criminal a lo largo de la semana.
* **Análisis de Lugares:** El análisis de `prem_typ_desc` (Figura 2) identificó "Street" (Calle), "RESIDENCE - APT. HOUSE" (Residencia - Edificio Aptos.), y "RESIDENCE - HOUSE" (Residencia - Casa) como los lugares con mayor volumen de denuncias.
* **Mapa de Densidad:** El mapa de calor (Figura 1) generado usando Folium visualizó eficazmente las áreas geográficas con mayor concentración de incidentes reportados.

### 3.4 Herramienta de Predicción Interactiva

La aplicación web incluye una herramienta interactiva (`/forecast`) que permite a los usuarios ingresar características de un incidente hipotético y obtener una predicción de la probabilidad de que sea violento (Felony). Las pruebas con esta herramienta (Figuras 3 y 4) demostraron la alta sensibilidad del modelo a la característica `ofns_desc`. Al seleccionar un `ofns_desc` típicamente asociado con Felonies (e.g., "ROBBERY"), la probabilidad predicha de violencia fue extremadamente alta (>95%), mientras que al seleccionar uno asociado con Misdemeanors (e.g., "ASSAULT 3 & RELATED OFFENSES"), la probabilidad fue muy baja (<2%).

### 3.5 Salida del Pipeline y Artefactos

La ejecución del pipeline orquestado con Prefect (`pipeline.py`) finalizó exitosamente, generando todos los artefactos necesarios para la evaluación y el despliegue de la aplicación:
* El modelo XGBoost entrenado (`crime_predictor_model.joblib`).
* El escalador ajustado (`scaler.joblib`).
* El codificador de etiquetas (`label_encoder.joblib`).
* El archivo JSON con métricas detalladas y análisis (`reports/dashboard_metrics.json`).
* Los gráficos estáticos para el dashboard (matriz de confusión, importancia de características, distribución de crímenes, crímenes por día) guardados en `app/static/img/`.
* La lista de `ofns_desc` comunes (`models/common_crimes.json`) para el dropdown de la app.

---

## 4. Discusión

Los resultados cuantitativos demuestran que el modelo XGBoost desarrollado es capaz de clasificar las denuncias criminales en NYC como violentas (Felony) o no violentas con una precisión general superior al 94% y un F1-Score ponderado similarmente alto. La metodología empleada, que incluyó ingeniería de características espaciales (K-Means) y temporales (cíclicas), junto con el manejo del desbalance de clases (SMOTE) y la optimización de hiperparámetros (RandomizedSearchCV), contribuyó a alcanzar este nivel de rendimiento métrico. Las decisiones metodológicas fueron respaldadas por los hallazgos del EDA, que confirmaron la pertinencia de la variable objetivo binaria y la relevancia predictiva de las características seleccionadas.

Sin embargo, una interpretación crítica de estos resultados es imperativa, particularmente a la luz del análisis de importancia de características. La **dependencia abrumadora del modelo en las características derivadas de `ofns_desc`** (Figura 2) constituye una clara indicación de **fuga de información (target leakage)**. Dado que la variable objetivo `is_violent` se define directamente a partir de `law_cat_cd`, y `law_cat_cd` está intrínsecamente ligada a `ofns_desc`, el modelo no está aprendiendo a predecir la violencia basándose en el contexto situacional (dónde y cuándo ocurre), sino que está primordialmente memorizando la relación casi determinista entre la descripción específica de un crimen y su clasificación legal. La herramienta de predicción interactiva (Figuras 3 y 4) ilustra este fenómeno de manera patente: cambiar únicamente el `ofns_desc` de "ASSAULT 3..." a "ROBBERY" cambia la predicción de ~1% a >95% de probabilidad de violencia, manteniendo constantes todos los demás factores contextuales.

![Ejemplo de Predicción - Baja Probabilidad (Assault 3)](prediction_low_prob.jpg)
*(Figura 3: Predicción para un incidente tipo "Assault 3", resultando en baja probabilidad de ser violento)*

![Ejemplo de Predicción - Alta Probabilidad (Robbery)](prediction_high_prob.jpg)
*(Figura 4: Predicción para un incidente tipo "Robbery", resultando en muy alta probabilidad de ser violento)*

**Limitaciones:**
* **Target Leakage:** Es la limitación más significativa. Restringe la utilidad del modelo como predictor *contextual* de violencia antes de conocer la naturaleza exacta del crimen. Funciona más como un clasificador *post-hoc*.
* **Interpretabilidad:** Aunque la importancia de características es informativa, la naturaleza de "caja negra" de los clusters K-Means y del ensamblaje de árboles de XGBoost dificulta una explicación causal detallada.
* **Sesgos Potenciales en los Datos:** Los datos de denuncias reflejan las interacciones reportadas con la policía y pueden no representar la totalidad de la actividad criminal o pueden contener sesgos sistémicos (ej. sobrerrepresentación o subrepresentación de ciertos tipos de crímenes o en ciertas áreas/comunidades). El modelo podría perpetuar estos sesgos.
* **Definición Simplificada de Violencia:** La categorización binaria basada exclusivamente en 'Felony' es una simplificación. La percepción de violencia puede variar, y esta definición no captura matices.

En conclusión, este proyecto ha resultado en un pipeline de ML funcional y una aplicación interactiva, alcanzando una precisión métrica en la tarea definida. Sin embargo, el reconocimiento y la discusión franca de la fuga de información son esenciales para comprender las verdaderas capacidades predictivas del modelo y para guiar futuras investigaciones hacia predicciones contextuales más robustas y significativas.

---

## 5. Bibliografía

1.  Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://doi.org/10.1145/2939672.2939785
2.  Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, *16*, 321–357. https://doi.org/10.1613/jair.953
3.  Lemaître, G., Nogueira, F., & Aridas, C. K. (2017). Imbalanced-learn: A Python toolbox to tackle the curse of imbalanced datasets in machine learning. *Journal of Machine Learning Research*, *18*(17), 1–5. http://jmlr.org/papers/v18/16-365.html
4.  NYC OpenData. (n.d.). *NYPD Complaint Data Historic*. Retrieved October 20, 2025, from https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i
5.  Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, *12*, 2825–2830. http://jmlr.org/papers/v12/pedregosa11a.html
6.  Prefect Technologies, Inc. (n.d.). *Prefect Documentation*. Retrieved October 20, 2025, from https://docs.prefect.io/ (Nota: Si se encuentra una publicación formal, reemplazar.)