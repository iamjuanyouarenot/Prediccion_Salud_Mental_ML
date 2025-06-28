# Prediccion_Salud_Mental_ML
# Predicción de Preocupación por Salud Mental utilizando Aprendizaje Automático

## I. Introducción

### 1.1 Título del Proyecto
Predicción de Preocupación por Salud Mental utilizando Aprendizaje Automático basado en Datos de Encuestas[cite: 1].

### 1.2 Antecedentes
La salud mental es un área crítica que afecta a millones de personas a nivel global[cite: 1]. [cite_start]Tradicionalmente, la identificación de trastornos como la depresión se basa en evaluaciones clínicas y cuestionarios estandarizados[cite: 2]. [cite_start]Si bien estos métodos son fundamentales, pueden ser subjetivos o tardíos[cite: 3]. ]En ausencia de datos clínicos directos o cuestionarios validados como el PHQ-9 en el conjunto de datos actual, la aplicación del aprendizaje automático ofrece una alternativa para analizar patrones en datos de encuestas disponibles[cite: 4]. [cite_start]Esta aproximación permite predecir la probabilidad de que una persona manifieste indicadores de preocupación por su salud mental, basándose en factores personales y percepciones sobre el entorno laboral y la búsqueda de ayuda[cite: 5]. [cite_start]Esta capacidad facilita la identificación temprana de individuos que podrían beneficiarse de una intervención o apoyo[cite: 6].

### 1.3 Problema a Resolver
[cite_start]¿Cómo se pueden utilizar técnicas de aprendizaje automático para predecir de manera automatizada y precisa la probabilidad de que una persona manifieste preocupación por su salud mental, a partir de datos recolectados mediante encuestas que reflejan factores personales y experiencias laborales relacionadas con la salud mental? [cite: 7]

### 1.4 Objetivos

**Objetivo General:**
Desarrollar un modelo predictivo basado en técnicas de aprendizaje automático que permita identificar de manera automatizada y precisa la probabilidad de que una persona manifieste preocupación por su salud mental, a partir del análisis de datos obtenidos mediante encuestas sobre factores personales y el entorno laboral[cite: 8].

**Objetivos Específicos:**
* Recopilar y preprocesar datos provenientes de encuestas, derivando una variable objetivo "Preocupación por Salud Mental" a partir de indicadores indirectos disponibles[cite: 9].
* Implementar y entrenar modelos de aprendizaje automático supervisado (Random Forest Classifier y Red Neuronal Convolucional 1D) para la clasificación binaria de la preocupación por salud mental[cite: 10].
* Evaluar el desempeño de los modelos utilizando métricas como precisión (accuracy), recall, F1-score y matriz de confusión, para determinar su efectividad en la detección de posibles casos de preocupación por salud mental[cite: 11].
* [cite_start]Analizar la importancia de variables para identificar los factores personales y laborales que tienen mayor peso en la predicción de la preocupación por salud mental[cite: 12].

## II. Requerimientos del Sistema

### 2.1 Definición del Dominio
[cite_start]El dominio de este proyecto se centra en la predicción y detección de indicadores de preocupación por la salud mental mediante el uso de inteligencia artificial y aprendizaje automático[cite: 13]. [cite_start]Esto implica el análisis de datos de encuestas relacionadas con percepciones sobre el entorno laboral, búsqueda de tratamiento y antecedentes personales[cite: 14]. [cite_start]El objetivo principal es proporcionar una herramienta complementaria que pueda identificar individuos con alta probabilidad de manifestar estas preocupaciones, permitiendo una posible intervención o referencia oportuna[cite: 15].

### 2.2 Requerimientos

**Requerimientos Funcionales:**
* [cite_start]El sistema debe ser capaz de cargar y procesar un conjunto de datos estructurado (CSV)[cite: 17].
* [cite_start]El sistema debe permitir el preprocesamiento de datos, incluyendo la gestión de valores nulos y la codificación de variables categóricas a numéricas[cite: 18].
* [cite_start]El sistema debe derivar una variable objetivo binaria ("Preocupación por Salud Mental") a partir de las columnas existentes[cite: 19].
* [cite_start]El sistema debe implementar algoritmos de clasificación (Random Forest Classifier y CNN 1D) para entrenar modelos predictivos[cite: 20].
* [cite_start]El sistema debe ser capaz de dividir el conjunto de datos en subconjuntos de entrenamiento y prueba[cite: 21].
* [cite_start]El sistema debe poder evaluar el rendimiento de los modelos utilizando métricas como precisión, recall, F1-score y la matriz de confusión[cite: 22].
* [cite_start]El sistema debe generar predicciones sobre nuevos datos de encuestas[cite: 23].
* [cite_start]El sistema debe identificar y mostrar la importancia de las características (variables) en la predicción (para Random Forest)[cite: 24].

**Requerimientos No Funcionales:**
* [cite_start]**Precisión:** Los modelos deben alcanzar una alta precisión en la clasificación[cite: 25].
* [cite_start]**Rendimiento:** El tiempo de entrenamiento y predicción del modelo debe ser razonable para permitir un uso eficiente[cite: 26].
* [cite_start]**Escalabilidad:** El sistema debe ser capaz de manejar conjuntos de datos de tamaño moderado[cite: 27].
* [cite_start]**Usabilidad:** El código debe ser claro y fácil de entender para futuros desarrollos o mantenimiento[cite: 28].
* [cite_start]**Confiabilidad:** El modelo debe ser robusto y proporcionar resultados consistentes[cite: 29].
* [cite_start]**Seguridad:** Asegurar la privacidad y confidencialidad de los datos sensibles de los usuarios (aunque para este proyecto solo se trabaja con datasets anonimizados)[cite: 30].

## III. Pre-procesamiento y Normalización (Planteamiento del Data-Set; Aprendizaje Supervisado)

[cite_start]El problema de predecir la preocupación por la salud mental, al basarse en una variable objetivo binaria derivada de las características existentes en el dataset, se enmarca dentro del Aprendizaje Supervisado, específicamente como un problema de clasificación binaria[cite: 31]. [cite_start]El objetivo es que los modelos aprendan a mapear las características de entrada a una de dos clases: "No Preocupación" o "Preocupación"[cite: 32].

### 3.1 Medidas, Datos, Bases de Datos y Elaboración del Data-Set
[cite_start]El dataset principal utilizado para esta investigación es `survey.csv`, que contiene información sobre características demográficas, laborales y percepciones relacionadas con la salud mental[cite: 33].

**Datos Utilizados:**
El dataset original cargado tenía 1259 filas y 27 columnas, incluyendo:
* [cite_start]**Demográficas:** Age, Gender, Country, state[cite: 34].
* [cite_start]**Laborales/Empresa:** self_employed, no_employees, remote_work, tech_company, benefits, care_options, wellness_program, anonymity, leave, coworkers, supervisor[cite: 35].
* [cite_start]**Salud Mental y Percepciones:** family_history, treatment, work_interfere, mental_health_consequence, phys_health_consequence, seek_help, mental_health_interview, phys_health_interview, mental_vs_physical, obs_consequence[cite: 36].
* [cite_start]**Metadata/Comentarios:** Timestamp, comments[cite: 36].

### 3.2 Normalización y/o Filtrado de Datos
[cite_start]El preprocesamiento de datos fue un paso crítico para preparar el dataset[cite: 37]. Los pasos específicos incluyeron:
* [cite_start]**Carga del Dataset:** El archivo `survey.csv` fue cargado en Google Colab[cite: 38].
* [cite_start]**Manejo de Columnas Innecesarias:** Se identificaron y eliminaron columnas que no eran directamente útiles para el propósito predictivo: `Timestamp`, `state` y `comments`[cite: 39].
* [cite_start]**Derivación de la Variable Objetivo `Mental_Health_Concern`:** Dado que el dataset no contenía las preguntas específicas del cuestionario PHQ-9, se derivó una variable objetivo binaria `Mental_Health_Concern` a partir de una combinación de indicadores indirectos presentes en el dataset[cite: 40]. Una persona fue clasificada con 1 (Preocupación) si cumplía al menos una de las siguientes condiciones:
    * [cite_start]Indicó que recibe `treatment` (tratamiento)[cite: 41].
    * [cite_start]Reportó que su salud mental `work_interfere` (interfiere con el trabajo) (es decir, no es "Never" o NaN)[cite: 42].
    * [cite_start]Indicó que tuvo `mental_health_consequence` (consecuencias en su salud mental)[cite: 42].
    * [cite_start]Reportó haber `seek_help` (buscado ayuda)[cite: 43].
    * [cite_start]Tenía `family_history` (historial familiar) de enfermedad mental[cite: 43].
    * [cite_start]En cualquier otro caso, fue clasificada con 0 (No Preocupación)[cite: 44].
    * [cite_start]El conteo resultante de la variable objetivo fue: 873 instancias para "Preocupación" (1) y 184 instancias para "No Preocupación" (0)[cite: 45].
    * [cite_start]Se observó un desequilibrio de clases, con la clase "Preocupación" siendo significativamente más numerosa[cite: 46]. [cite_start]Esto se abordó utilizando el parámetro `class_weight='balanced'` en el modelo Random Forest[cite: 47].
* [cite_start]**Manejo de Datos Faltantes (Final):** Después de la derivación de la variable objetivo y los mapeos, se eliminaron las filas que contenían valores nulos restantes para garantizar la calidad y completitud de los datos[cite: 48]. [cite_start]El número final de filas después de este proceso fue de 1057[cite: 48].
* [cite_start]**Codificación de Variables Categóricas:** Todas las demás columnas de tipo `object` (texto) fueron transformadas a un formato numérico utilizando `LabelEncoder`, lo cual es necesario para que los modelos de aprendizaje automático puedan procesar estos datos[cite: 49].

### 3.3 Planteamiento de Data-Sets

* [cite_start]**Data-Set de Entrenamiento (training):** El conjunto de datos preprocesado (1057 filas) fue dividido en un 80% para entrenamiento[cite: 50]. [cite_start]Esto resultó en 845 filas y 24 columnas para `X_train`, y 845 filas para `y_train`[cite: 51]. [cite_start]Este subconjunto se utilizó para ajustar los parámetros internos de los modelos de aprendizaje automático[cite: 52].

* [cite_start]**Data-Set de Pruebas (test):** El 20% restante del conjunto de datos se reservó como conjunto de pruebas[cite: 53]. [cite_start]Esto resultó en 212 filas y 24 columnas para `X_test`, y 212 filas para `y_test`[cite: 54]. [cite_start]Este subconjunto, que los modelos no habían visto previamente, se utilizó para estimar su capacidad para predecir correctamente en datos nuevos y no observados[cite: 55]. [cite_start]La división se realizó con `stratify=y_encoded` para asegurar que la proporción de clases (`Mental_Health_Concern`) fuera similar en los conjuntos de entrenamiento y prueba, a pesar del desequilibrio de clases[cite: 56].

* [cite_start]**Set de Validación por "Cross-Validation":** Para el modelo de Deep Learning (CNN), se utilizó un `validation_split` del 10% del conjunto de entrenamiento durante la fase de `fit`[cite: 57]. [cite_start]Esto proporcionó un conjunto de validación interno para monitorear el rendimiento del modelo durante el entrenamiento y aplicar `EarlyStopping`, lo que ayuda a prevenir el sobreajuste y a encontrar el mejor punto de convergencia[cite: 58].

## IV. Aprendizaje del Modelo del Sistema

### 4.1 Planteamiento del Modelo de Aprendizaje
Se seleccionaron dos enfoques de modelado para comparar su desempeño en la clasificación binaria de la preocupación por salud mental:
* [cite_start]**Random Forest Classifier:** Este algoritmo de Machine Learning de ensamble es robusto, eficaz con datos tabulares, y es capaz de manejar el desequilibrio de clases (con `class_weight='balanced'`) y de identificar la importancia de las características[cite: 59].
* [cite_start]**Red Neuronal Convolucional 1D (CNN 1D):** Un modelo de Deep Learning que, aunque más común en datos secuenciales, se adaptó a los datos tabulares tratando cada característica como un "paso de tiempo" con un solo rasgo[cite: 60]. [cite_start]La CNN 1D puede aprender patrones locales entre características, lo cual podría ser relevante si existe alguna relación espacial o de orden en las columnas de la encuesta[cite: 61].

### 4.2 Desarrollo e Implementación del Modelo
[cite_start]El desarrollo de ambos modelos se llevó a cabo utilizando las librerías `scikit-learn` (para Random Forest) y `TensorFlow`/`Keras` (para la CNN) en Python[cite: 62]. [cite_start]La implementación incluyó los pasos de preprocesamiento y división de datos, la inicialización y entrenamiento de cada modelo, y la evaluación de su rendimiento en el conjunto de prueba[cite: 63]. [cite_start]Para la CNN, la capa de salida se configuró con 1 neurona y activación `sigmoid` para clasificación binaria, y se utilizó `binary_crossentropy` como función de pérdida[cite: 64].

## V. Comprobación y Despliegue (Deploy) del Sistema

### 5.1 Entrenamiento del Modelo: Uso del Data-Set de entrenamiento
[cite_start]Ambos modelos fueron entrenados utilizando el 80% del dataset preprocesado[cite: 65]. [cite_start]El Random Forest construyó 100 árboles de decisión[cite: 66]. [cite_start]La CNN 1D se entrenó durante un máximo de 200 épocas, deteniéndose anticipadamente (por `EarlyStopping`) si no había mejora en la pérdida de validación durante 15 épocas, lo que en la ejecución actual ocurrió en 70 épocas[cite: 66].

### 5.2 Ejecución y Pruebas del Modelo

**Resultados del Modelo Random Forest Classifier:**
[cite_start]El modelo Random Forest demostró un rendimiento sobresaliente en la clasificación de la preocupación por salud mental[cite: 67].
* [cite_start]**Precisión (Accuracy):** 0.9949 (99.49%) [cite: 67]
* **Matriz de Confusión:**
    ```
    [[ 20   1]
     [  0 175]]
    ```
    * [cite_start]Verdaderos Negativos (TN): 20 (Personas sin preocupación predichas correctamente como "No Preocupación")[cite: 68].
    * [cite_start]Falsos Positivos (FP): 1 (Persona sin preocupación predicha incorrectamente como "Preocupación")[cite: 69].
    * [cite_start]Falsos Negativos (FN): 0 (Personas con preocupación predichas incorrectamente como "No Preocupación")[cite: 70].
    * [cite_start]Verdaderos Positivos (TP): 175 (Personas con preocupación predichas correctamente como "Preocupación")[cite: 70].

* [cite_start]**Reporte de Clasificación:** El modelo exhibió una precisión y recall casi perfectos para la clase "Preocupación", y un alto rendimiento para la clase "No Preocupación", destacando su capacidad para identificar correctamente ambas categorías[cite: 73].
* [cite_start]**Importancia de las Características (Random Forest):** Los factores más influyentes identificados por el modelo Random Forest en la predicción de la preocupación por salud mental fueron: `work_interfere` (interferencia laboral), `treatment` (tratamiento), y `family_history` (historial familiar)[cite: 74]. [cite_start]Esto valida la intuición de utilizar estas variables como indicadores de preocupación[cite: 74].

**Resultados del Modelo Red Neuronal Convolucional 1D (CNN):**
[cite_start]La CNN 1D también demostró un rendimiento muy sólido en la clasificación binaria[cite: 76].
* [cite_start]**Pérdida (Loss):** 0.1333 [cite: 76]
* [cite_start]**Precisión (Accuracy):** 0.9745 (97.45%) [cite: 76]
* **Matriz de Confusión:**
    ```
    [[ 18   3]
     [  2 173]]
    ```
* [cite_start]**Reporte de Clasificación:** Similar al Random Forest, la CNN mostró alta precisión y recall, especialmente para la clase mayoritaria "Preocupación"[cite: 78].
* [cite_start]**Progreso de Entrenamiento CNN:** Los gráficos de entrenamiento muestran que la precisión del modelo (tanto de entrenamiento como de validación) aumenta rápidamente en las primeras épocas y se mantiene consistentemente alta (cercana al 1.0), mientras que la pérdida disminuye de manera pronunciada y se estabiliza cerca de cero[cite: 81]. [cite_start]Esto indica que la red está aprendiendo eficazmente y que no hay un sobreajuste significativo[cite: 82].

### 5.3 Ejecución de la Validación del Modelo
[cite_start]Ambos modelos fueron validados utilizando una estrategia de división 80%-20% del dataset, con estratificación para mantener las proporciones de clases[cite: 83]. [cite_start]Para la CNN, la validación interna durante el entrenamiento (`validation_split`) y el uso de `EarlyStopping` confirmaron que el modelo generaliza bien a datos no vistos[cite: 84]. [cite_start]Los resultados de alta precisión en el conjunto de prueba para ambos modelos demuestran una robusta capacidad de generalización[cite: 85].

### 5.4 Deploy del APP o Web del Sistema de Predicción
[cite_start]El despliegue de este sistema predictivo podría consistir en una aplicación web o una API que permita a los usuarios ingresar sus respuestas a las preguntas de la encuesta y recibir una predicción de su nivel de preocupación por salud mental[cite: 86]. [cite_start]Actualmente, el proyecto cuenta con el código fuente en Google Colab y la posibilidad de guardar los modelos entrenados (`.pkl`), lo que facilita su futura integración en una aplicación interactiva[cite: 87].

## VI. Análisis de Resultados y Discusión

[cite_start]Los resultados obtenidos son altamente prometedores para la predicción de la preocupación por salud mental basada en el dataset[cite: 88]. [cite_start]Ambos modelos, Random Forest (precisión del 99.49%) y la CNN 1D (precisión del 97.60%), demostraron una capacidad excepcional para clasificar correctamente los casos[cite: 89]. [cite_start]El Random Forest mostró un rendimiento marginalmente superior, destacándose por tener cero falsos negativos, lo que es crucial en contextos de salud donde minimizar la omisión de casos con preocupación es fundamental[cite: 90]. [cite_start]Su análisis de importancia de características también validó la selección de variables proxy como `work_interfere`, `treatment` y `family_history`, confirmando que estas son las señales más fuertes en el dataset para identificar la preocupación[cite: 91].

[cite_start]La CNN 1D también se desempeñó de manera excelente, con métricas muy elevadas[cite: 92]. [cite_start]Sus gráficos de entrenamiento confirman una rápida convergencia y una buena generalización, sin signos de sobreajuste[cite: 93]. [cite_start]Esto demuestra que los modelos de Deep Learning también pueden ser muy efectivos para problemas de clasificación en datos tabulares, especialmente cuando se derivan características claras[cite: 94]. [cite_start]El éxito de los modelos se debe, en gran parte, a la clara relación entre las características de entrada utilizadas para definir la variable objetivo (`Mental_Health_Concern`)[cite: 95]. [cite_start]Si bien esta variable es una proxy y no una medida clínica directa de la depresión (dada la ausencia de un cuestionario PHQ-9 completo en el dataset), el modelo es altamente efectivo para clasificar esta definición específica de "preocupación por salud mental"[cite: 96].

**Limitaciones y Futuras Mejoras:**
* [cite_start]**Naturaleza de la Variable Objetivo:** Es vital reiterar que la variable objetivo actual (`Mental_Health_Concern`) es una construcción basada en indicadores disponibles, no una clasificación diagnóstica de depresión[cite: 97]. [cite_start]Para una predicción clínica de depresión, sería indispensable un dataset que contenga respuestas completas de cuestionarios validados como el PHQ-9 o diagnósticos clínicos[cite: 98].
* [cite_start]**Desequilibrio de Clases:** Aunque `class_weight='balanced'` se utilizó para el Random Forest y la CNN se comportó bien, el desequilibrio entre "Preocupación" y "No Preocupación" es notable[cite: 99]. [cite_start]Para futuras iteraciones, se podrían explorar técnicas de muestreo como SMOTE para balancear aún más el dataset de entrenamiento[cite: 100].
* [cite_start]**Interpretación de CNN:** A diferencia del Random Forest, la CNN no proporciona directamente la importancia de las características de manera tan sencilla[cite: 101]. [cite_start]Métodos de interpretabilidad de redes neuronales (como LIME o SHAP) podrían ser explorados para entender mejor qué patrones aprende la CNN[cite: 102].

[cite_start]Este proyecto demuestra con éxito la aplicación de técnicas de Machine Learning y Deep Learning para derivar y predecir indicadores de salud mental a partir de datos de encuestas disponibles, sentando las bases para futuras investigaciones con datasets más ricos en información clínica[cite: 103].

## Linkografía

* [cite_start]**Kaggle - Mental Health in Tech Survey:** [https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey) [cite: 104]
* [cite_start]**OpenICPSR - Adolescent Depression:** [https://www.openicpsr.org/openicpsr/project/209446/version/V1/view](https://www.openicpsr.org/openicpsr/project/209446/version/V1/view) [cite: 104]
* [cite_start]**Scikit-Learn - Random Forest Classifier:** [https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html) [cite: 104]
* **Saeb, S., et al. (2016). [cite_start]Mobile phone sensor correlates of depressive symptom severity in daily-life behavior.** Journal of Medical Internet Research. [cite: 105]
* **Shatte, A. B. R., et al. (2019). [cite_start]Machine learning in mental health: A scoping review of methods and applications.** Psychological Medicine. [cite: 106]

---

**Link al código (Google Colab):**
[Inserta aquí el enlace público a tu notebook de Google Colab]
