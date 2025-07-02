# Predicción de Niveles de Depresión utilizando PHQ-9 y Aprendizaje Automático

## I. Introducción

### 1.1 Título del Proyecto
Predicción de Niveles de Depresión utilizando PHQ-9 y Aprendizaje Automático

### 1.2 Antecedentes
La **depresión** es un trastorno de salud mental de alta incidencia a nivel global, y su detección temprana es fundamental para la prevención y el tratamiento oportuno. Tradicionalmente, la evaluación se basa en valoraciones clínicas y cuestionarios estandarizados. El **PHQ-9 (Patient Health Questionnaire-9)** es un cuestionario validado clínicamente que permite evaluar la severidad de los síntomas depresivos. En este proyecto, buscamos automatizar la clasificación de los niveles de depresión (Ninguna, Leve, Moderada, Moderadamente Severa, Severa) utilizando algoritmos de **Machine Learning (ML)** y **Deep Learning (DL)**, lo que facilita una identificación más rápida y precisa.

### 1.3 Problema a Resolver
¿Cómo predecir de manera automática y precisa el nivel de depresión de un individuo a partir de sus respuestas al cuestionario **PHQ-9**, empleando técnicas de Aprendizaje Supervisado como **Random Forest** y una **Red Neuronal Multicapa (MLP)**?

### 1.4 Objetivos

**Objetivo General:**
Desarrollar un sistema de predicción de niveles de depresión mediante el análisis del cuestionario PHQ-9 y técnicas de Aprendizaje Supervisado.

**Objetivos Específicos:**
* Realizar el preprocesamiento y normalización de datos provenientes del PHQ-9.
* Implementar y comparar el rendimiento de modelos de clasificación como Random Forest y una Red Neuronal Multicapa (MLP) con backpropagation.
* Evaluar el desempeño de los modelos utilizando métricas de clasificación estándar como precisión (accuracy), recall y F1-score.
* Publicar el código del proyecto y los modelos entrenados en un repositorio en línea (GitHub) para facilitar su acceso y reutilización.

---

## II. Requerimientos del Sistema

### 2.1 Definición del Dominio
El proyecto se enmarca en la **Inteligencia Artificial aplicada a la Salud Mental y la Ciencia de Datos en Salud**. Su objetivo es crear un sistema predictivo capaz de clasificar automáticamente los niveles de depresión en personas, utilizando como base las respuestas al **cuestionario clínico PHQ-9** y técnicas avanzadas de Aprendizaje Supervisado (ML y DL).

Este trabajo integra principios de análisis estadístico, modelado predictivo y procesamiento de datos clínicos. Busca aportar soluciones innovadoras que puedan ser implementadas en diversos entornos, incluyendo:
* **Entornos clínicos:** Centros de atención primaria, servicios de salud pública.
* **Aplicaciones digitales:** Telemedicina, plataformas de bienestar emocional, herramientas de autoevaluación.

La combinación de librerías como **Scikit-learn** y **TensorFlow** permite construir modelos robustos que, una vez entrenados, pueden integrarse en sistemas de apoyo a la toma de decisiones médicas o en plataformas digitales, ofreciendo una alternativa tecnológica eficiente para la detección temprana y el seguimiento de trastornos depresivos.

### 2.2 Determinación de Requisitos (Requerimientos)

**Requerimientos Funcionales:**
* El sistema debe permitir la **carga de datos** en formato CSV (que contenga las respuestas al PHQ-9).
* El sistema debe permitir el **preprocesamiento de datos**, incluyendo la gestión de valores nulos y la normalización de variables numéricas.
* El sistema debe **calcular el puntaje total del cuestionario PHQ-9** para cada registro.
* El sistema debe permitir la **clasificación de los registros en cinco categorías** de nivel de depresión (Ninguna, Leve, Moderada, Moderadamente Severa, Severa).
* El sistema debe **codificar las etiquetas de salida** en formato numérico para el entrenamiento de los modelos.
* El sistema debe implementar un modelo de clasificación mediante **Random Forest**.
* El sistema debe implementar un modelo de clasificación mediante una **Red Neuronal Multicapa (MLP)** con backpropagation.
* El sistema debe **dividir automáticamente el dataset** en conjuntos de entrenamiento, prueba y validación.
* El sistema debe **evaluar los modelos** generando métricas como precisión (accuracy), recall, F1-score y matriz de confusión.
* El sistema debe permitir **guardar los modelos entrenados** en formatos estándar (.pkl para Random Forest y .h5 para la Red Neuronal).

**Requerimientos No Funcionales:**
* El sistema debe garantizar **tiempos de ejecución razonables** durante el entrenamiento y la predicción.
* El sistema debe ser **ejecutable en entornos de desarrollo** como Google Colab o Jupyter Notebook.
* El código del sistema debe estar **documentado y estructurado** para facilitar su mantenimiento y futuras modificaciones.
* El sistema debe utilizar únicamente **librerías de código abierto** como Scikit-learn, TensorFlow y Pandas.
* El sistema debe permitir la **reutilización del modelo entrenado** en futuras implementaciones o aplicaciones.
* El sistema debe asegurar la **confidencialidad de los datos utilizados**, evitando la exposición de información personal identificable (se trabaja con datasets anonimizados).

---

## III. Pre-procesamiento y Normalización (Planteamiento del Data-Set; Aprendizaje Supervisado)

El presente proyecto se desarrolla bajo el enfoque de **Aprendizaje Supervisado**, ya que el dataset utilizado cuenta con una variable objetivo claramente definida: el nivel de depresión del individuo, categorizado en cinco clases según el puntaje total del cuestionario PHQ-9. Este es un problema de **clasificación multiclase**.

### 3.1 Medidas, Datos, Bases de Datos y Elaboración del Data-Set
El dataset empleado corresponde al archivo `"Dataset_14-day_AA_depression_symptoms_mood_and_PHQ-9.csv"`. Este contiene registros de autoevaluaciones diarias, incluyendo las respuestas a las nueve preguntas del cuestionario PHQ-9 (`phq1` a `phq9`), junto con otras variables.

Para este proyecto, se seleccionaron como **variables predictoras exclusivamente las columnas correspondientes a las nueve preguntas del PHQ-9**, dado que están directamente relacionadas con el diagnóstico clínico de depresión.

### 3.2 Normalización y/o Filtrado de Datos
Durante el preprocesamiento se realizaron las siguientes etapas clave:
* **Eliminación de registros con valores nulos** en cualquiera de las columnas `phq1` a `phq9`, garantizando la integridad de los datos.
* **Cálculo del puntaje total PHQ-9** sumando las respuestas de cada registro.
* **Clasificación de cada individuo** en una de las cinco categorías clínicas: "Ninguna", "Leve", "Moderada", "Moderadamente Severa" o "Severa", según los rangos estandarizados para el PHQ-9.
* **Codificación numérica de la variable objetivo** (`Depression_Level`) mediante `LabelEncoder`, para que los modelos puedan procesar estas etiquetas.
* **Aplicación de la técnica de normalización `StandardScaler`** para escalar las variables predictoras (las preguntas del PHQ-9), lo cual es fundamental para mejorar la convergencia y el rendimiento de la Red Neuronal.

### 3.3 Planteamiento de la División del Data-Set

* **Data-Set de entrenamiento (training):**
    El sistema dividió automáticamente el dataset, asignando el **80% de los registros** al conjunto de entrenamiento. Este subconjunto se utilizó para ajustar los parámetros internos de los modelos durante la fase de aprendizaje.
* **Data-Set de Pruebas (test):**
    El **20% restante** de los datos fue reservado para el conjunto de prueba. Este subconjunto, que los modelos no habían visto previamente, se utilizó para estimar su capacidad para predecir correctamente en datos nuevos y no observados.
* **Set de Validación por "Cross-Validation" (para MLP):**
    Para el modelo de Deep Learning (MLP), se aplicó un `validation_split` del **10% sobre el conjunto de entrenamiento**. Esto proporcionó un conjunto de validación interno para monitorear el rendimiento del modelo durante cada época de entrenamiento y aplicar la técnica de `EarlyStopping` para detener el entrenamiento de manera automática en caso de no detectar mejoras en la pérdida de validación, lo que ayuda a prevenir el sobreajuste.

---

## IV. Aprendizaje del Modelo del Sistema

### 4.1 Planteamiento del Modelo de Aprendizaje
Dado que el objetivo del proyecto es realizar una **clasificación multiclase** de los niveles de depresión, el enfoque utilizado fue el de **Aprendizaje Supervisado**, en el cual el sistema aprende a partir de ejemplos etiquetados (registros con su respectiva categoría de nivel de depresión).

Se implementaron dos tipos de modelos para realizar la predicción:
* **Random Forest Classifier:** Un algoritmo basado en un conjunto de árboles de decisión (ensemble learning), reconocido por ser robusto ante ruido en los datos y muy eficaz para tareas de clasificación multiclase.
* **Red Neuronal Multicapa (MLP - Multilayer Perceptron):** Una red neuronal de arquitectura `feedforward` que utiliza el algoritmo de `backpropagation` para el ajuste iterativo de los pesos sinápticos, permitiendo al modelo aprender relaciones no lineales complejas entre las variables de entrada y la salida.

### 4.2 Desarrollo e Implementación del Modelo

**Random Forest Classifier:**
* **Número de árboles:** 100 (`n_estimators=100`).
* **Parámetro de reproducibilidad:** `random_state=42`.
* Se utilizó el conjunto de entrenamiento normalizado (`X_train`).
* El modelo fue entrenado para identificar las cinco categorías de nivel de depresión.
* El modelo entrenado fue guardado en un archivo `.pkl` (`joblib.dump`).

**Red Neuronal Multicapa (MLP - Deep Learning):**
* **Arquitectura de la red (Modelo `Sequential` de Keras):**
    * **Capa de entrada (`Dense`):** 128 neuronas con función de activación `ReLU`.
    * **Capa oculta (`Dense`):** 64 neuronas con activación `ReLU`.
    * **Capa de salida (`Dense`):** 5 neuronas (correspondientes a las cinco clases de depresión) con función de activación `Softmax` (para clasificación multiclase).
* **Proceso de compilación:**
    * **Optimizador:** `Adam`.
    * **Función de pérdida:** `Sparse Categorical Crossentropy` (adecuada para clasificación multiclase con etiquetas enteras).
    * **Métrica de evaluación:** `Accuracy`.
* **Entrenamiento (`model_nn.fit`):**
    * **Epochs:** Hasta 100, con `EarlyStopping` para evitar `overfitting` (se detiene si no hay mejora en la pérdida de validación durante 5 épocas).
    * **Tamaño del batch:** 16 muestras por iteración.
    * Se utilizó un 10% del conjunto de entrenamiento como validación interna (`validation_data`).
* El modelo entrenado fue exportado y guardado en formato `.h5` (`model_nn.save`).

---

## V. Comprobación y Despliegue (Deploy) del Sistema

### 5.1 Entrenamiento del Modelo: Uso del Data-Set de entrenamiento
Ambos modelos fueron entrenados utilizando el 80% del dataset previamente preprocesado y normalizado.

* **Random Forest Classifier:** El modelo fue ajustado utilizando el conjunto de entrenamiento completo. Gracias a la simplicidad inherente de los árboles de decisión y el tamaño del dataset, logró una rápida convergencia sin necesidad de validación interna adicional durante su entrenamiento.
* **Red Neuronal Multicapa (MLP):** El modelo MLP fue entrenado durante un máximo de 100 épocas. Se aplicó un 10% del conjunto de entrenamiento como conjunto de validación interna, lo que permitió monitorear la pérdida y la precisión en cada época. Además, se implementó un `callback` de `EarlyStopping` que detuvo el entrenamiento tras detectar que la pérdida de validación no mejoraba durante 5 épocas consecutivas. Notablemente, la red alcanzó el **100% de precisión (accuracy) en el conjunto de entrenamiento en apenas 3 épocas**.

### 5.2 Ejecución y Pruebas del Modelo
Ambos modelos fueron evaluados exhaustivamente utilizando el conjunto de prueba (el 20% de los datos totales, no vistos durante el entrenamiento).

Los resultados obtenidos fueron los siguientes:

**Random Forest Classifier:**
* **Accuracy (Precisión): 100.00%**
* **Matriz de Confusión:** Sin errores de clasificación (predicciones perfectas).
* **Reporte de Clasificación:** Precisión, recall y F1-score de **1.00 en todas las clases** ("Ninguna", "Leve", "Moderada", "Moderadamente severa", "Severa").
