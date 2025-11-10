# Clasificación Automática de Tickets de Quejas de Clientes con PLN (NLP)

## Integrantes

* Bellido Santa María José Boris (jboris.bsm@gmail.com)

* Bautista Arcani Mayra (995.mayr.995@gmail.com)

* Zapata Inturias Giovany Lucas (giovazapata666@gmail.com)

## Descripción del Proyecto

Este repositorio contiene el trabajo de un proyecto de **Procesamiento de Lenguaje Natural (PLN)** enfocado en la **clasificación automática de tickets de quejas de clientes** de una entidad financiera.

El objetivo central es desarrollar un modelo capaz de clasificar las quejas no etiquetadas, ubicadas en el archivo de datos (`_dataset/complaints.json`), en categorías relevantes de productos o servicios. Esta clasificación inicial permite la segregación eficiente de los tickets a los departamentos correspondientes, facilitando una resolución más rápida y efectiva del problema.

---

## Objetivo

Desarrollar un **flujo de trabajo integral** que utilice el **Topic Modeling**, específicamente la Factorización de Matrices No Negativas (NMF), para etiquetar el conjunto de datos inicial y, posteriormente, entrenar un **modelo de clasificación supervisado** que pueda clasificar cualquier nuevo ticket de soporte en su departamento o categoría correspondiente.

---

## Clasificaciones Temáticas

Dado que los datos iniciales no estaban etiquetados, se aplicó NMF para analizar patrones y clasificar los tickets en cinco grupos temáticos principales basados en los productos y servicios de la entidad:

1. **Tarjetas de Crédito / Tarjetas Prepagadas** (*Credit card / Prepaid Card*)
2. **Servicios de Cuentas de Banco** (*Bank account services*)
3. **Reportes de Robos / Disputas** (*Theft/Dispute reporting*)
4. **Préstamos Hipotecarios y Otros Préstamos** (*Mortgages/loans*)
5. **Otros** (*Others*)

---

## Metodología y Flujo de Trabajo

El proyecto se estructura en un **flujo de trabajo con ocho tareas principales** que abarcan desde la preparación de los datos hasta la inferencia del modelo:

1. **Carga de Datos** (*Loading the data*). Los datos del archivo .json son cargados a un DataFrame de pandas.
2. **Preparación de Datos** (Data preparation). Convertir el texto en minúsculas y eliminar las filas que no tengan información en la columna `complaint`.
3. **Preparación del texto para el modelado de tópicos** (*Prepare the text for topic modeling*): Incluye limpieza, tokenización, lematización y filtrado por *Part-of-Speech* (manteniendo únicamente sustantivos 'NN').
4. **Análisis Exploratorio de Datos** (Exploratory data analysis to get familiar with the data): Visualiza la distribución de la longitud de las quejas, generar una nube de palabras con las 40 palabras más frecuentes post-procesamiento y encontrar los top unigramas, bigramas y trigramas por frecuencia.
5. **Extracción de Características** (*Feature Extraction*): Uso de `TfidfVectorizer` para generar la Matriz de Términos y Documentos (DTM).
6. **Modelado de tópicos usando NMF** (*Topic Modeling using NMF*): Aplicación de **NMF** para determinar y asignar las etiquetas temáticas.
7. **Modelado manual de tópicos** (*Manual Topic Modeling*): Encontrar los tópicos más importantes.
8. **Modelo supervisado para clasificar nuevas quejas en los tópicos correspondientes** (Supervised model to predict any new complaints to the relevant Topics): Configuración y entrenamiento de modelos de clasificación.

---

## Tecnologías y Bibliotecas

El desarrollo se llevó a cabo en **Python** y se apoyó en las siguientes librerías principales:

| Área                    | Librerías Principales                                                                                                       |
|:----------------------- |:--------------------------------------------------------------------------------------------------------------------------- |
| **Data Science**        | `pandas`, `numpy`                                                                                                           |
| **NLP**                 | `nltk`, `spacy` (`en_core_web_sm`)                                                                                          |
| **Modelado de tópicos** | `sklearn.decomposition.NMF`                                                                                                 |
| **Clasificación**       | `sklearn.linear_model.LogisticRegression`, `sklearn.tree.DecisionTreeClassifier`, `sklearn.ensemble.RandomForestClassifier` |
| **Métricas**            | `sklearn.metrics` (`accuracy_score`, `f1_score`, `classification_report`)                                                   |
| **Visualización**       | `seaborn`, `matplotlib`, `plotly`, `wordcloud`                                                                              |

---

## Instalación de librerías principales

`pip install pandas numpy scikit-learn nltk spacy seaborn matplotlib plotly wordcloud`

# Descarga del modelo de SpaCy y recursos de NLTK

python -m spacy download en_core_web_sm
python -m nltk download stopwords
python -m nltk download punkt
python -m nltk download wordnet
python -m nltk download omw-1.4

### Resultados y Evaluación del Modelo

Se evaluaron múltiples modelos de clasificación supervisada para predecir la etiqueta de tema generada por NMF. A continuación, se presenta un resumen de las métricas de rendimiento obtenidas en el conjunto de prueba:

| Modelo                                      | Precisión (*Accuracy*) | F1-Macro   |
|:------------------------------------------- |:----------------------:|:----------:|
| Árbol de Decisión (*Decision Tree*)         | **1.0000**             | **1.0000** |
| Regresión Logística (*Logistic Regression*) | 0.8000                 | 0.7333     |
| Naive Bayes                                 | 0.8000                 | 0.7333     |
| Bosque Aleatorio (*Random Forest*)          | 0.6000                 | 0.4667     |

El modelo seleccionado como el de mejor rendimiento fue el **Árbol de Decisión** (*Decision Tree*), logrando una precisión y una puntuación F1-Macro perfectas en el conjunto de prueba evaluado.
