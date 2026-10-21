# 2. Marco teórico de las tecnologías y bibliotecas

[Inicio](Home.md) | [Anterior](01-Introduccion.md) | [Siguiente](03-Descripcion-Datasets.md)

## 2.1. Entorno de trabajo

**Google Colab** es un servicio de notebooks Jupyter que se ejecuta en el navegador sobre máquinas de Google, con acceso opcional a GPU y a Google Drive. Se eligió frente a JupyterLab porque no requiere instalación local y permite compartir el notebook directamente. El notebook final se ejecutó con Python 3.13 y una GPU T4.

**GitHub** aloja el código y los resultados con control de versiones, y su **Wiki** permite publicar documentación en Markdown junto al repositorio.

## 2.2. Procesamiento y almacenamiento de datos

**Pandas** ofrece la estructura `DataFrame` para leer, transformar, agregar y combinar datos tabulares (McKinney, 2010). En este trabajo lee CSV y Parquet, ejecuta consultas SQL con `read_sql_query`, integra fuentes con `merge` y agrupa resultados por motor, articulación, ensayo o categoría.

**NumPy** proporciona arreglos multidimensionales y operaciones vectorizadas (Harris et al., 2020). Se usa para construir las matrices de entrada de los modelos, segmentar las señales en ventanas y calcular estadísticas.

**SQLite** es un motor de base de datos relacional que guarda toda la base en un archivo y no necesita servidor. Se accede desde Python con el módulo estándar `sqlite3`. Aquí se usa para crear una base con claves primarias y foráneas, validar su integridad con `PRAGMA integrity_check` y `PRAGMA foreign_key_check`, y hacer consultas con `JOIN` y agregaciones.

**PyArrow** permite leer archivos Parquet por columnas y por lotes. Así se procesa el archivo de voraus-AD (alrededor de 1 GB) sin cargarlo entero en memoria.

**rosbags** lee archivos ROS bag sin instalar ROS. Solo se usa en el modo de ejecución que extrae los datos desde `running.bag`.

## 2.3. Visualización

**Matplotlib** genera gráficos estáticos: series temporales, barras, dispersión, diagramas de caja y matrices de confusión (Hunter, 2007).

**Bokeh** genera gráficos interactivos en el navegador, con zoom, desplazamiento y etiquetas emergentes (`HoverTool`). En el notebook muestra la velocidad de un motor y el torque de las cuatro rodillas del Go1.

## 2.4. Aprendizaje automático: Isolation Forest

**Scikit-learn** reúne algoritmos de aprendizaje automático, transformadores y métricas con una interfaz común (Pedregosa et al., 2011).

**Isolation Forest** (Liu et al., 2008, 2012) detecta anomalías aislando observaciones mediante particiones aleatorias. Cada árbol elige al azar una variable y un punto de corte. Las observaciones raras se aíslan con menos cortes, así que la longitud media del camino en el bosque sirve como puntuación de anomalía. No necesita etiquetas y escala bien con muchos datos. Sus hiperparámetros principales son el número de árboles (`n_estimators`), las muestras por árbol (`max_samples`) y la fracción de variables (`max_features`).

## 2.5. Aprendizaje profundo: autoencoders

**TensorFlow** es una plataforma para definir y entrenar modelos de aprendizaje profundo (Abadi et al., 2016). **Keras** es su interfaz de alto nivel para construir redes por capas.

Un **autoencoder** es una red que aprende a reproducir su entrada pasando por una representación comprimida (espacio latente) (Hinton y Salakhutdinov, 2006). Si se entrena solo con datos nominales, reconstruye bien los patrones habituales y peor los poco frecuentes. El **error de reconstrucción** (aquí, el error cuadrático medio) se usa como puntuación de anomalía, y el umbral se elige con un conjunto de validación. Un error alto indica que el patrón estaba poco representado en el entrenamiento, no que exista una falla física.

## 2.6. Características temporales

Una medición aislada no describe cómo cambia una señal. Por eso las señales se dividen en **ventanas** de longitud fija y en cada una se calculan estadísticas: media, desviación estándar, RMS, máximo absoluto, rango y cambio absoluto medio entre muestras consecutivas. Con 24 señales y 6 estadísticas se obtienen 144 características por ventana. Las puntuaciones de las ventanas de un mismo ensayo se resumen en un percentil para clasificar el ensayo completo.

## 2.7. Métricas de evaluación

Con VP, FP, VN y FN como verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos:

| Métrica | Definición | Interpretación |
|---|---|---|
| Precision | VP / (VP + FP) | Proporción de alarmas que son anomalías reales |
| Recall | VP / (VP + FN) | Proporción de anomalías detectadas |
| F1-score | 2·P·R / (P + R) | Equilibrio entre precision y recall |
| FPR | FP / (FP + VN) | Proporción de casos nominales con falsa alarma |
| ROC-AUC | Área bajo la curva ROC | Capacidad de ordenar anómalos por encima de nominales, sin depender del umbral (0,5 equivale al azar) |
| Average Precision | Resumen de la curva precision-recall | Más informativa que ROC-AUC cuando hay desbalance de clases |

Precision, recall, F1 y FPR dependen del umbral elegido; ROC-AUC y AP no.

---

[Siguiente: Descripción de los datasets](03-Descripcion-Datasets.md)
