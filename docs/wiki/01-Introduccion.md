# 1. Introducción

[Inicio](Home.md) | [Siguiente: Marco teórico](02-Marco-Teorico.md)

## 1.1. Contexto

Los robots registran de forma continua el estado de sus articulaciones: posición, velocidad, torque, corriente y temperatura. Estos datos permiten estudiar su funcionamiento y son la materia prima de las aplicaciones de inteligencia artificial para supervisión y mantenimiento. Para aprovecharlos se necesitan herramientas que lean formatos distintos, organicen la información, la representen gráficamente y apliquen modelos de aprendizaje automático.

## 1.2. Problema

Detectar comportamientos atípicos en la telemetría de un robot no es sencillo:

- Las señales cambian mucho según la tarea que ejecuta el robot, así que un valor poco frecuente no es necesariamente un fallo.
- Muchos registros reales, como los del Unitree Go1 usados aquí, no tienen etiquetas de falla verificadas. Sin ellas no se puede medir si un modelo acierta.
- Un modelo puede parecer bueno con una métrica (por ejemplo, el recall) y a la vez generar demasiadas falsas alarmas.

## 1.3. Objetivo general

Desarrollar y evaluar, en Google Colab, un flujo de procesamiento, almacenamiento, visualización y detección de anomalías en telemetría robótica con bibliotecas de software libre. Se combina el análisis exploratorio del Unitree Go1 con una evaluación cuantitativa sobre el dataset etiquetado voraus-AD.

## 1.4. Objetivos específicos

1. Extraer y procesar telemetría articular del Unitree Go1 desde un archivo ROS bag y guardarla en CSV.
2. Construir y consultar una base de datos relacional SQLite generada automáticamente con scripts SQL.
3. Integrar la información del CSV y de SQLite con Pandas y generar un dataset final.
4. Representar las variables con Matplotlib y Bokeh.
5. Aplicar Isolation Forest (Scikit-learn) y un autoencoder (TensorFlow/Keras) para detectar mediciones atípicas.
6. Evaluar ambos modelos con las etiquetas de voraus-AD y comparar características instantáneas con características temporales.
7. Documentar el trabajo para que sea reproducible con Google Colab y GitHub.

## 1.5. Requisitos de la actividad

| Requisito | Cómo se aborda |
|---|---|
| JupyterLab o Google Colab | Google Colab |
| Dos fuentes de datos, una CSV y otra en base de datos | Dos fuentes robóticas independientes (Leg-KILO y voraus-AD). La telemetría del Go1 se procesa en CSV y se almacena y consulta en SQLite (página 3) |
| Pandas o similar | Pandas para leer CSV y Parquet, consultar SQL e integrar datos |
| Matplotlib, Bokeh o PyWalker | Matplotlib y Bokeh |
| Archivos en GitHub | Dataset inicial y dataset final (CSV), SQL de creación e inserción, notebook que genera la base, scripts, métricas, figuras e instrucciones de reproducción |

## 1.6. Alcance y delimitaciones

- **Experimento A (Unitree Go1):** procesamiento CSV/SQLite, visualización y detección **exploratoria**. Las mediciones atípicas no se interpretan como fallas mecánicas.
- **Experimento B (voraus-AD):** evaluación cuantitativa de los modelos con anomalías **inducidas** en un manipulador de seis ejes. Sus resultados no se trasladan al Unitree Go1.
- No se diseñaron modelos para uso operativo; el objetivo es aplicar y evaluar las herramientas estudiadas en la asignatura.

---

[Siguiente: Marco teórico](02-Marco-Teorico.md)
