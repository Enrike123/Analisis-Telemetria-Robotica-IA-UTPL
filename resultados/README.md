# Resultados

Todos los archivos de esta carpeta proceden de **una sola ejecución**: la ejecución definitiva del notebook `Unitree_Go1_Trabajo_Academico_FINAL.ipynb` (83 celdas de código ejecutadas en orden, de la 1 a la 83, sin errores). No se combinan resultados de ejecuciones distintas.

## Figuras (`figuras/`)

Las figuras se **extrajeron de las salidas incrustadas en el notebook**, sin volver a generarlas ni editarlas. Por eso tienen la resolución de pantalla de Colab y no la de 300 ppp con que se guardaron las copias en Drive. Las copias de 300 ppp que existían en otras carpetas no se usaron, porque algunas correspondían a una ejecución anterior.

| Archivo | Sección del notebook | Contenido |
|---|---|---|
| `go1_8_1_posicion_motor1_serie.png` | 8.1 | Serie temporal de la posición del motor 1 |
| `go1_8_2_velocidad_motor1_serie.png` | 8.2 | Serie temporal de la velocidad del motor 1 |
| `go1_8_3_torque_promedio_por_articulacion.png` | 8.3 | Torque absoluto promedio por tipo de articulación |
| `go1_8_4_torque_promedio_rodillas.png` | 8.4 | Torque absoluto promedio de las cuatro rodillas |
| `go1_10_2_isolation_forest_dispersion.png` | 10.2 | Velocidad frente a torque de las 9 000 mediciones de prueba, con las atípicas resaltadas |
| `go1_10_2_isolation_forest_por_articulacion.png` | 10.2 | Porcentaje de mediciones atípicas por articulación |
| `go1_11_3_entrenamiento_autoencoder.png` | 11.3 | Curvas de pérdida del autoencoder |
| `go1_11_5_comparacion_anomalias_modelos.png` | 11.5 | Número de mediciones atípicas por modelo |
| `go1_11_7_anomalias_por_motor.png` | 11.7 | Mediciones atípicas por motor y modelo |
| `go1_11_7_dispersion_puntuaciones.png` | 11.7 | Puntuación de Isolation Forest frente al error del autoencoder por medición |
| `go1_11_8_correlacion_por_articulacion.png` | 11.8 | Correlación de Spearman entre puntuaciones, por articulación |
| `voraus_12_3_matriz_confusion_isolation_forest.png` | 12.3 | Matriz de confusión V1 por registro |
| `voraus_12_4_entrenamiento_autoencoder.png` | 12.4 | Curvas de pérdida del autoencoder V1 |
| `voraus_12_4_matriz_confusion_autoencoder.png` | 12.4 | Matriz de confusión V1 por registro |
| `voraus_12_5_comparacion_modelos.png` | 12.5 | Comparación V1 por registro |
| `voraus_12_6_matrices_confusion_por_ensayo.png` | 12.6.2 | Matrices de confusión V1 por ensayo |
| `voraus_12_6_recall_por_categoria.png` | 12.6.3 | Recall por categoría de anomalía (V1) |
| `voraus_12_6_comparacion_linea_base.png` | 12.6.4 | Comparación con la línea base "siempre anómalo" |
| `voraus_12_11_falsas_alarmas_por_accion.png` | 12.11 | Tasa de falsas alarmas por acción operativa |
| `voraus_12_12_puntuaciones_por_accion.png` | 12.12 | Distribución de puntuaciones nominales por acción |
| `voraus_12_16_comparacion_modelos_temporales_v3.png` | 12.16.4 | Comparación V3 en validación |
| `voraus_12_17_matrices_confusion_prueba_exploratoria_v3.png` | 12.17 | Matrices de confusión V3, prueba exploratoria (146 ensayos) |
| `voraus_12_19_evaluacion_reservada_v3.png` | 12.19 | Matrices de confusión V3, 1 127 ensayos reservados |
| `voraus_12_19_estabilidad_f1.png` | 12.19.1 | F1 en la prueba exploratoria y en los ensayos reservados |

Las 24 figuras del notebook están incluidas. Las 11 figuras `go1_*` se calculan con datos de Leg-KILO y las 13 figuras `voraus_*` con datos de voraus-AD.

## Métricas (`metricas/`)

| Archivo | Contenido |
|---|---|
| `metricas_holdout_v3.csv` | Métricas V3 en los 1 127 ensayos reservados (precision, recall, F1, ROC-AUC, AP, FPR, VN, FP, FN, VP) |
| `comparacion_generalizacion.csv` | Métricas V3 en la prueba exploratoria y en los ensayos reservados |
| `configuracion_v3.json` | Ventana, estadísticas, 144 columnas, hiperparámetros, percentiles, umbrales y semilla |
| `comparacion_linea_base_v1.csv` | Modelos V1 por registro frente a "siempre anómalo" |
| `metricas_por_ensayo_v1.csv` | Modelos V1 evaluados por ensayo (percentil 95) |
| `recall_por_categoria_v1.csv` | Recall V1 por categoría de anomalía |
| `predicciones_por_ensayo_v1.csv` | Puntuación y predicción V1 de cada uno de los 146 ensayos de prueba |

## Reproducibilidad (`reproducibilidad/`)

`entorno_python.json` registra la versión de Python y de las bibliotecas del entorno de Colab en el que se ejecutó el notebook.

## Licencia de los resultados derivados

- Las figuras y métricas de **voraus-AD** son obras derivadas de un dataset con licencia CC BY-NC-SA 4.0. Se distribuyen con atribución a Brockmann et al. (2024), sin fines comerciales y bajo la misma licencia CC BY-NC-SA 4.0.
- Las figuras de **Unitree Go1** son obras derivadas del dataset Leg-KILO (Ou et al., 2024), cuyas grabaciones y datos de referencia tienen licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) ([commit `bbdb842`](https://github.com/ouguangjun/legkilo-dataset/commit/bbdb84250876bebfaa1c6b44e0625738d4f37f07)). Se distribuyen bajo CC BY 4.0, con atribución a los autores. Cambios: extracción, muestreo, cálculo de estadísticas y modelos de detección.
