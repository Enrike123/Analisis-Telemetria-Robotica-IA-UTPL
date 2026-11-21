# Atribución y licencias

Este repositorio combina tres tipos de contenido con condiciones distintas: **datos de terceros**, **datos y resultados derivados** y **código propio**. Ninguna licencia de datos se extiende al código, y ninguna licencia de código se extiende a los datos.

| Componente | Archivos | Licencia o condición |
|---|---|---|
| Derivados de Leg-KILO | `datos/unitree_go1/*.csv`, `sql/insertar_datos.sql`, figuras `go1_*.png` y salidas del notebook sobre el Go1 | CC BY 4.0 |
| Derivados de voraus-AD | `datos/voraus_ad/particiones_ensayos.csv`, `resultados/metricas/*`, figuras `voraus_*.png` y salidas del notebook sobre voraus-AD | CC BY-NC-SA 4.0 |
| Código y documentación propios | Notebook (código), `scripts/`, `sql/crear_base_datos.sql`, catálogo de motores, documentación | Trabajo académico del autor, sin licencia de reutilización asignada |
| Bibliotecas de terceros | No se redistribuyen | Sus propias licencias |

## Leg-KILO (Unitree Go1)

- **Autores:** Guangjun Ou, Dong Li y Hanmin Li.
- **Publicación:** Ou, G., Li, D., & Li, H. (2024). Leg-KILO: Robust Kinematic-Inertial-Lidar Odometry for Dynamic Legged Robots. *IEEE Robotics and Automation Letters, 9*(10), 8194–8201. https://doi.org/10.1109/LRA.2024.3440730
- **Dataset original:** https://github.com/ouguangjun/legkilo-dataset (secuencia `running`, archivo `running.bag`).
- **Licencia del dataset:** las grabaciones del dataset (archivos rosbag) y los datos de referencia (ground truth) que se distribuyen mediante los enlaces de descarga de su README están bajo **Creative Commons Attribution 4.0 International (CC BY 4.0)**, https://creativecommons.org/licenses/by/4.0/.
  - Fuente de la licencia: sección "Dataset License" del README del dataset, añadida por su autor en el commit [`bbdb842`](https://github.com/ouguangjun/legkilo-dataset/commit/bbdb84250876bebfaa1c6b44e0625738d4f37f07) (26/10/2026). Ese mismo texto indica que la licencia se aplica solo a las grabaciones y a los datos de referencia, y **no** al código fuente, a los SDK ni a otros contenidos del repositorio.
  - El código del proyecto Leg-KILO (https://github.com/ouguangjun/Leg-KILO) tiene licencia MIT, que se aplica solo a ese código. Este repositorio no redistribuye código ni SDK de Leg-KILO.
- **Cambios realizados sobre los datos originales:**
  1. Lectura del tópico `/high_state` de `running.bag` (primeros 12 elementos de `motorState`: `q`, `dq`, `tauEst` y `temperature`).
  2. Selección de 2 500 instantes equiespaciados (30 000 mediciones = 2 500 × 12 motores) y cálculo del tiempo relativo.
  3. Conversión a CSV (`unitree_go1_telemetria_30000.csv`).
  4. Generación de scripts SQL y de una base SQLite (`insertar_datos.sql`).
  5. Integración con un catálogo de motores (pata y articulación) elaborado para este trabajo, siguiendo la numeración del SDK de Unitree (`unitree_go1_dataset_integrado.csv`).
  6. Cálculo de estadísticas, visualizaciones y detección de anomalías.
- **Licencia de los derivados:** los CSV, `insertar_datos.sql` y las figuras y salidas calculadas con estos datos se distribuyen bajo **CC BY 4.0**. Al reutilizarlos debe citarse la publicación de Ou, Li y Li (2024), enlazar la licencia e indicar los cambios. La licencia no se aplica al código de este repositorio.
- **No incluido:** `running.bag` (unos 216 MB), por tamaño. Se descarga de la fuente oficial.

## voraus-AD

- **Autores:** Jan Thieß Brockmann, Marco Rudolph, Bodo Rosenhahn y Bastian Wandt.
- **Publicación:** Brockmann, J. T., Rudolph, M., Rosenhahn, B., & Wandt, B. (2024). The voraus-AD Dataset for Anomaly Detection in Robot Applications. *IEEE Transactions on Robotics, 40*, 438–451. https://doi.org/10.1109/TRO.2023.3332224
- **Dataset:** https://github.com/vorausrobotik/voraus-ad-dataset
- **Licencia del dataset:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0), https://creativecommons.org/licenses/by-nc-sa/4.0/. El código de ese repositorio tiene licencia MIT.
- **Derivados incluidos:** `datos/voraus_ad/particiones_ensayos.csv`, las métricas de `resultados/metricas/` y las figuras `voraus_*.png`. Se distribuyen **bajo CC BY-NC-SA 4.0**, con atribución a los autores y sin fines comerciales. Cambios realizados: selección de ensayos, partición, extracción de características, entrenamiento de modelos y cálculo de métricas.
- **No incluido:** el archivo Parquet original (1,1 GB) ni las particiones procesadas en CSV (520 MB).

## Bibliotecas

Python, Pandas, NumPy, PyArrow, Matplotlib, Bokeh, Scikit-learn, TensorFlow, Keras, joblib y rosbags se usan bajo sus propias licencias de código abierto. No se redistribuye ninguna de ellas en este repositorio.

## Trabajo propio

El notebook, los scripts, `sql/crear_base_datos.sql`, el catálogo de motores y la documentación son trabajo académico de Jaime Enrique Villegas Medina y del Ing. Jorge Luis Castro Valdivia para la asignatura Herramientas para Inteligencia Artificial (Maestría en Inteligencia Artificial Aplicada, UTPL). No se les ha asignado una licencia de reutilización.
