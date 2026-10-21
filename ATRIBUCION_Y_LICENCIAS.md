# Atribución y licencias

## Datos de terceros

### Leg-KILO (Unitree Go1)

- **Autores:** Guangjun Ou, Dong Li y Hanmin Li.
- **Publicación:** Ou, G., Li, D., & Li, H. (2024). Leg-KILO: Robust Kinematic-Inertial-Lidar Odometry for Dynamic Legged Robots. *IEEE Robotics and Automation Letters, 9*(10), 8194–8201. https://doi.org/10.1109/LRA.2024.3440730
- **Dataset:** https://github.com/ouguangjun/legkilo-dataset
- **Código asociado:** https://github.com/ouguangjun/Leg-KILO (licencia MIT, **solo para el código**).
- **Licencia de los datos:** no declarada. No se ha confirmado ningún permiso para redistribuir las mediciones originales ni sus derivados.
- **Uso en este trabajo:** secuencia `running`, tópico `/high_state`. Los datos originales no fueron registrados por el autor de este trabajo. Su aporte consiste en la extracción de las 12 articulaciones, la selección de 30 000 mediciones, la generación de los CSV, el diseño de la base SQLite, el catálogo de motores (elaborado para este trabajo siguiendo la numeración del SDK de Unitree), las consultas, las visualizaciones y los modelos de detección.
- **Estado en esta versión pública:** no se incluyen mediciones derivadas. El notebook omite las salidas que las reproducen, y las figuras incluidas son agregadas.

### voraus-AD

- **Autores:** Jan Thieß Brockmann, Marco Rudolph, Bodo Rosenhahn y Bastian Wandt.
- **Publicación:** Brockmann, J. T., Rudolph, M., Rosenhahn, B., & Wandt, B. (2024). The voraus-AD Dataset for Anomaly Detection in Robot Applications. *IEEE Transactions on Robotics, 40*, 438–451. https://doi.org/10.1109/TRO.2023.3332224
- **Dataset:** https://github.com/vorausrobotik/voraus-ad-dataset
- **Licencia del dataset:** Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0), https://creativecommons.org/licenses/by-nc-sa/4.0/. El código de ese repositorio tiene licencia MIT.
- **Derivados incluidos:** `datos/voraus_ad/particiones_ensayos.csv`, las métricas de `resultados/metricas/` y las figuras `voraus_*.png`. Se distribuyen **bajo CC BY-NC-SA 4.0**, con atribución a los autores y sin fines comerciales. Cambios realizados: selección de ensayos, partición, extracción de características, entrenamiento de modelos y cálculo de métricas.
- **No incluido:** el archivo Parquet original ni las particiones procesadas en CSV.

## Bibliotecas

Python, Pandas, NumPy, PyArrow, Matplotlib, Bokeh, Scikit-learn, TensorFlow, Keras, joblib y rosbags se usan bajo sus propias licencias de código abierto. No se redistribuye ninguna de ellas en este repositorio.

## Trabajo propio

El notebook, los scripts y la documentación son trabajo académico de Jaime Enrique Villegas Medina para la asignatura Herramientas para Inteligencia Artificial (Maestría en Inteligencia Artificial Aplicada, UTPL). No se ha asignado todavía una licencia de reutilización a este código.
