# 5. Conclusiones

[Inicio](Home.md) | [Anterior](04-Metodologia-y-Resultados.md) | [Siguiente](06-Bibliografia.md)

## 5.1. Sobre las herramientas

1. Google Colab, Pandas, NumPy, SQLite, Matplotlib, Bokeh, Scikit-learn y TensorFlow/Keras bastaron para construir el flujo completo: extracción desde ROS bag, almacenamiento en CSV y SQLite, consultas SQL, integración, visualización estática e interactiva, entrenamiento de modelos y evaluación.
2. Generar los scripts SQL desde Python y validar la base con `PRAGMA integrity_check` y `PRAGMA foreign_key_check` permite reconstruir la base sin depender de un archivo `.db` previo.
3. Leer el Parquet por columnas y por lotes con PyArrow permitió trabajar con más de 2,3 millones de registros en Colab sin cargar el archivo completo en memoria.

## 5.2. Sobre el Unitree Go1

4. En la secuencia analizada, las rodillas soportan el mayor torque absoluto promedio (7,49 N·m), frente a 3,53 N·m en los muslos y 2,02 N·m en las caderas laterales. También son las articulaciones más calientes (62,75 °C de media).
5. Isolation Forest marcó 134 mediciones atípicas (1,49 %) y el autoencoder 297 (3,30 %), con **una sola coincidencia**. Sin etiquetas no se puede decidir qué modelo acierta. Estas detecciones son desviaciones estadísticas, no fallas mecánicas.

## 5.3. Sobre la evaluación con voraus-AD

6. Las etiquetas de voraus-AD mostraron que los modelos con variables instantáneas apenas superan a la línea base trivial (F1 de 0,4566 frente a 0,4413 por registro). Un recall alto no sirve si va acompañado de muchas falsas alarmas.
7. Las puntuaciones dependían de la acción que ejecutaba el robot. Por eso un umbral único por registro producía falsas alarmas en casi todas las acciones.
8. Las características temporales (144 por ventana de 100 muestras) y la evaluación por ensayo mejoraron claramente al autoencoder. En los 1 127 ensayos reservados, el **Autoencoder V3** obtuvo F1 = 0,6880, ROC-AUC = 0,8372, precision = 0,7572, recall = 0,6304 y FPR = 13,94 %, por encima de Isolation Forest V3 (F1 = 0,5772, ROC-AUC = 0,7035, FPR = 20,09 %).
9. Aun así, el Autoencoder V3 no detectó 170 de los 460 ensayos anómalos y generó 93 falsas alarmas en 667 ensayos nominales. No es adecuado para aplicaciones críticas sin más trabajo.

## 5.4. Sobre la reproducibilidad

10. Guardar los modelos, los escaladores, la configuración y los umbrales permitió reproducir exactamente las predicciones (diferencia de 0,0). Volver a entrenar el autoencoder con la misma semilla dio resultados algo distintos (F1 0,6880 frente a 0,6949 en una ejecución anterior). Reproducir la inferencia no garantiza reproducir el entrenamiento, y conviene informar la variabilidad entre semillas.

## 5.5. Limitaciones

- El Unitree Go1 no tiene etiquetas de falla verificadas.
- voraus-AD es un manipulador de seis ejes con anomalías inducidas y etiquetadas por ensayo.
- `setting` se asocia por completo con la clase en los datos analizados.
- Los ensayos reservados proceden de la misma campaña experimental que los de desarrollo.
- No se evaluó la generalización entre plataformas robóticas.

## 5.6. Trabajo futuro

- Modelos que procesen secuencias completas (redes recurrentes o convolucionales 1D).
- Estudio de la variabilidad del entrenamiento con varias semillas e intervalos de confianza.
- Particiones que separen configuraciones `setting` para controlar los factores de confusión.
- Aplicación a datos de robots cuadrúpedos con anomalías anotadas.
- Métodos de interpretación que relacionen cada alarma con las señales y articulaciones responsables.

## 5.7. Conclusión general

El trabajo aplicó las herramientas de la asignatura a datos robóticos reales y mostró que la evaluación con etiquetas es imprescindible para valorar un detector de anomalías. Con características temporales y evaluación por ensayo, el autoencoder alcanzó una capacidad de discriminación apreciable (ROC-AUC 0,84) en voraus-AD. Este resultado es evidencia experimental sobre ese dataset y bajo este protocolo, no una validación de detección de fallas reales.

---

[Siguiente: Bibliografía](06-Bibliografia.md)
