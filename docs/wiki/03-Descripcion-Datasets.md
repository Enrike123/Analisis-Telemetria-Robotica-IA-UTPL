# 3. Descripción de los datasets

[Inicio](Home.md) | [Anterior](02-Marco-Teorico.md) | [Siguiente](04-Metodologia-y-Resultados.md)

## 3.1. Dos fuentes independientes, dos formatos de trabajo

| | Leg-KILO (Unitree Go1) | voraus-AD |
|---|---|---|
| Robot | Cuadrúpedo, 12 motores | Manipulador de seis ejes |
| Formato original | ROS bag | Parquet |
| Etiquetas de anomalía | No | Sí, por ensayo |
| Formatos usados en el trabajo | CSV y base SQLite | Parquet y CSV |
| Finalidad | Procesamiento, almacenamiento, visualización y detección exploratoria | Evaluación cuantitativa de los modelos |

**Aclaración metodológica.** El CSV y la base SQLite del Unitree Go1 **no son dos datasets distintos**. La base se generó a partir del CSV: su tabla `mediciones` contiene las mismas 30 000 mediciones, y su tabla `motores` contiene un catálogo de 12 articulaciones elaborado para este trabajo. El requisito de usar CSV y una base de datos se cumple como flujo de trabajo (leer el CSV, construir y consultar la base, e integrar ambas fuentes con Pandas). La segunda fuente de datos realmente independiente es voraus-AD.

## 3.2. Leg-KILO: Unitree Go1

### Origen

El dataset Leg-KILO (Ou, Li y Li, 2024) fue grabado con un Unitree Go1 equipado con encoders articulares, sensores de contacto, una IMU y un LiDAR Velodyne VLP-16. Se usó la secuencia **`running`**, en la que, según sus autores, el robot recorre un circuito corto al aire libre a 1,5 m/s de media. Del tópico `/high_state` se tomaron los 12 primeros elementos de `motorState`. Estos datos no fueron registrados por el autor de este trabajo.

### Dimensiones y variables

| Característica | Valor |
|---|---:|
| Mediciones | 30 000 |
| Instantes (equiespaciados a lo largo de la secuencia) | 2 500 |
| Motores | 12 (4 patas × cadera lateral, muslo y rodilla) |
| Duración cubierta | 32,7 s aproximadamente |
| Etiquetas de falla | No disponibles |

| Variable | Unidad | Descripción |
|---|---|---|
| `instante` | – | Índice del mensaje en la secuencia |
| `tiempo_ros`, `tiempo_relativo_s` | s | Marca de tiempo ROS y tiempo desde el inicio |
| `motor_id` | – | 0 a 11 (numeración del SDK de Unitree: FR 0–2, FL 3–5, RR 6–8, RL 9–11) |
| `posicion_rad` | rad | Posición angular |
| `velocidad_rad_s` | rad/s | Velocidad angular |
| `torque_Nm` | N·m | Torque estimado por el controlador del robot |
| `temperatura_C` | °C | Temperatura del motor |
| `sesion` | – | Nombre de la secuencia (`running`) |

### Base de datos SQLite

```text
motores (motor_id PK, pata, articulacion)                       12 filas
mediciones (medicion_id PK, motor_id FK → motores, instante,    30 000 filas
            tiempo_ros, tiempo_relativo_s, posicion_rad,
            velocidad_rad_s, torque_Nm, temperatura_C, sesion,
            UNIQUE(sesion, instante, motor_id))
```

El dataset final (`unitree_go1_dataset_integrado.csv`) tiene 30 000 filas y 11 columnas: las 9 de la telemetría más `pata` y `articulacion`.

### Disponibilidad

Las grabaciones y los datos de referencia de Leg-KILO están bajo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), según la sección "Dataset License" del README del dataset ([commit `bbdb842`](https://github.com/ouguangjun/legkilo-dataset/commit/bbdb84250876bebfaa1c6b44e0625738d4f37f07), 26/10/2026); la licencia no cubre su código ni sus SDK. El repositorio incluye, bajo esa misma licencia y con atribución a Ou, Li y Li (2024), el [dataset inicial](../../datos/unitree_go1/unitree_go1_telemetria_30000.csv), el [dataset final](../../datos/unitree_go1/unitree_go1_dataset_integrado.csv) y el [script de inserción](../../sql/insertar_datos.sql), además del [script de creación](../../sql/crear_base_datos.sql). Solo `running.bag` (unos 216 MB) queda fuera por su tamaño; se descarga de la fuente oficial.

## 3.3. voraus-AD

### Origen

voraus-AD (Brockmann et al., 2024) contiene datos de un manipulador robótico de seis ejes que ejecuta repetidamente una misma tarea. Hay ensayos nominales y ensayos con anomalías inducidas de forma controlada. Se usó la versión de 100 Hz (`voraus-ad-dataset-100hz.parquet`), con licencia CC BY-NC-SA 4.0.

### Dimensiones

| Característica | Valor |
|---|---:|
| Registros | 2 321 690 |
| Columnas | 137 |
| Ensayos (`sample`) | 2 122 |
| Registros nominales / anómalos | 1 488 328 / 833 362 |
| Categorías | 13: anomalías 0–11 y nominal 12 |
| Valores distintos de `setting` | 77 |

Cada ensayo tiene una única etiqueta (`anomaly`) y una única categoría. Se usaron **24 variables**: `motor_position`, `motor_velocity`, `motor_torque` y `motor_iq` (corriente) de los motores 1 a 6. Las columnas `sample`, `time`, `anomaly` y `category` no se usan como entradas de los modelos.

### Subconjuntos

| Subconjunto | Ensayos | Registros | Ventanas V3 | Uso |
|---|---:|---:|---:|---|
| Entrenamiento | 703 | 768 468 (533 663 nominales) | 7 247 | Ajuste de modelos (solo nominales) |
| Validación | 146 | 159 493 | 1 507 | Hiperparámetros, percentil y umbral |
| Prueba exploratoria | 146 (41 anómalos) | 159 236 | 1 497 | Evaluación consultada durante el desarrollo |
| Reservados | 1 127 (460 anómalos) | – | 11 664 | Evaluación final con modelos congelados |

La selección y partición de los 995 ensayos de desarrollo la hace `preparar_voraus.py` con semilla 42 (hasta 30 ensayos por categoría de anomalía y hasta 700 nominales). Ningún ensayo aparece en dos particiones.

### Consideraciones

- Las ventanas de un ensayo anómalo heredan su etiqueta aunque no todas contengan la anomalía.
- En el subconjunto de desarrollo hay 75 valores de `setting` y **ninguno aparece en ambas clases**. Esta asociación completa con la etiqueta es un posible factor de confusión.
- Las puntuaciones de los modelos cambian según la acción del robot (`action`), lo que dificulta usar un umbral único por registro.

---

[Siguiente: Descripción de los pasos realizados](04-Metodologia-y-Resultados.md)
