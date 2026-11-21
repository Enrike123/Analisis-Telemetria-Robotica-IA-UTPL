# Datos: obtención y reconstrucción

El trabajo usa dos fuentes robóticas independientes: **Leg-KILO** (cuadrúpedo Unitree Go1) y **voraus-AD** (manipulador de seis ejes). Los archivos originales (`running.bag` y el Parquet de voraus-AD) no se incluyen por su tamaño. Sí se incluyen el **dataset inicial** y el **dataset final** del Go1, derivados de Leg-KILO bajo CC BY 4.0. Este documento explica la procedencia de cada archivo, sus licencias y cómo reconstruirlos.

## 1. Unitree Go1 (Leg-KILO)

### 1.1. Fuente original

- Repositorio del dataset: https://github.com/ouguangjun/legkilo-dataset (las descargas en Google Drive y Baidu se enlazan desde su README).
- Secuencia utilizada: `running` (archivo `running.bag`, ROS 1, unos 216 MB). Según sus autores, el robot recorre un circuito corto al aire libre a una velocidad media de 1,5 m/s.
- Tópico utilizado: `/high_state` (`unitree_legged_msgs/HighState`), primeros 12 elementos de `motorState`: `q` (rad), `dq` (rad/s), `tauEst` (N·m) y `temperature` (°C).
- Referencia: Ou, G., Li, D., & Li, H. (2024). *Leg-KILO: Robust Kinematic-Inertial-Lidar Odometry for Dynamic Legged Robots*. IEEE Robotics and Automation Letters, 9(10), 8194–8201. https://doi.org/10.1109/LRA.2024.3440730
- Licencia: las grabaciones rosbag y los datos de referencia (ground truth) están bajo **[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)**, según la sección "Dataset License" del README del dataset ([commit `bbdb842`](https://github.com/ouguangjun/legkilo-dataset/commit/bbdb84250876bebfaa1c6b44e0625738d4f37f07), 26/10/2026). La licencia no cubre el código ni los SDK de sus repositorios; el código de Leg-KILO tiene licencia MIT.

### 1.2. Archivos derivados

| Archivo | Contenido | Generado en | En el repositorio |
|---|---|---|---|
| [`unitree_go1/unitree_go1_telemetria_30000.csv`](unitree_go1/unitree_go1_telemetria_30000.csv) | **Dataset inicial**: 30 000 mediciones (2 500 instantes × 12 motores, 9 columnas) | Sección 4 del notebook (modo ROS) | Sí, CC BY 4.0 |
| [`../sql/crear_base_datos.sql`](../sql/crear_base_datos.sql) | Estructura de las tablas `motores` y `mediciones` (sin registros) | Sección 5 | Sí |
| [`../sql/insertar_datos.sql`](../sql/insertar_datos.sql) | 12 motores y 30 000 mediciones como sentencias `INSERT` | Sección 5 | Sí, CC BY 4.0 |
| `unitree_go1_generada.db` | Base SQLite construida con los dos scripts anteriores | Sección 6 | No (se reconstruye) |
| [`unitree_go1/unitree_go1_dataset_integrado.csv`](unitree_go1/unitree_go1_dataset_integrado.csv) | **Dataset final**: mediciones más `pata` y `articulacion` (30 000 × 11) | Sección 9 | Sí, CC BY 4.0 |

Columnas del dataset inicial: `instante`, `tiempo_ros` (s), `motor_id` (0–11), `posicion_rad` (rad), `velocidad_rad_s` (rad/s), `torque_Nm` (N·m, estimado por el robot), `temperatura_C` (°C), `tiempo_relativo_s` (s) y `sesion` (`running`). El dataset final añade `pata` y `articulacion`.

**Atribución requerida al reutilizar estos archivos:** citar a Ou, G., Li, D., & Li, H. (2024), enlazar https://creativecommons.org/licenses/by/4.0/ e indicar que son datos derivados (extracción, muestreo, conversión e integración descritos en [`../ATRIBUCION_Y_LICENCIAS.md`](../ATRIBUCION_Y_LICENCIAS.md)).

### 1.3. Reconstrucción de la base de datos

La base SQLite no se incluye como binario porque se reconstruye con los dos scripts:

```bash
sqlite3 unitree_go1.db < sql/crear_base_datos.sql
sqlite3 unitree_go1.db < sql/insertar_datos.sql
```

Resultado verificado: 12 motores, 30 000 mediciones, `PRAGMA integrity_check` = `ok` y `PRAGMA foreign_key_check` sin filas.

### 1.4. Reconstrucción desde el archivo original

1. Descargar `running.bag` desde la fuente oficial.
2. Abrir el notebook en Google Colab y fijar `MODO_EJECUCION = "ROS"` en la sección 3.
3. Ejecutar las secciones 3 y 4. El notebook instala `rosbags`, solicita `running.bag`, extrae el tópico `/high_state`, elige 2 500 instantes equiespaciados con `np.linspace` y guarda `unitree_go1_telemetria_30000.csv`.
4. Ejecutar las secciones 5 a 9 para generar los scripts SQL, la base SQLite y el dataset integrado.

Comprobaciones que hace el notebook: 30 000 filas, 12 motores, 2 500 instantes, ningún valor nulo, integridad de SQLite correcta y ninguna clave foránea huérfana.

Para comparar la reconstrucción con los archivos usados en el trabajo:

| Archivo | SHA-256 de referencia |
|---|---|
| `running.bag` | `1a1b0793e5b654d5635381bdc35580283b263d6702c6a273ddf8e102b02538b8` |
| `unitree_go1_telemetria_30000.csv` | `a443e0370503b348ead3e6cc4bbadd08d910cea76c4768f5169a81a236f034e0` |
| `unitree_go1_dataset_integrado.csv` | `d5ea866e773174baa9407bc3f35d24f4e1ba9acc4729fdd2803efbef77134bd5` |

`insertar_datos.sql` puede diferir byte a byte entre entornos aunque contenga los mismos valores, porque la forma en que SQLite escribe los números reales con `iterdump()` depende de su versión. Conviene compararlo cargando los datos, no por su hash.

**Precisión numérica de los archivos publicados** (verificado el 21/11/2026 leyendo los decimales con el lector exacto de Python):

- Los enteros, los textos, `torque_Nm` y el orden de las filas son idénticos en los dos CSV y en `insertar_datos.sql`.
- `insertar_datos.sql` reproduce los reales del dataset inicial con una diferencia relativa máxima de 2,0 × 10⁻¹⁶ (la última cifra binaria).
- El dataset final difiere del inicial en los reales como máximo en 8,4 × 10⁻¹³ (velocidad), 3,1 × 10⁻¹³ (posición) y 1,9 × 10⁻¹⁵ (tiempo relativo), en relativo. Se debe a que el notebook vuelve a leer el CSV con el lector rápido de Pandas en la sección 9 antes de escribir el dataset final. Es la salida original del notebook y no afecta a ningún resultado: las consultas y los modelos dan las mismas cifras.

## 2. voraus-AD

### 2.1. Fuente original

- Repositorio: https://github.com/vorausrobotik/voraus-ad-dataset
- Archivo utilizado: `voraus-ad-dataset-100hz.parquet` (unos 1,1 GB; 2 321 690 registros, 137 columnas, 2 122 ensayos), descargado desde el enlace oficial de ese repositorio.
- SHA-256 del archivo usado: `c90ab1c78af52651b954d41787f7e89d750f0a128b57600b0e5ceec22621f704`
- Referencia: Brockmann, J. T., Rudolph, M., Rosenhahn, B., & Wandt, B. (2024). *The voraus-AD Dataset for Anomaly Detection in Robot Applications*. IEEE Transactions on Robotics, 40, 438–451. https://doi.org/10.1109/TRO.2023.3332224
- Licencia del dataset: **CC BY-NC-SA 4.0** (https://creativecommons.org/licenses/by-nc-sa/4.0/).

### 2.2. Archivos derivados

| Archivo | Incluido | Motivo |
|---|---|---|
| `voraus_ad/particiones_ensayos.csv` | Sí (20 KB) | Lista de los 995 ensayos y su partición. Derivado de voraus-AD, CC BY-NC-SA 4.0 |
| `voraus_entrenamiento.csv`, `voraus_validacion.csv`, `voraus_prueba.csv` | No | Unos 520 MB; se reconstruyen con el script |
| `voraus_v3_*.csv` (características temporales) | No | Las genera el notebook (sección 12.14) |
| Modelos `*.keras` y `*.joblib` | No | Los genera el notebook (sección 12.20). Los `.joblib` solo deben cargarse si proceden de una fuente de confianza |

### 2.3. Reconstrucción

Los scripts de `scripts/` buscan el Parquet en `dataset_voraus_AD/`, **en la misma carpeta donde está el script**. Para usarlos sin modificarlos:

```text
carpeta_de_trabajo/
├── preparar_voraus.py              (copiado desde scripts/)
└── dataset_voraus_AD/
    └── voraus-ad-dataset-100hz.parquet
```

```bash
python preparar_voraus.py
```

El script elige con semilla 42 hasta 30 ensayos por categoría de anomalía y hasta 700 nominales, y los reparte (70/15/15 por categoría) en `dataset_voraus_AD/procesado/`. **Borra y vuelve a crear** los tres CSV de esa carpeta. El resultado debe coincidir con `datos/voraus_ad/particiones_ensayos.csv`; esta comprobación se hizo el 20/10/2026 y la selección de 995 ensayos fue idéntica.

Los scripts `inspeccionar_voraus.py` y `verificar_etiquetas_voraus.py` son exploratorios: muestran el esquema, las etiquetas, las categorías y las variables `setting`, `action` y `active`, y guardan dos resúmenes CSV.

Después, copiar `dataset_voraus_AD/` a Google Drive y ajustar `CARPETA_TRABAJO` en la sección 12.1 del notebook.
