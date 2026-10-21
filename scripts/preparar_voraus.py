
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent
ORIGEN = BASE / "dataset_voraus_AD" / "voraus-ad-dataset-100hz.parquet"
DESTINO = BASE / "dataset_voraus_AD" / "procesado"
DESTINO.mkdir(parents=True, exist_ok=True)

SEMILLA = 42
MAX_ENSAYOS_ANOMALOS = 30
MAX_ENSAYOS_NOMINALES = 700
rng = np.random.default_rng(SEMILLA)

# 1. Leer solamente metadatos de ensayos
identificacion = pd.read_parquet(
    ORIGEN,
    columns=["sample", "anomaly", "category"]
)

ensayos = (
    identificacion.groupby("sample")
    .agg(
        anomaly=("anomaly", "first"),
        category=("category", "first"),
        registros=("anomaly", "size"),
        n_etiquetas=("anomaly", "nunique"),
        n_categorias=("category", "nunique")
    )
    .reset_index()
)

assert (ensayos["n_etiquetas"] == 1).all()
assert (ensayos["n_categorias"] == 1).all()

# 2. Selección y partición por categoría
particiones = []

for categoria, grupo in ensayos.groupby("category"):
    ids = grupo["sample"].to_numpy().copy()
    rng.shuffle(ids)

    if categoria == 12:
        max_ensayos = MAX_ENSAYOS_NOMINALES
    else:
        max_ensayos = MAX_ENSAYOS_ANOMALOS

    ids = ids[:max_ensayos]
    n = len(ids)

    if n < 3:
        raise ValueError(
            f"Categoría {categoria}: solo {n} ensayos"
        )

    n_prueba = max(1, round(n * 0.15))
    n_validacion = max(1, round(n * 0.15))

    for i, sample_id in enumerate(ids):
        if i < n_prueba:
            particion = "prueba"
        elif i < n_prueba + n_validacion:
            particion = "validacion"
        else:
            particion = "entrenamiento"

        particiones.append({
            "sample": int(sample_id),
            "category": int(categoria),
            "particion": particion
        })

df_particiones = pd.DataFrame(particiones)

# 3. Comprobar separación de ensayos
assert not df_particiones["sample"].duplicated().any()

print("Ensayos seleccionados:", len(df_particiones))
print(pd.crosstab(
    df_particiones["category"],
    df_particiones["particion"]
))

# 4. Seleccionar variables de los seis motores
variables = []

for motor in range(1, 7):
    variables.extend([
        f"motor_position_{motor}",
        f"motor_velocity_{motor}",
        f"motor_torque_{motor}",
        f"motor_iq_{motor}"
    ])

columnas = [
    "time", "sample", "anomaly",
    "category"
] + variables

# 5. Lectura por lotes
parquet = pq.ParquetFile(ORIGEN)

mapa = dict(zip(
    df_particiones["sample"],
    df_particiones["particion"]
))

archivos = {
    p: DESTINO / f"voraus_{p}.csv"
    for p in ["entrenamiento", "validacion", "prueba"]
}

# Eliminar salidas anteriores, no el Parquet
for ruta in archivos.values():
    if ruta.exists():
        ruta.unlink()

cabeceras = {p: False for p in archivos}

for lote in parquet.iter_batches(
    batch_size=50000,
    columns=columnas
):
    df = lote.to_pandas()

    df["particion"] = df["sample"].map(mapa)
    df = df.dropna(subset=["particion"])

    for particion, ruta in archivos.items():
        bloque = df[
            df["particion"] == particion
        ].drop(columns=["particion"])

        if bloque.empty:
            continue

        bloque.to_csv(
            ruta,
            mode="a",
            index=False,
            header=not cabeceras[particion]
        )

        cabeceras[particion] = True

# 6. Guardar registro de particiones
df_particiones.to_csv(
    DESTINO / "particiones_ensayos.csv",
    index=False
)

# 7. Validación de archivos generados
for particion, ruta in archivos.items():
    assert ruta.exists(), f"Falta {ruta}"

    df_check = pd.read_csv(
        ruta,
        usecols=["sample", "anomaly", "category"]
    )

    ids_esperados = set(
        df_particiones.loc[
            df_particiones["particion"] == particion,
            "sample"
        ]
    )

    assert set(df_check["sample"].unique()) == ids_esperados

    print(
        particion,
        "| filas:", len(df_check),
        "| ensayos:", df_check["sample"].nunique(),
        "| anomalías:", int(df_check["anomaly"].sum())
    )


print("\n=== DISTRIBUCIÓN FINAL ===")

for particion, ruta in archivos.items():
    df_check = pd.read_csv(
        ruta,
        usecols=["sample", "anomaly", "category"]
    )

    nominales = int((~df_check["anomaly"]).sum())
    anomalos = int(df_check["anomaly"].sum())
    total = len(df_check)

    print(f"\n{particion.upper()}")
    print("Registros:", total)
    print("Nominales:", nominales)
    print("Anómalos:", anomalos)
    print("Porcentaje de anomalías:",
          round(anomalos / total * 100, 2), "%")

    print("Ensayos nominales:",
          df_check.loc[
              ~df_check["anomaly"], "sample"
          ].nunique())

    print("Ensayos anómalos:",
          df_check.loc[
              df_check["anomaly"], "sample"
          ].nunique())



print("\nDATASETS PREPARADOS CORRECTAMENTE")
print("Carpeta:", DESTINO)
