
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent
RUTA = BASE / "dataset_voraus_AD" / "voraus-ad-dataset-100hz.parquet"

# Leer únicamente las columnas de identificación
columnas = [
    "time", "sample", "anomaly",
    "category", "setting", "action", "active"
]

tabla = pq.read_table(RUTA, columns=columnas)
df = tabla.to_pandas()

print("=== 1. DIMENSIONES ===")
print("Registros:", len(df))
print("Columnas:", len(df.columns))

print("\n=== 2. ETIQUETAS DE ANOMALÍA ===")
print(df["anomaly"].value_counts(dropna=False))

print("\n=== 3. CATEGORÍAS ===")
print(df["category"].value_counts(
    dropna=False
).sort_index())

print("\n=== 4. CATEGORÍA VS. ANOMALÍA ===")
print(pd.crosstab(
    df["category"],
    df["anomaly"],
    dropna=False
))

print("\n=== 5. IDENTIFICADORES SAMPLE ===")
print("Valores diferentes:", df["sample"].nunique())
print("Mínimo:", df["sample"].min())
print("Máximo:", df["sample"].max())

# Comprobar si un identificador contiene varias etiquetas
etiquetas_por_sample = df.groupby("sample").agg(
    registros=("anomaly", "size"),
    etiquetas_distintas=("anomaly", "nunique"),
    categorias_distintas=("category", "nunique"),
    tiempo_min=("time", "min"),
    tiempo_max=("time", "max")
)

print("\nPrimeros 10 identificadores:")
print(etiquetas_por_sample.head(10).to_string())

print("\nSamples con más de una etiqueta:")
print(int(
    (etiquetas_por_sample["etiquetas_distintas"] > 1).sum()
))

print("\n=== 6. CAMBIOS EN LA SECUENCIA ===")

# Detectar límites potenciales entre ensayos
cambio_sample = df["sample"].ne(
    df["sample"].shift()
)

reinicio_tiempo = df["time"].diff().lt(0)

print("Cambios de sample:", int(cambio_sample.sum()))
print("Reinicios de tiempo:", int(reinicio_tiempo.sum()))

# Segmentos consecutivos, sin asumir todavía
# que cada sample representa un ensayo completo
inicio_segmento = cambio_sample | reinicio_tiempo
segmento_id = inicio_segmento.cumsum()

resumen_segmentos = df.groupby(segmento_id).agg(
    sample=("sample", "first"),
    filas=("sample", "size"),
    tiempo_inicial=("time", "first"),
    tiempo_final=("time", "last"),
    categorias=("category", "nunique"),
    etiquetas=("anomaly", "nunique")
)

print("\nSegmentos consecutivos:",
      len(resumen_segmentos))

print("\nPrimeros 10 segmentos:")
print(resumen_segmentos.head(10).to_string())

print("\n=== 7. VALORES ADICIONALES ===")
for columna in ["setting", "action", "active"]:
    print(f"\n{columna}:")
    print(df[columna].value_counts(
        dropna=False
    ).head(15).to_string())

# Guardar resúmenes pequeños
salida = BASE / "dataset_voraus_AD"

etiquetas_por_sample.to_csv(
    salida / "resumen_samples.csv"
)

resumen_segmentos.to_csv(
    salida / "resumen_segmentos.csv",
    index_label="segmento_id"
)

print("\nINSPECCIÓN COMPLETADA")
