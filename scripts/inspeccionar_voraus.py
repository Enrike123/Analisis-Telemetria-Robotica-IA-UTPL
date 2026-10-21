
from pathlib import Path
import pyarrow.parquet as pq

# Archivo original
RUTA = (
    Path(__file__).resolve().parent
    / "dataset_voraus_AD"
    / "voraus-ad-dataset-100hz.parquet"
)

if not RUTA.exists():
    raise FileNotFoundError(f"No se encontró: {RUTA}")

# Leer metadatos sin cargar todo el dataset
archivo = pq.ParquetFile(RUTA)

print("=== DATASET VORAUS-AD ===")
print("Filas:", archivo.metadata.num_rows)
print("Columnas:", archivo.metadata.num_columns)
print("Grupos de filas:", archivo.metadata.num_row_groups)

print("\n=== COLUMNAS Y TIPOS ===")
for campo in archivo.schema_arrow:
    print(f"{campo.name}: {campo.type}")

# Leer solamente el primer grupo de filas
tabla = archivo.read_row_group(0)
df = tabla.slice(0, 5).to_pandas()

print("\n=== PRIMERAS 5 FILAS ===")
print(df.to_string())

print("\n=== COLUMNAS DE IDENTIFICACIÓN ===")
for columna in df.columns:
    if any(
        termino in columna.lower()
        for termino in [
            "anomaly", "category", "trial",
            "run", "experiment", "label", "task"
        ]
    ):
        print(columna)
