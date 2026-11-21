"""Validación automática de la entrega (solo lectura; usa solo la biblioteca estándar).

Uso, desde la raíz del repositorio:
    python scripts/validar_repositorio.py            # comprobaciones locales
    python scripts/validar_repositorio.py --en-linea # además, URL públicas sin autenticación

Comprueba: archivos obligatorios, integridad de los CSV, construcción de la base
SQLite con los scripts SQL, consultas SQL del notebook, notebook definitivo
(estructura, salidas y métricas), figuras, enlaces internos, referencias a la
Wiki, concordancia de métricas en la documentación, licencias, ausencia de
secretos (archivos y historial Git) y coherencia de los archivos publicados.

Cada comprobación termina como CORRECTA, FALLO o NO EJECUTADA. El script nunca
modifica archivos. Código de salida 0 solo si no hay ningún FALLO.
"""

import argparse
import base64
import csv
import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
REPO = "Enrike123/Analisis-Telemetria-Robotica-IA-UTPL"
WIKI = f"https://github.com/{REPO}/wiki"
NB = RAIZ / "notebooks" / "Unitree_Go1_Trabajo_Academico_FINAL.ipynb"
CSV_INI = RAIZ / "datos" / "unitree_go1" / "unitree_go1_telemetria_30000.csv"
CSV_FIN = RAIZ / "datos" / "unitree_go1" / "unitree_go1_dataset_integrado.csv"
SQL_CREAR = RAIZ / "sql" / "crear_base_datos.sql"
SQL_INSERT = RAIZ / "sql" / "insertar_datos.sql"

# Valores de referencia verificados (ejecución definitiva del notebook)
SHA_REFERENCIA = {
    CSV_INI: "a443e0370503b348ead3e6cc4bbadd08d910cea76c4768f5169a81a236f034e0",
    CSV_FIN: "d5ea866e773174baa9407bc3f35d24f4e1ba9acc4729fdd2803efbef77134bd5",
    NB: "b44d268915f62dda162cbc08f0e5d8580ed1fa49a2d263214e684bd333b18fed",
}
METRICAS_V3 = {   # 1 127 ensayos reservados de voraus-AD
    "Isolation Forest V3": {"Precision": 0.6427, "Recall": 0.5239, "F1": 0.5772, "ROC-AUC": 0.7035,
                            "AP": 0.6185, "FPR": 0.2009, "VN": 533, "FP": 134, "FN": 219, "VP": 241},
    "Autoencoder V3": {"Precision": 0.7572, "Recall": 0.6304, "F1": 0.6880, "ROC-AUC": 0.8372,
                       "AP": 0.8050, "FPR": 0.1394, "VN": 574, "FP": 93, "FN": 170, "VP": 290},
}
CAT_MOTORES = {i: (p, a) for i, (p, a) in enumerate(
    [(p, a) for p in ("Delantera derecha", "Delantera izquierda", "Trasera derecha", "Trasera izquierda")
     for a in ("Cadera lateral", "Muslo", "Rodilla")])}
OBLIGATORIOS = [
    "README.md", "ATRIBUCION_Y_LICENCIAS.md", "requirements.txt", ".gitignore",
    "notebooks/Unitree_Go1_Trabajo_Academico_FINAL.ipynb",
    "datos/README.md", "datos/unitree_go1/unitree_go1_telemetria_30000.csv",
    "datos/unitree_go1/unitree_go1_dataset_integrado.csv", "datos/voraus_ad/particiones_ensayos.csv",
    "sql/crear_base_datos.sql", "sql/insertar_datos.sql",
    "scripts/unitree_go1_trabajo_academico_final.py", "scripts/preparar_voraus.py",
    "resultados/README.md", "resultados/metricas/metricas_holdout_v3.csv",
    "resultados/metricas/configuracion_v3.json",
] + [f"docs/wiki/{p}.md" for p in ("Home", "01-Introduccion", "02-Marco-Teorico", "03-Descripcion-Datasets",
                                    "04-Metodologia-y-Resultados", "05-Conclusiones", "06-Bibliografia")]

resultados = []


def registrar(nombre, estado, detalle=""):
    """estado: True (correcta), False (fallo) o None (no ejecutada)."""
    etiqueta = {True: "CORRECTA", False: "FALLO", None: "NO EJECUTADA"}[estado]
    resultados.append((nombre, etiqueta, detalle))
    print(f"{etiqueta:12s} {nombre}" + (f" | {detalle}" if detalle else ""))


def sha256(ruta):
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def leer_csv(ruta):
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def archivos_versionados():
    try:
        salida = subprocess.run(["git", "-C", str(RAIZ), "ls-files", "-z"], capture_output=True, check=True)
        return [RAIZ / x for x in salida.stdout.decode("utf-8").split("\0") if x]
    except (OSError, subprocess.CalledProcessError):
        return [p for p in RAIZ.rglob("*") if p.is_file() and ".git" not in p.parts]


# 1. Archivos obligatorios -----------------------------------------------------
faltan = [r for r in OBLIGATORIOS if not (RAIZ / r).is_file()]
registrar("Archivos obligatorios presentes", not faltan, f"{len(OBLIGATORIOS)} exigidos; faltan: {faltan}")
for ruta, esperado in SHA_REFERENCIA.items():
    if ruta.is_file():
        registrar(f"SHA-256 de {ruta.relative_to(RAIZ).as_posix()} igual al verificado",
                  sha256(ruta) == esperado, sha256(ruta)[:16])

# 2. Integridad de los CSV -------------------------------------------------------
ini, fin = leer_csv(CSV_INI), leer_csv(CSV_FIN)
cols_ini = ["instante", "tiempo_ros", "motor_id", "posicion_rad", "velocidad_rad_s", "torque_Nm",
            "temperatura_C", "tiempo_relativo_s", "sesion"]
registrar("Dataset inicial: 30 000 filas y 9 columnas esperadas",
          len(ini) == 30000 and list(ini[0]) == cols_ini, f"{len(ini)} filas, {len(ini[0])} columnas")
registrar("Dataset final: 30 000 filas y 11 columnas (9 + pata, articulacion)",
          len(fin) == 30000 and list(fin[0]) == cols_ini + ["pata", "articulacion"],
          f"{len(fin)} filas, {len(fin[0])} columnas")
motores = {int(r["motor_id"]) for r in ini}
instantes = {int(r["instante"]) for r in ini}
claves = {(r["sesion"], r["instante"], r["motor_id"]) for r in ini}
vacios = sum(1 for r in ini for v in r.values() if v == "") + sum(1 for r in fin for v in r.values() if v == "")
registrar("12 motores (0–11), 2 500 instantes, 12 mediciones por instante, claves únicas y sin vacíos",
          motores == set(range(12)) and len(instantes) == 2500 and len(claves) == 30000 and vacios == 0
          and {r["sesion"] for r in ini} == {"running"},
          f"motores={len(motores)} instantes={len(instantes)} claves únicas={len(claves)} vacíos={vacios}")
dif_max, ok_corr = 0.0, True
for a, b in zip(ini, fin):
    for c in ("instante", "motor_id", "temperatura_C", "sesion", "torque_Nm"):
        ok_corr &= a[c] == b[c] or float(a[c]) == float(b[c])
    for c in ("tiempo_ros", "posicion_rad", "velocidad_rad_s", "tiempo_relativo_s"):
        x, y = float(a[c]), float(b[c])
        dif_max = max(dif_max, abs(x - y) / max(abs(x), 1e-300))
    ok_corr &= CAT_MOTORES[int(b["motor_id"])] == (b["pata"], b["articulacion"])
# El notebook relee el CSV con el lector rápido de Pandas (sección 9) antes de escribir el
# dataset final; eso introduce diferencias relativas de hasta ~1e-12 en los reales.
registrar("Dataset final = dataset inicial + catálogo de motores (fila a fila; reales ≤ 1e-12 relativo)",
          ok_corr and dif_max <= 1e-12, f"diferencia relativa máxima: {dif_max:.1e}")

# 3. SQL y SQLite ---------------------------------------------------------------
con = sqlite3.connect(":memory:")
con.execute("PRAGMA foreign_keys = ON")
try:
    con.executescript(SQL_CREAR.read_text(encoding="utf-8"))
    con.executescript(SQL_INSERT.read_text(encoding="utf-8"))
    q = lambda s: con.execute(s).fetchall()
    tablas = {t for (t,) in q("SELECT name FROM sqlite_master WHERE type='table'")}
    registrar("SQL de creación e inserción compatibles: tablas motores y mediciones",
              {"motores", "mediciones"} <= tablas, str(sorted(tablas)))
    n_mot, n_med = q("SELECT COUNT(*) FROM motores")[0][0], q("SELECT COUNT(*) FROM mediciones")[0][0]
    registrar("Inserción: 12 motores y 30 000 mediciones", (n_mot, n_med) == (12, 30000), f"{n_mot} / {n_med}")
    registrar("PRAGMA integrity_check = ok y foreign_key_check sin filas",
              q("PRAGMA integrity_check") == [("ok",)] and q("PRAGMA foreign_key_check") == [])
    dup = q("SELECT COUNT(*) FROM (SELECT 1 FROM mediciones GROUP BY sesion, instante, motor_id HAVING COUNT(*) > 1)")[0][0]
    pk = q("SELECT COUNT(DISTINCT medicion_id), MIN(medicion_id), MAX(medicion_id) FROM mediciones")[0]
    registrar("Claves primarias únicas y sin duplicados de (sesion, instante, motor_id)",
              dup == 0 and pk == (30000, 1, 30000), f"duplicados={dup}, medicion_id={pk}")
    cat_sql = {m: (p, a) for m, p, a in q("SELECT motor_id, pata, articulacion FROM motores")}
    registrar("Catálogo de motores en SQLite igual al del dataset final", cat_sql == CAT_MOTORES)
    filas = q("SELECT instante, motor_id, temperatura_C, torque_Nm, posicion_rad, velocidad_rad_s "
              "FROM mediciones ORDER BY medicion_id")
    rel = max(abs(f[4] - float(r["posicion_rad"])) / max(abs(float(r["posicion_rad"])), 1e-300)
              for f, r in zip(filas, ini))
    exactos = all((f[0], f[1], f[2], f[3]) == (int(r["instante"]), int(r["motor_id"]), int(r["temperatura_C"]),
                                               float(r["torque_Nm"])) for f, r in zip(filas, ini))
    registrar("Mediciones del SQL = dataset inicial (enteros y torque exactos; reales ≤ 1e-15 relativo)",
              exactos and rel <= 1e-15, f"diferencia relativa máxima: {rel:.1e}")
    torque = q("SELECT m.articulacion, ROUND(AVG(ABS(d.torque_Nm)), 3) FROM mediciones d "
               "JOIN motores m ON d.motor_id = m.motor_id GROUP BY m.articulacion")
    rodillas = q("SELECT d.motor_id, ROUND(AVG(ABS(d.torque_Nm)), 3) FROM mediciones d JOIN motores m "
                 "ON d.motor_id = m.motor_id WHERE m.articulacion = 'Rodilla' GROUP BY d.motor_id")
    registrar("Consultas SQL del notebook reproducidas (torque promedio por articulación y por rodilla)",
              dict(torque) == {"Rodilla": 7.485, "Muslo": 3.526, "Cadera lateral": 2.019}
              and dict(rodillas) == {2: 7.293, 5: 7.145, 8: 8.256, 11: 7.247}, f"{dict(torque)} {dict(rodillas)}")
except sqlite3.Error as e:
    registrar("Construcción de la base SQLite con los scripts SQL", False, str(e))

# 4. Notebook definitivo ------------------------------------------------------
nb = json.loads(NB.read_text(encoding="utf-8"))
celdas = nb["cells"]
codigo = [c for c in celdas if c["cell_type"] == "code"]
salidas = sum(len(c.get("outputs", [])) for c in codigo)
errores = sum(o["output_type"] == "error" for c in codigo for o in c.get("outputs", []))
registrar("Notebook: 144 celdas, 83 de código, 184 salidas, ejecución 1→83 sin errores",
          (len(celdas), len(codigo), salidas, errores) == (144, 83, 184, 0)
          and [c.get("execution_count") for c in codigo] == list(range(1, 84)),
          f"{len(celdas)}/{len(codigo)}/{salidas}, errores={errores}")


def texto_salida(i):
    return "".join("".join(o.get("text", "")) + "".join(o.get("data", {}).get("text/plain", ""))
                   for o in celdas[i].get("outputs", []))


go1 = texto_salida(54) + texto_salida(67) + texto_salida(69) + texto_salida(70)
registrar("Go1: 9 000 en prueba, Isolation Forest 134 atípicas, Autoencoder 297, una coincidencia",
          all(s in go1 for s in ("Prueba: 9000", "Mediciones atípicas: 134", "Mediciones atípicas: 297",
                                 "Anomalías detectadas por ambos: 1")))
hold = texto_salida(135)
ok_nb = "Ensayos: 1127" in hold
for modelo, m in METRICAS_V3.items():
    linea = next((l for l in hold.splitlines() if modelo in l), "")
    ok_nb &= all(f"{m[k]:.4f}" in linea for k in ("Precision", "Recall", "F1", "ROC-AUC", "AP", "FPR"))
registrar("voraus-AD: salida de la sección 12.19 con 1 127 ensayos y métricas V3 definitivas", ok_nb)
with open(RAIZ / "resultados" / "metricas" / "metricas_holdout_v3.csv", encoding="utf-8") as f:
    hold_csv = {r["Modelo"]: r for r in csv.DictReader(f)}
ok_csv = all(abs(float(hold_csv[mod][k]) - v) < 5e-5 if isinstance(v, float) else int(hold_csv[mod][k]) == v
             for mod, m in METRICAS_V3.items() for k, v in m.items())
registrar("metricas_holdout_v3.csv coincide con las métricas definitivas", ok_csv)
imagenes_nb = {hashlib.sha256(base64.b64decode(o["data"]["image/png"])).hexdigest()
               for c in codigo for o in c.get("outputs", []) if "image/png" in o.get("data", {})}
figuras = sorted((RAIZ / "resultados" / "figuras").glob("*.png"))
ajenas = [p.name for p in figuras if sha256(p) not in imagenes_nb]
registrar("Las 24 figuras son salidas de la ejecución definitiva del notebook",
          len(figuras) == 24 and not ajenas, f"{len(figuras)} figuras; ajenas: {ajenas}")

# 5. Documentación: métricas, enlaces y Wiki ------------------------------------
docs = {p.relative_to(RAIZ).as_posix(): p.read_text(encoding="utf-8") for p in RAIZ.rglob("*.md")
        if ".git" not in p.parts}
claves_v3 = ["0,5772", "0,6880", "0,7035", "0,8372", "0,7572", "0,6304", "574 / 93 / 170 / 290", "533 / 134 / 219 / 241"]
falta_v3 = [k for k in claves_v3 if k not in docs["README.md"]]
registrar("README contiene las métricas y matrices V3 definitivas", not falta_v3, str(falta_v3))
anteriores = [(n, l.strip()[:80]) for n, t in docs.items() for l in t.splitlines()
              if re.search(r"0,6949|0,7584|0,8380|0,694935", l) and "anterior" not in l.lower()]
registrar("Cifras de la ejecución anterior solo como 'ejecución anterior'", not anteriores, str(anteriores[:3]))
rotos = []
for nombre, t in docs.items():
    base = (RAIZ / nombre).parent
    for destino in re.findall(r"\]\(([^)\s]+)\)", t):
        if destino.startswith(("http", "#", "mailto:")):
            continue
        if not (base / destino.split("#")[0]).exists():
            rotos.append((nombre, destino))
registrar("Enlaces relativos de todos los Markdown resuelven", not rotos, str(rotos[:5]))
paginas = ["", "/01-Introduccion", "/02-Marco-Teorico", "/03-Descripcion-Datasets",
           "/04-Metodologia-y-Resultados", "/05-Conclusiones", "/06-Bibliografia"]
registrar("README enlaza las 7 páginas de la Wiki", all(f"({WIKI}{p})" in docs["README.md"] for p in paginas))

# 6. Licencias ---------------------------------------------------------------
lic = docs["README.md"] + docs["ATRIBUCION_Y_LICENCIAS.md"] + docs["datos/README.md"]
registrar("Licencias documentadas: CC BY 4.0 (Leg-KILO) con commit bbdb842 y CC BY-NC-SA 4.0 (voraus-AD)",
          all(s in lic for s in ("creativecommons.org/licenses/by/4.0/", "bbdb84250876bebfaa1c6b44e0625738d4f37f07",
                                 "creativecommons.org/licenses/by-nc-sa/4.0/", "10.1109/LRA.2024.3440730",
                                 "10.1109/TRO.2023.3332224")))
obsoletas = [(n, l.strip()[:80]) for n, t in docs.items() for l in t.splitlines()
             if re.search(r"no declara|sin licencia declarada|no está autorizada|pendiente de (su )?autorización|"
                          r"No incluido físicamente", l, re.I)]
registrar("Sin declaraciones obsoletas sobre la falta de licencia de Leg-KILO", not obsoletas, str(obsoletas[:3]))

# 7. Seguridad y coherencia de los archivos publicados ------------------------
versionados = archivos_versionados()
SECRETO = re.compile(rb"(gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|"
                     rb"AIza[0-9A-Za-z_\-]{35}|sk-[A-Za-z0-9]{32,}|xox[baprs]-[A-Za-z0-9-]{10,}|"
                     rb"-----BEGIN [A-Z ]*PRIVATE KEY-----)")
nombres_sensibles = [p.name for p in versionados
                     if re.search(r"recovery|\.env$|\.pem$|\.key$|id_rsa|credential", p.name, re.I)]
registrar("Ningún archivo sensible versionado (recovery codes, .env, claves)", not nombres_sensibles,
          str(nombres_sensibles))
con_secreto = []
for p in versionados:
    datos = p.read_bytes()
    if p.suffix == ".ipynb":   # las imágenes base64 pueden contener secuencias aleatorias
        datos = re.sub(rb'"image/png": "[^"]*"', b"", datos)
    if SECRETO.search(datos):
        con_secreto.append(p.relative_to(RAIZ).as_posix())
registrar("Ningún token, clave privada o credencial en los archivos versionados", not con_secreto, str(con_secreto))
try:
    historial = subprocess.run(["git", "-C", str(RAIZ), "log", "--all", "-p", "--no-color"],
                               capture_output=True, check=True).stdout
    historial = re.sub(rb'"image/png": "[^"]*"', b"", historial)
    nombres_hist = subprocess.run(["git", "-C", str(RAIZ), "log", "--all", "--name-only", "--format="],
                                  capture_output=True, check=True).stdout.decode("utf-8", "ignore")
    registrar("Historial Git sin secretos ni archivos de recuperación",
              not SECRETO.search(historial) and not re.search(r"recovery", nombres_hist, re.I))
except (OSError, subprocess.CalledProcessError):
    registrar("Historial Git sin secretos ni archivos de recuperación", None, "git no disponible")
prohibidos = [p.relative_to(RAIZ).as_posix() for p in versionados
              if p.suffix in {".db", ".sqlite", ".sqlite3", ".bag", ".parquet", ".keras", ".joblib", ".pkl", ".zip"}]
grandes = [p.relative_to(RAIZ).as_posix() for p in versionados if p.stat().st_size > 50 * 2**20]
registrar("Sin binarios pesados ni modelos versionados; ningún archivo supera 50 MB",
          not prohibidos and not grandes, f"{prohibidos} {grandes}")

# 8. URL públicas (opcional) ---------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--en-linea", action="store_true", help="comprobar las URL públicas sin autenticación")
args = parser.parse_args()
if args.en_linea:
    def estado(url):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code
        except OSError as e:
            return str(e)[:40]
    raw = f"https://raw.githubusercontent.com/{REPO}/main/"
    urls = [f"https://github.com/{REPO}",
            "https://colab.research.google.com/github/" + REPO + "/blob/main/notebooks/Unitree_Go1_Trabajo_Academico_FINAL.ipynb"]
    urls += [raw + r for r in OBLIGATORIOS if not r.startswith(".")]
    urls += [raw + "resultados/figuras/" + p.name for p in figuras]
    urls += [WIKI + p for p in paginas]
    malos = [(u, s) for u in urls if (s := estado(u)) != 200]
    registrar("URL públicas responden 200 sin autenticación", not malos, f"{len(urls)} URL; con error: {malos[:5]}")
else:
    registrar("URL públicas responden 200 sin autenticación", None, "usar --en-linea")

total = {e: sum(1 for _, x, _ in resultados if x == e) for e in ("CORRECTA", "FALLO", "NO EJECUTADA")}
print(f"\nResumen: {total['CORRECTA']} correctas, {total['FALLO']} fallos, {total['NO EJECUTADA']} no ejecutadas")
sys.exit(1 if total["FALLO"] else 0)
