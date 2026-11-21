# Análisis de telemetría robótica y detección de anomalías con inteligencia artificial

**Trabajo académico final de la asignatura Herramientas para Inteligencia Artificial**
**Programa:** Maestría en Inteligencia Artificial Aplicada, Universidad Técnica Particular de Loja (UTPL)
**Autores:** Jaime Enrique Villegas Medina ([@Enrike123](https://github.com/Enrike123)) e Ing. Jorge Luis Castro Valdivia ([@jorgelcv62-ai](https://github.com/jorgelcv62-ai))
**Afiliación profesional de ambos autores:** Universidad Católica de Santa María (UCSM), Perú
**Plataforma:** Google Colab

## Presentación

En este trabajo se procesa, almacena, visualiza y analiza telemetría de articulaciones robóticas con Python y bibliotecas de software libre. Se aplican dos modelos de detección de anomalías: Isolation Forest (Scikit-learn) y un autoencoder (TensorFlow/Keras).

Se usan dos fuentes de datos robóticas independientes:

- **Leg-KILO:** 30 000 mediciones de los 12 motores de un cuadrúpedo Unitree Go1. Se procesan como CSV, se almacenan en una base SQLite y se analizan sin etiquetas de falla.
- **voraus-AD:** datos de un manipulador de seis ejes con anomalías inducidas y etiquetadas, que permiten medir el desempeño de los modelos.

## Resultado principal

Con 144 características temporales (ventanas de 100 muestras) y evaluación sobre **1 127 ensayos reservados** de voraus-AD, en la ejecución definitiva del notebook:

| Métrica | Isolation Forest V3 | Autoencoder V3 |
|---|---:|---:|
| Precision | 0,6427 | 0,7572 |
| Recall | 0,5239 | 0,6304 |
| F1-score | 0,5772 | 0,6880 |
| ROC-AUC | 0,7035 | 0,8372 |
| Average Precision | 0,6185 | 0,8050 |
| Tasa de falsos positivos | 20,09 % | 13,94 % |

Estos valores describen la detección de anomalías inducidas en voraus-AD. No demuestran la detección de fallas reales en el Unitree Go1 ni en otros robots.

## Contenido

1. [Introducción](01-Introduccion.md)
2. [Marco teórico de las tecnologías y bibliotecas](02-Marco-Teorico.md)
3. [Descripción de los datasets](03-Descripcion-Datasets.md)
4. [Descripción de los pasos realizados y visualizaciones](04-Metodologia-y-Resultados.md)
5. [Conclusiones](05-Conclusiones.md)
6. [Bibliografía](06-Bibliografia.md)

## Repositorio

El notebook académico, los scripts, el SQL de la base de datos, las métricas y las figuras están en el repositorio asociado a esta Wiki. Su README explica cómo obtener los datos y reproducir los experimentos.

Repositorio: https://github.com/Enrike123/Analisis-Telemetria-Robotica-IA-UTPL

El dataset inicial y el dataset final del Unitree Go1 están en [`datos/unitree_go1/`](https://github.com/Enrike123/Analisis-Telemetria-Robotica-IA-UTPL/tree/main/datos/unitree_go1) (derivados de Leg-KILO, CC BY 4.0).
