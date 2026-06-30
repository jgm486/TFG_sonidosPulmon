# Clasificación automática de patologías pulmonares a partir de sonidos estetoscópicos

Repositorio del código del **Trabajo Fin de Grado** de
**José Miguel García Martín**, Grado en Ingeniería Informática,
Universidad de Almería, curso 2025/2026.

Tutoras: Gracia Ester Martín Garzón, Gloria Ortega López.

---

## Descripción

Sistema de clasificación multiclase de seis enfermedades respiratorias
(normal, asma, neumonía, insuficiencia cardíaca, bronquiectasia/bronquitis
y EPOC) a partir de sonidos pulmonares grabados con estetoscopio digital,
empleando técnicas de *machine learning* clásico.

El análisis se basa en dos bases de datos públicas:

- **ICBHI 2017 Challenge Database** (Rocha et al., 2019)
- **Fraiwan et al. Dataset** (Mendeley Data, 2021)

El análisis principal se realiza sobre un **subconjunto balanceado de 60
pacientes** (10 por clase) construido a partir de la fusión de ambas
bases.

---

## Aportaciones metodológicas principales

- **Validación cruzada por paciente con *folds* fijos**: cada paciente
  aparece íntegramente en una única partición, lo que elimina la fuga de
  información paciente-grabación habitual en la literatura.
- **Comparación de cinco familias de modelos clásicos** bajo condiciones
  experimentales idénticas: SVM (núcleo RBF), Random Forest, regresión
  logística, KNN y XGBoost.
- **Uso exclusivo de bases de datos públicas**, lo que permite la
  replicación independiente de los experimentos.
- **Reproducibilidad**: fijación de semillas, *folds* deterministas y
  documentación de versiones de bibliotecas.

---

## Resultados

Métricas one-vs-rest macro sobre los 60 pacientes:

| Modelo | Accuracy | Sensibilidad | Especificidad | F1 | Kappa |
|--------|---------:|-------------:|--------------:|----:|------:|
| **SVM (RBF)**        | **87,78%** | **63,33%** | **92,67%** | **63,21%** | **55,98%** |
| Random Forest        | 87,22%     | 61,67%     | 92,33%     | 62,28%     | 54,67%     |
| Regresión logística  | 86,67%     | 60,00%     | 92,00%     | 60,08%     | 52,17%     |
| KNN                  | 86,11%     | 58,33%     | 91,67%     | 58,82%     | 50,53%     |
| XGBoost              | 86,11%     | 58,33%     | 91,67%     | 58,69%     | 50,39%     |

La clase **EPOC se clasifica perfectamente (10/10) en los cinco modelos**.
Las confusiones más frecuentes se concentran entre clases acústicamente
próximas, particularmente entre normal y asma.

---

## Bases de datos

**Las bases de datos no se incluyen en este repositorio** por motivos de
licencia y tamaño. Para reproducir los resultados desde cero, descárgalas
desde sus fuentes originales:

- **ICBHI**: https://bhichallenge.med.auth.gr/
- **Fraiwan et al.**: https://data.mendeley.com/datasets/jwyy9np4gv/3

Una vez descargadas, colócalas en una carpeta accesible y ajusta las
rutas en los cuadernos de extracción de características.

---

## Requisitos

Python 3.10 o superior. Las dependencias están en `requirements.txt`:

```bash
pip install -r requirements.txt
```

Probado con Python 3.10, 3.11, 3.12 y 3.13.

---

## Estructura del repositorio

```
notebooks/    Cuadernos numerados con el pipeline completo del trabajo.
scripts/      Scripts auxiliares (análisis comparativos).
datos/        CSV del subconjunto balanceado con folds asignados.
resultados/   Carpeta destino para tablas, figuras y predicciones.
```

---

## Uso

Los cuadernos de `notebooks/` están numerados y se ejecutan en orden:

1. **`01_extraccion_features_icbhi.ipynb`**: extrae las 41 características
   acústicas de cada ventana de cinco segundos de la base ICBHI.
2. **`02_extraccion_features_fraiwan.ipynb`**: extrae las mismas
   características de la base Fraiwan, con asignación automática de
   diagnóstico desde el nombre del archivo.
4. **`03_subconjunto_balanceado.ipynb`**: construye el subconjunto de 60
   pacientes (10 por clase) y asigna los *folds* fijos.
5. **`04a_modelos_validacion_ventana.ipynb`**: evaluación preliminar de
   las tres configuraciones (ICBHI, Fraiwan, fusión) con validación por
   ventana (StratifiedKFold), equivalente al esquema habitual en la
   literatura.
6. **`04b_modelos_validacion_paciente.ipynb`**: misma evaluación con
   validación por paciente (StratifiedGroupKFold), que elimina la fuga
   de información ventana-paciente.
7. **`05_busqueda_hiperparametros.ipynb`**: rejillas de hiperparámetros
   para KNN, Random Forest, XGBoost y regresión logística sobre el
   subconjunto balanceado.
8. **`06_evaluacion_por_paciente.ipynb`**: evaluación final por paciente
   con votación mayoritaria sobre el subconjunto balanceado de 60
   pacientes.

Y los scripts auxiliares en `scripts/`:

- **`voto_vs_probabilidades.py`**: compara las dos estrategias de
  agregación de predicciones a nivel de paciente (votación mayoritaria
  vs promedio de probabilidades).

---

## Reproducibilidad

Todos los componentes que admiten aleatoriedad emplean `random_state=42`.
La asignación de pacientes a *folds* es determinista y se almacena
junto al subconjunto en `datos/dataset_60_topado.csv`.

---

## Licencia

Este código se distribuye bajo licencia MIT. Ver [LICENSE](LICENSE).

---

## Cita

Si usas este código en investigación, cita el TFG asociado:

> García Martín, J. M. (2026). *Clasificación automática de patologías
> pulmonares a partir de sonidos estetoscópicos utilizando Machine
> Learning Clásico*. Trabajo Fin de Grado, Universidad de Almería.
