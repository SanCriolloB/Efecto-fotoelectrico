# Efecto Fotoeléctrico — EDA y Modelado (Clasificación)

Análisis del efecto fotoeléctrico con énfasis en
**EDA** y **clasificación del filtro óptico (nm)** a partir de **Voltaje** y **Corriente (nA)**.
Incluye **preprocesamiento**, **normalización (MinMax)**, **balanceo de clases** y comparación de
modelos (**Regresión Logística**, **Árbol de Decisión**, **KNN**, **SVM**, **Naive-Bayes**),
con **validación cruzada estratificada (5-Fold)**. El flujo completo está en el cuaderno principal.

> **Cuaderno principal:** `notebook/Mineria_EFE.ipynb`
>
> **Datos del experimento:** `data/datos.xls` (este archivo es el origen de los datos usados en el cuaderno).

## Autores

Santiago Criollo Bermúdez & Daniel Felipe Ramirez Cabrera

---

## Estructura del repositorio

```
.
├─ notebook/
│  └─ Mineria_EFE.ipynb          # EDA + preprocesamiento + modelos + evaluación
└─ data/
   └─ datos.xls                  # Archivo original del experimento (formato .xls)
```

---

## Requisitos

Dependencias principales empleadas en el cuaderno:

- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`
- `imbalanced-learn` (oversampling/undersampling)
- `xlrd` (lectura de archivos **.xls**)

Instalación rápida (opcional, para correr localmente):

```bash
python -m venv .venv
source .venv/bin/activate   # En Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xlrd
```

---

## Cómo ejecutar

1. (Opcional) Crea y activa el entorno virtual e instala los requisitos (ver arriba).
2. Verifica que `data/datos.xls` esté presente.
3. Abre `notebook/Parcial2_Mineria.ipynb` y ejecuta las celdas en orden.
   - Si usas Google Colab, ajusta la celda de lectura a la ruta correspondiente a tu Drive.
   - Si trabajas localmente, asegúrate de usar rutas relativas (`data/datos.xls`).

---

## Qué hace el notebook

1. **EDA**: estadísticas descriptivas, visualización y revisión de variables
   (p. ej., Voltaje, Corriente (nA) y etiqueta de **Filtro (nm)**).
2. **Preprocesamiento**:
   - Limpieza y renombrado de columnas con espacios.
   - Conversión de la etiqueta **Filtro** a entero (ej. `"465nm"` → `465`).
   - (Opcional) Manejo de columnas auxiliares como **Intensidad** si están presentes.
3. **Normalización** con `MinMaxScaler` para variables numéricas.
4. **División** de los datos en entrenamiento y prueba (`train/test split`).
5. **Balanceo de clases** con `RandomOverSampler` y/o `RandomUnderSampler`.
6. **Modelado** y **comparación** de varios algoritmos:
   - **Regresión Logística** (distintos *solvers*)
   - **Árbol de Decisión** (criterios `gini`, `entropy`, `log_loss`)
   - **KNN** (ajuste de `n_neighbors` y algoritmo)
   - **SVM** (kernels `linear`, `poly`, `rbf`, `sigmoid`)
   - **Naive-Bayes Gaussiano**
7. **Evaluación**:
   - **Accuracy** en train/test, `classification_report` y (opcional) matrices de confusión.
   - **Validación cruzada estratificada (5-Fold)** para estimar robustez.

> **Objetivo:** predecir el **Filtro (nm)** a partir de **Voltaje** y **Corriente (nA)**.
Resultados de carácter exploratorio; para mayor fidelidad física se sugiere ampliar
características y mediciones del fenómeno (p. ej., estimación de función trabajo,
curvas I–V por longitud de onda, etc.).

---

## Resultados esperados

- Métricas comparativas entre modelos (accuracy, reporte de clasificación).
- Efecto del **balanceo** sobre el desempeño.
- Promedios de **validación cruzada** (Stratified 5-Fold).

---


