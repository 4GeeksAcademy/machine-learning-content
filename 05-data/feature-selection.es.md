# Selección de características

**¿Qué es la selección de características?**

El objetivo de la selección de características es mejorar la interpretabilidad de los modelos, acelerar el proceso de aprendizaje y aumentar el rendimiento predictivo.

**¿Cuándo deberíamos reducir el número de características utilizadas por nuestro modelo?**

Algunos casos en los que es necesaria la selección de caracteristicas:

- Cuando hay una fuerte colinealidad entre las características.

- Hay una cantidad abrumadora de características.

- No hay suficiente poder computacional para procesar todas las características.

- El algoritmo obliga al modelo a usar todas las características, incluso cuando no son útiles (más a menudo en modelos paramétricos o lineales).

- Cuando queremos simplificar el modelo por cualquier motivo. Por ejemplo, para que sea más fácil de explicar, se necesita menos potencia de cálculo.

**¿Cuándo es innecesaria la selección de características?**

Algunos casos en los que la selección de características no es necesaria:

- Hay relativamente pocas características.

- Todas las características contienen señales útiles e importantes.

- No hay colinealidad entre las características.

- El modelo seleccionará automáticamente las características más útiles.

- Los recursos informáticos pueden manejar el procesamiento de todas las características.

- Explicar a fondo el modelo a una audiencia no técnica no es fundamental.

**¿Cuáles son los tres tipos de métodos de selección de características?**

- Métodos de filtrado: la selección de características se realiza independientemente del algoritmo de aprendizaje, antes de realizar cualquier modelado. Un ejemplo es encontrar la correlación entre cada característica y el objetivo y descartar aquellas que no alcanzan un umbral. Fácil, rápido, pero ingenuo y no tan eficaz como otros métodos.

    - Método básico.

    - Método de correlación.
    
    - Métodos estadísticos (Ganancia de información / Chi Square / ANOVA).

- Métodos de envoltorio: entrena modelos en subconjuntos de características y usa el subconjunto que resulte en el mejor rendimiento. Algunos ejemplos son la selección de características por Pasos o Recursiva. Las ventajas son que considera cada característica en el contexto de las otras características, pero puede ser computacionalmente costoso.

    - Selección de avance.

    - Eliminación hacia atrás.

    - Búsqueda exhaustiva.

- Métodos incorporados: los algoritmos de aprendizaje tienen una selección de características incorporada. Por ejemplo: regularización L1.

    - Regularización LASSO.
    
    - Importancias de las características.


Usa el siguiente [notebook](https://github.com/priyamnagar/feature_selection_titanic/blob/master/Titanic.ipynb) para ver cómo aplicar cada uno de estos métodos en un conjunto de datos que ya se ha dividido en conjuntos de entrenamiento y validación.

Considera los siguientes enlaces para métodos estadísticos si planeas aplicar la selección de características:

-[Chi square test](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2)

-[Anova](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression)

-[Mutual Information](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif)


