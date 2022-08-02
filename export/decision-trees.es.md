### Árboles de decisión

**¿Qué son los árboles de decisión?**

**¿Cuáles son los usos comunes de los algoritmos de árboles de decisión?**

1. Clasificación.

2. Regresión.

3. Medir la importancia de la característica.

4. Selección de características.

**¿Cuáles son los principales hiperparámetros que puedes ajustar para los árboles de decisión?**

En términos generales, los árboles de decisión tienen los siguientes parámetros:

- Max depth - profundidad máxima del árbol.

- Min samples split - número mínimo de muestras para dividir un nodo.

- Min samples leaf - número mínimo de muestras para cada nodo hoja.

- Max leaf nodes - el número máximo de nodos hoja en el árbol.

- Max features - número máximo de características que se evalúan para la división en cada nodo (solo válido para algoritmos que aleatorizan las características consideradas en cada división).

El árbol de decisión tradicional es codicioso y analiza todas las funciones en cada punto de división, pero muchas implementaciones modernas permiten dividir en funciones aleatorias (como se ve en scikit learn), por lo que las funciones máximas pueden o no ser un hiperparámetro ajustable.

**¿Cómo afecta cada hiperparámetro a la capacidad de aprendizaje del modelo?**

- Max depth: aumentando la profundidad máxima disminuirá el sesgo y aumentará la varianza.

- Min samples split: aumentando la división de muestras mínimas aumenta el sesgo y disminuye la varianza.

- Min samples leaf: el aumento de la hoja de muestras mínimas aumenta el sesgo y disminuye la varianza.

- Max leaf nodes: la disminución del nodo de hoja máxima aumenta el sesgo y disminuye la varianza.

- Max features: la disminución de las características máximas aumenta el sesgo y disminuye la varianza.

Puede haber casos en los que cambiar los hiperparámetros no tenga efecto en el modelo.

**¿Qué métricas se utilizan normalmente para calcular las divisiones?**

Gini impureza o entropía. Ambos generalmente producen resultados similares.

**¿Qué es la impureza de Gini?**

La impureza de Gini es una medida de la frecuencia con la que un registro elegido al azar se clasificaría incorrectamente si se clasificara al azar utilizando la distribución del conjunto de muestras.

Un Gini bajo (cerca de 0) significa que la mayoría de los registros de la muestra están en la misma clase.

Un Gini alto (máximo de 1 o menos, según el número de clases) = los registros de la muestra se distribuyen uniformemente entre las clases.

**¿Qué es la entropía?**

La entropía es la medida de la pureza de los miembros entre las clases no vacías. Es muy similar a Gini en concepto, pero un cálculo ligeramente diferente.

Baja entropía (cerca de 0) significa que la mayoría de los registros de la muestra están en la misma clase.

Una entropía alta (máximo de 1) significa que los registros de la muestra se distribuyen uniformemente entre las clases.

**¿Los árboles de decisión son modelos paramétricos o no paramétricos?**

No paramétrico. El número de parámetros del modelo no se determina antes de crear el modelo.

**¿Cuáles son algunas formas de reducir el sobreajuste con árboles de decisión?**

- Reducir la profundidad máxima.

- Aumentar la división mínima de muestras.

- Equilibrar los datos para evitar el sesgo hacia las clases dominantes.

- Aumentar el número de muestras.

- Disminuir el número de funciones.

**¿Cómo se evalúa la importancia de las características en los modelos basados ​​en árboles de decisión?**

Las características que se dividen con mayor frecuencia y que están más cerca de la parte superior del árbol, lo que afecta a la mayor cantidad de muestras, se consideran las más importantes.
