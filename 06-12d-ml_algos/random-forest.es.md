# Random Forest

**¿Qué es el Random Forest**

**¿En qué se diferencia Random Forest de los algoritmos tradicionales de Decision Tree?**

Random Forest es un método de conjunto que utiliza árboles de decisión empaquetados con subconjuntos de características aleatorias elegidos en cada punto de división. Luego, promedia los resultados de predicción de cada árbol (regresión) o usa los votos de cada árbol (clasificación) para hacer la predicción final.

**¿Qué hiperparámetros se pueden ajustar para un bosque aleatorio que se suman a los hiperparámetros de cada árbol individual?**

El Random Forest es esencialmente árboles de decisión empaquetados con subconjuntos de características aleatorias elegidos en cada punto de división, por lo que tenemos 2 nuevos hiperparámetros que podemos ajustar:

Num estimators - el número de árboles de decisión en el bosque.

características máximas: número máximo de características que se evalúan para dividir en cada nodo.

**¿Son los modelos de Random Forest propensos al sobreajuste? ¿Por qué?**

No, los modelos de Random Forest generalmente no son propensos a sobreajustarse porque la selección aleatoria de funciones y el embolsado tienden a promediar cualquier ruido en el modelo. La adición de más árboles no provoca el sobreajuste, ya que el proceso de aleatorización continúa promediando el ruido (más árboles generalmente reducen el sobreajuste en el bosque aleatorio).

En general, los algoritmos de ensacado son resistentes al sobreajuste.

Dicho esto, es posible sobreajustar con modelos de Random Forest si los árboles de decisión subyacentes tienen una varianza extremadamente alta. En cada punto de división se considera una profundidad extremadamente alta y una división de muestra mínima baja, y un gran porcentaje de características. Por ejemplo, si todos los árboles son idénticos, el Random Forest puede sobreajustar los datos.
