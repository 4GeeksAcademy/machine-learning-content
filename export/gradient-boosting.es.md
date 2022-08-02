# Aumento de gradiente

**¿En qué se diferencian las máquinas de aumento de gradiente de los algoritmos de árboles de decisión tradicionales?**

El aumento de gradiente implica el uso de múltiples predictores débiles (árboles de decisión) para crear un predictor fuerte. Específicamente, incluye una función de pérdida que calcula el gradiente del error con respecto a cada característica y luego crea iterativamente nuevos árboles de decisión que minimizan el error actual. Se agregan más y más árboles al modelo actual para continuar corrigiendo errores hasta que las mejoras caen por debajo de un umbral mínimo o se ha creado un número predeterminado de árboles.

**¿Qué hiperparámetros se pueden ajustar en el aumento de gradiente además de los hiperparámetros de cada árbol individual?**

Los principales hiperparámetros que se pueden ajustar con los modelos GBM son:

- Loss function - función de pérdida para calcular el gradiente de error.

- Learning rate - la velocidad a la que los árboles nuevos corrigen/modifican el predictor existente.

- Num estimators - el número total de árboles a producir para el predictor final.

**Hiperparámetros adicionales específicos de la función de pérdida**

Algunas implementaciones específicas, por ejemplo, el aumento de gradiente estocástico, pueden tener hiperparámetros adicionales, como el tamaño de la submuestra (el tamaño de la submuestra afecta la aleatorización en las variaciones estocásticas).

**¿Cómo podemos reducir el sobreajuste al aumentar el gradiente?**

Reducir la tasa de aprendizaje o reducir el número máximo de estimadores son las dos formas más fáciles de lidiar con modelos de aumento de gradiente que sobreajustan los datos.

Con el aumento de gradiente estocástico, la reducción del tamaño de la submuestra es una forma adicional de combatir el sobreajuste.

Los algoritmos de refuerzo tienden a ser vulnerables al sobreajuste, por lo que es importante saber cómo reducir el sobreajuste.
