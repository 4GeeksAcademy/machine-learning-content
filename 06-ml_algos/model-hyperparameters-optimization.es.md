# Optimización de Hiperparámetros del Modelo

## ¿Qué es un hiperparámetro modelo?

Un hiperparámetro de modelo es el parámetro cuyo valor se establece antes de que el modelo comience a entrenar. No se pueden aprender ajustando el modelo a los datos.

Ejemplos de hiperparámetros de modelo en diferentes modelos:

- Tasa de aprendizaje en descenso de gradiente.

- Número de iteraciones en descenso de gradiente.

- Número de capas en una Red Neuronal.

- Número de neuronas por capa en una Red Neuronal.

- Número de agrupaciones (k) en k significa agrupamiento.

## Diferencia entre parámetro e hiperparámetro

Un parámetro del modelo es una variable del modelo seleccionado que se puede estimar ajustando los datos proporcionados al modelo. Por ejemplo, en la regresión lineal, la pendiente y la intersección de la línea son dos parámetros estimados ajustando una línea recta a los datos minimizando el RMSE.

![parameter_vs_hyperparameter]([https://github.com/4GeeksAcademy/machine-learning-content/raw/master/assets/parameter_vs_hyperparameter.jpg?](https://raw.githubusercontent.com/4GeeksAcademy/machine-learning-content/master/assets/parameter_vs_hyperparameter.jpg))

Y, como ya mencionamos, se establece un valor de hiperparámetro del modelo antes de que el modelo comience a entrenarse y no se puede aprender ajustando el modelo a los datos.

La mejor parte es que tiene la opción de seleccionarlos para su modelo. Por supuesto, debe seleccionar de una lista específica de hiperparámetros para un modelo determinado, ya que varía de un modelo a otro.

A menudo, no conocemos los valores óptimos para los hiperparámetros que generarían el mejor resultado del modelo. Entonces, lo que le decimos al modelo es que explore y seleccione la arquitectura del modelo óptimo automáticamente. Este procedimiento de selección de hiperparámetros se conoce como ajuste de hiperparámetros.

## ¿Cuáles son dos formas comunes de automatizar el ajuste de hiperparámetros?

El ajuste de hiperparámetros es una técnica de optimización y es un aspecto esencial del proceso de aprendizaje automático. Una buena elección de hiperparámetros puede hacer que su modelo cumpla con la métrica deseada. Sin embargo, la plétora de hiperparámetros, algoritmos y objetivos de optimización puede conducir a un ciclo interminable de esfuerzo continuo de optimización.

1. Búsqueda en cuadrícula: prueba todas las combinaciones posibles de valores de hiperparámetros predefinidos y selecciona la mejor.

Ejemplo:

```py

# Importar bibliotecas
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

# Cargar los datos
iris = datasets.load_iris()

# Establecer parámetros
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

# Eligir el modelo
svc = svm.SVC()

# Buscar todas las combinaciones posibles
clf = GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)

# Obtener las claves de hiperparámetro
sorted(clf.cv_results_.keys())

```

Consulta la documentación completa de scikit-learn sobre GridSearchCV:

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html   


2. Búsqueda aleatoria: prueba aleatoriamente las posibles combinaciones de valores de hiperparámetros predefinidos y selecciona la mejor probada.

Ejemplo:

```py

# Importar bibliotecas
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform

# Cargar los datos
iris = load_iris()

# Elige el modelo
logistic = LogisticRegression(solver='saga', tol=1e-2, max_iter=200, random_state=0)

# Establecer posibles hiperparámetros
distributions = dict(C=uniform(loc=0, scale=4), penalty=['l2', 'l1'])

# Hacer una búsqueda aleatoria en posible combinación entre los hiperparámetros establecidos
clf = RandomizedSearchCV(logistic, distributions, random_state=0)
search = clf.fit(iris.data, iris.target)

# Obtener los mejores valores de hiperparámetro
search.best_params_

{'C': 2..., 'penalty': 'l1'}

```

Consulta la documentación completa de scikit-learn sobre RandomizedSearchCV:

https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html   


**¿Cuáles son los pros y los contras de la búsqueda en cuadrícula?**

**Pros:**

Grid Search es excelente cuando necesitamos ajustar hiperparámetros en un pequeño espacio de búsqueda automáticamente. Por ejemplo, si tenemos 100 conjuntos de datos diferentes que esperamos que sean similares, como resolver el mismo problema repetidamente con diferentes poblaciones. Podemos usar la búsqueda en cuadrícula para ajustar automáticamente los hiperparámetros para cada modelo.

**Cons:** 

Grid Search es computacionalmente costoso e ineficiente, a menudo busca en un espacio de parámetros que tiene muy pocas posibilidades de ser útil, lo que hace que sea extremadamente lento. Es especialmente lento si necesitamos buscar en un espacio grande ya que su complejidad aumenta exponencialmente a medida que se optimizan más hiperparámetros.

**¿Cuáles son los pros y los contras de la búsqueda aleatoria?**

**Pros:**

La búsqueda aleatoria hace un buen trabajo al encontrar hiperparámetros casi óptimos en un espacio de búsqueda muy grande con relativa rapidez y no sufre el mismo problema de escala exponencial que la búsqueda en cuadrícula. 

**Cons:**

La búsqueda aleatoria no ajusta los resultados tanto como lo hace la búsqueda en cuadrícula, ya que normalmente no prueba todas las combinaciones posibles de parámetros.

**Ejemplos de preguntas que nos responderá el ajuste de hiperparámetros**

- ¿Cuál debería ser el valor para la profundidad máxima del árbol de decisión?

- ¿Cuántos árboles debo seleccionar en un modelo Random Forest?

- Debería usar una red neuronal de una sola capa o de múltiples capas, si hay varias capas, ¿cuántas capas debería haber?

- ¿Cuántas neuronas debo incluir en la Red Neuronal?

- ¿Cuál debería ser el valor de división de muestra mínimo para el árbol de decisión?

- ¿Qué valor debo seleccionar para la hoja de muestra mínima para mi árbol de decisión?

- ¿Cuántas iteraciones debo seleccionar para la red neuronal?

- ¿Cuál debería ser el valor de la tasa de aprendizaje para descenso de gradiente?

- ¿Qué método de resolución es el más adecuado para mi red neuronal?

- ¿Cuál es la K en K-nearest Neighbors?

- ¿Cuál debería ser el valor de C y sigma en Support Vector Machine?


Fuente: 

https://www.geeksforgeeks.org/difference-between-model-parameters-vs-hyperparameters/

https://www.mygreatlearning.com/blog/hyperparameter-tuning-explained/

https://medium.com/codex/do-i-need-to-tune-logistic-regression-hyperparameters-1cb2b81fca69
