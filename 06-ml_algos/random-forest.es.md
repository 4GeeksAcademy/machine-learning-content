---
description: >-
  Learn about Random Forests, a powerful Machine Learning method for
  classification and regression. Discover how to build robust models with ease!
---
## Random Forests

Un **random forest** (suele referirse a él en su término en inglés y no en su traducción) es un método de Machine Learning que se utiliza para tareas de clasificación y regresión. Es un tipo de aprendizaje en el que se **ensamblan modelos** (*model ensembling*) para combinar las predicciones de múltiples árboles de decisión para generar una salida más precisa y robusta.

Cada árbol en un random forest se construye de manera independiente utilizando un subconjunto aleatorio de los datos de entrenamiento. Luego, para hacer la predicción, cada árbol en el bosque hace su propia predicción y la predicción final se toma por votación de mayoría en el caso de clasificación, o promedio en el caso de regresión.

Este enfoque ayuda a superar el problema del sobreajuste, que es común con los árboles de decisión individuales.

### Estructura

Un random forest es una colección de árboles de decisión. Cada uno de estos árboles es un modelo que consta de nodos de decisión y hojas. Los nodos de decisión, recordemos, son puntos donde se toman decisiones basadas en ciertos atributos o características, y las hojas son los resultados finales o las predicciones.

Así, para construir cada árbol de decisión, el random forest selecciona un subconjunto aleatorio de los datos de entrenamiento. Este proceso se llama **empaquetado** (*bagging*) o **agregación de bootstrap** (*bootstrap aggregating*).

Además de seleccionar subconjuntos aleatorios de datos, el random forest también selecciona un subconjunto aleatorio de las características de cada árbol. Esto añade otra capa de aleatoriedad al modelo, lo que ayuda a aumentar la diversidad entre los árboles y mejorar la robustez del modelo en general.

![Un random_forest](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/random_forest.PNG?raw=true)

Una vez entrenado, cada árbol de decisión dentro del random forest realiza su propia predicción. Para problemas de clasificación, la clase que obtenga la mayoría de votos entre todos los árboles se selecciona como la predicción final. Para problemas de regresión, la predicción final se obtiene promediando las predicciones de todos los árboles.

La estructura del random forest, con su combinación de aleatoriedad y agregación, ayuda a crear un modelo robusto que es menos propenso a sobreajustarse a los datos de entrenamiento en comparación con un solo árbol de decisión.

### Hiperparametrización del modelo

Podemos construir un árbol de decisión fácilmente en Python utilizando la librería `scikit-learn` y las funciones `RandomForestClassifier` y `RandomForestRegressor`. Algunos de sus hiperparámetros más importantes y los primeros en los que debemos centrarnos son:

- `n_estimators`: Este es probablemente el hiperparámetro más importante. Define el número de árboles de decisión en el bosque. En general, un número mayor de árboles aumenta la precisión y hace que las predicciones sean más estables, pero también puede ralentizar considerablemente el tiempo de cálculo.
- `bootstrap`: Este hiperparámetro se usa para controlar si se utilizan muestras de bootstrap (muestreo con reemplazo) para la construcción de árboles.
- `max_depth`: La profundidad máxima de los árboles. Esto es esencialmente cuántas divisiones puede hacer el árbol antes de hacer una predicción.
- `min_samples_split`: El número mínimo de muestras necesarias para dividir un nodo en cada árbol. Si se establece un valor alto, evita que el modelo aprenda relaciones demasiado específicas y, por tanto, ayuda a prevenir el sobreajuste.
- `min_samples_leaf`: El número mínimo de muestras que se deben tener en un nodo hoja en cada árbol.
- `max_features`: El número máximo de características a considerar al buscar la mejor división dentro de cada árbol. Por ejemplo, si tenemos 10 características, podemos elegir que cada árbol considere solo un subconjunto de ellas al decidir dónde dividir.

Como podemos ver, solo los dos primeros hiperparámetros hacen referencia al random forest, mientras que el resto se relacionan con los árboles de decisión. Otro hiperparámetro muy importante es el `random_state`, que controla la semilla de generación aleatoria. Este atributo es crucial para asegurar la replicabilidad.
