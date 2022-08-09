# Básicos de Machine Learning 

## ¿Qué es Machine Learning? 

Machine learning es el campo de la ciencia que estudia algoritmos que aproximan funciones cada vez mejor a medida que se les dan más observaciones. 

Los algoritmos de Machine Learning son programas que pueden aprender de los datos y mejorar a partir de la experiencia, sin intervención humana. Las tareas de aprendizaje pueden incluir el aprendizaje de la función que asigna la entrada a la salida, el aprendizaje de la estructura oculta en los datos no etiquetados; o "aprendizaje basado en instancias", donde se produce una etiqueta de clase para una nueva instancia comparando la nueva instancia (fila) con las instancias de los datos de entrenamiento, que se almacenaron en la memoria.

## Tipos de algoritmos de Machine Learning

### Aprendizaje supervisado

Es útil en los casos en que una etiqueta está disponible para un determinado conjunto de entrenamiento, pero falta y debe predecirse para otras instancias. Utiliza los datos de entrenamiento etiquetados para aprender la función de mapeo que convierte las variables de entrada (X) en la variable de salida (Y).

Los datos etiquetados son datos que tienen la información sobre la variable de destino para cada instancia.

Tipos de algoritmos de Machine Learning supervisado:

- La clasificación se utiliza para predecir el resultado de una muestra dada cuando la variable de salida está en forma de categorías.

- La regresión se utiliza para predecir el resultado de una muestra dada cuando la variable de salida está en forma de valores reales.

- El ensamblaje es otro tipo de aprendizaje supervisado. Significa combinar las predicciones de múltiples modelos de Machine Learning que son individualmente débiles para producir una predicción más precisa en una nueva muestra. Ensamblar significa combinar los resultados de múltiples estudiantes (clasificadores) para obtener mejores resultados, mediante votación o promediación. La votación se usa durante la clasificación y el promedio se usa durante la regresión. La idea es que los conjuntos de alumnos se desempeñen mejor que los alumnos individuales. El Bagging (embolsado) y el Boosting (impulso) son dos tipos de algoritmos de ensamblaje.

**¿Cuál es la diferencia entre Bagging y Boosting?**

El Bagging y el Boosting son métodos de ensamblado, lo que significa que combinan muchos predictores débiles para crear un predictor fuerte. Una diferencia clave es que el embolsado construye modelos independientes en paralelo, mientras que el impulso construye modelos secuencialmente, en cada paso enfatizando las observaciones que se perdieron en los pasos anteriores.

Algunos ejemplos de algoritmos de aprendizaje supervisado son:

1. Árboles de decisión.

2. Clasificación Naive Bayes.

3. Regresión de mínimos cuadrados ordinarios.

4. Regresión logística.

5. Máquinas de vectores de soporte.

### Aprendizaje sin supervisión

Es útil en los casos en que el desafío es descubrir relaciones implícitas en un conjunto de datos no etiquetado dado. En otras palabras, solo tenemos las variables de entrada (X) y ninguna variable de salida correspondiente.

Tipos de aprendizaje no supervisado:

- La asociación se utiliza para descubrir la probabilidad de la co-ocurrencia de elementos en una colección. Por ejemplo, se podría utilizar un modelo de asociación para descubrir que si un cliente compra pan, tiene un 80 % de probabilidades de que también compre huevos.

- La agrupación se utiliza para agrupar muestras de modo que los objetos dentro del mismo grupo sean más similares entre sí que con los objetos de otro grupo.

- La reducción de dimensionalidad se utiliza para reducir el número de variables de un conjunto de datos y, al mismo tiempo, garantizar que aún se transmita información importante. La reducción de la dimensionalidad se puede realizar utilizando métodos de extracción de características y métodos de selección de características.

Algunos ejemplos de algoritmos de aprendizaje no supervisados ​​son:

1. K-means

2. PCA

### Aprendizaje reforzado

ESe encuentra entre estos 2 extremos — Describe un conjunto de algoritmos que aprenden del resultado de cada decisión.
Los algoritmos de refuerzo generalmente aprenden acciones óptimas a través de prueba y error.

Por ejemplo, un robot podría usar el aprendizaje por refuerzo para aprender que caminar hacia adelante contra una pared es malo, pero alejarse de una pared y caminar es bueno, o imagina un videojuego en el que el jugador necesita moverse a ciertos lugares en ciertos momentos para ganar puntos. Un algoritmo de refuerzo que jugaría ese juego comenzaría moviéndose aleatoriamente pero, con el tiempo, a través de prueba y error, aprendería dónde y cuándo necesita mover el personaje del juego para maximizar su total de puntos.

## Aprendizaje en línea vs fuera de línea

Si bien el aprendizaje en línea tiene sus usos, Machine Learning tradicional se realiza fuera de línea mediante el método de aprendizaje por lotes (batch).

En el aprendizaje por lotes, los datos se acumulan durante un período de tiempo. Luego, el modelo de machine learning se entrena con estos datos acumulados de vez en cuando en lotes. Si ingresan nuevos datos, se debe ingresar un lote nuevo completo (incluidos todos los datos antiguos y nuevos) en el algoritmo para aprender de los nuevos datos. En el aprendizaje por lotes, el algoritmo de machine learning actualiza sus parámetros solo después de consumir lotes de datos nuevos.

Es exactamente lo opuesto al aprendizaje en línea porque el modelo no puede aprender de manera incremental a partir de un flujo de datos en vivo. El aprendizaje en línea se refiere a la actualización gradual de los modelos a medida que obtienen más información.

## ¿Cómo se dividen los datos?

¿Qué son los datos de entrenamiento y para qué sirven?

Los datos de entrenamiento son un conjunto de ejemplos que se utilizarán para entrenar el modelo de machine learning.
Para machine learning supervisado, estos datos de entrenamiento deben tener una etiqueta. Lo que está tratando de predecir debe ser definido.

Para el aprendizaje automático no supervisado, los datos de entrenamiento contendrán solo características y no usarán objetivos etiquetados. Lo que está tratando de predecir no está definido.

¿Qué es un conjunto de validación y por qué utilizar uno?

Un conjunto de validación es un conjunto de datos que se utiliza para evaluar el rendimiento de un modelo durante el entrenamiento/selección del modelo. Una vez que se entrenan los modelos, se evalúan en el conjunto de validación para seleccionar el mejor modelo posible.

Nunca debe usarse para entrenar el modelo directamente.

Tampoco debe usarse como el conjunto de datos de prueba porque hemos sesgado la selección de nuestro modelo para que funcione bien con estos datos, incluso si el modelo no se entrenó directamente con ellos.

¿Qué es un equipo de prueba y por qué utilizar uno?

Un conjunto de prueba es un conjunto de datos que no se utilizan durante el entrenamiento o la validación. El rendimiento del modelo se evalúa en el conjunto de prueba para predecir qué tan bien se generalizará a nuevos datos.

### Técnicas de entrenamiento y validación

Un objetivo del aprendizaje supervisado es construir un modelo que funcione bien con nuevos datos. Si tiene datos nuevos, es una buena idea ver cómo funciona su modelo en ellos. El problema es que es posible que no tenga datos nuevos, pero puede simular esta experiencia con un procedimiento como train test split.

**Train-test-split**

Train test split es un procedimiento de validación de modelo que le permite simular cómo se comportaría un modelo con datos nuevos/no vistos. Así es como funciona el procedimiento.

1. Asegúrate de que tus datos estén organizados en un formato aceptable para train test split. En scikit-learn, esto consiste en separar su conjunto de datos completo en Funciones y Objetivo. 

2. Divide el conjunto de datos en dos partes: un conjunto de entrenamiento y un conjunto de prueba. Esto consiste en un muestreo aleatorio sin reemplazo de alrededor del 75 % (puede variar esto) de las filas y colocarlas en su conjunto de entrenamiento y colocar el 25 % restante en su conjunto de prueba. Ten en cuenta que los colores en "Features" (Características) y "Target" (Objetivo) indican dónde irán tus datos (“X_train”, “X_test”, “y_train”, “y_test”) para un train test split en particular.

3. Entrena al modelo en el conjunto de entrenamiento. Esto es "X_train" y "y_train" en la imagen.

4. Prueba el modelo en el conjunto de prueba ("X_test" y "y_test" en la imagen) y evalúa el rendimiento.

![train_test_split](../assets/train_test_split.jpg)

Código de ejemplo:

```py

# Separar tu conjunto de datos completo en Características y Objetivo

features = [‘bedrooms’,’bathrooms’,’sqft_living’,’sqft_lot’,’floors’]
X = df.loc[:, features]
y = df.loc[:, [‘price’]]

# Dividir los datos y devuelve una lista que contiene cuatro arrays NumPy.

#train_size = 0.75 coloca el 75% de los datos en un conjunto de entrenamiento y el 25% restante en un conjunto de prueba

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, train_size = 0.75)

# Ajustar el modelo

reg.fit(X_train, y_train)

# Hacer predicciones sobre el conjunto de validación (X_test)

reg.predict(X_test)

# Midiendo los resultados

score = reg.score(X_test, y_test)
print(score)

```

**Validación Cruzada**

Otra técnica de validación de modelos es la validación cruzada, también conocida como técnica fuera de muestreo. Esta es una técnica de remuestreo para entrenar y validar modelos con mayor precisión. Rota qué datos se retienen del entrenamiento del modelo para usarlos como datos de validación.

Se entrenan y evalúan varios modelos, y cada pieza de datos se obtiene de un modelo. A continuación, se calcula el rendimiento medio de todos los modelos.

Es una forma más confiable de validar modelos, pero es más costosa desde el punto de vista computacional. Por ejemplo, la validación cruzada de 5 veces requiere entrenar y validar 5 modelos en lugar de 1.

![cross_validation](../assets/cross_validation.jpg)

Algunas técnicas de validación cruzada:

- Omitir p validación cruzada.

- Dejar uno fuera de la validación cruzada.

- Validación cruzada de retención.

- Validación de submuestreo aleatorio repetido.

- validación cruzada k-fold.

- Validación cruzada estratificada de k-fold.

- Validación cruzada de series temporales.

- Validación cruzada anidada.

La implementación de estas validaciones cruzadas se puede encontrar en el paquete sklearn. Lee esta [documentación de sklearn](https://scikit-learn.org/stable/modules/cross_validation.html) para obtener más detalles. Las validaciones cruzadas K-fold y K-fold estratificadas son las técnicas más utilizadas. Aquí mostraremos cómo implementar la validación cruzada K-fold.

La validación cruzada K-fold es una técnica superior para validar el rendimiento de nuestro modelo. Evalúa el modelo utilizando diferentes fragmentos del conjunto de datos como conjunto de validación.


**¿Qué es el sobreajuste?**

Sobreajuste cuando un modelo hace predicciones mucho mejores sobre datos conocidos (datos incluidos en el conjunto de entrenamiento) que sobre datos desconocidos (datos no incluidos en el conjunto de entrenamiento).

¿Cómo podemos combatir el sobreajuste?

Algunas formas de combatir el sobreajuste son:

- Simplificar el modelo (a menudo se hace cambiando).

- Seleccionar un modelo diferente.

- Usar más datos de entrenamiento.

- Recopilar datos de mejor calidad para combatir el sobreajuste.

¿Cómo podemos saber si nuestro modelo está sobreajustando los datos?

Si nuestro error de entrenamiento es bajo y nuestro error de validación es alto, lo más probable es que nuestro modelo esté sobreajustando nuestros datos de entrenamiento.

¿Cómo podemos saber si nuestro modelo no se ajusta bien a los datos?

Si nuestro error de entrenamiento y validación son relativamente iguales y muy altos, lo más probable es que nuestro modelo no se ajuste bien a nuestros datos de entrenamiento.

**¿Qué son los datos de pipelines?**

Cualquier colección de transformaciones ordenadas en datos.

Fuente:
    
https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

https://www.kdnuggets.com/2020/09/understanding-bias-variance-trade-off-3-minutes.html

https://medium.com/@ranjitmaity95/7-tactics-to-combat-imbalanced-classes-in-machine-learning-datase-4266029e2861

