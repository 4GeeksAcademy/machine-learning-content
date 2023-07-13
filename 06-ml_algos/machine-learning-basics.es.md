## Introducción al Machine Learning

### Definición

El **Aprendizaje Automático** (*Machine learning*) es una rama de la Inteligencia Artificial que se centra en la construcción de sistemas que pueden aprender de los datos, en lugar de seguir solo reglas programadas explícitamente.

Los sistemas de Machine Learning utilizan algoritmos para analizar datos, aprender de ellos y luego hacer predicciones, en lugar de ser programados específicamente para llevar a cabo la tarea. Por ejemplo, un modelo de Machine Learning podría ser entrenado para reconocer gatos al proporcionarle miles de imágenes con y sin gatos. Con suficientes ejemplos, el sistema "aprende" a distinguir las características que definen a un gato, así podría identificarlos en nuevas imágenes que nunca antes había visto.

### Tipos de aprendizaje

Según cómo el modelo puede aprender a partir de los datos, existen varios tipos de aprendizaje:

#### Aprendizaje supervisado

Los modelos son entrenados en un conjunto de datos etiquetado. Un conjunto de datos etiquetado es un conjunto de datos que contiene tanto los datos de entrada como las respuestas correctas, también conocidas como etiquetas o valores objetivo.

El objetivo del aprendizaje supervisado es aprender una función que transforme las entradas en salidas. Dependiendo del tipo de salida que queremos que el modelo sea capaz de generar, podemos dividir los modelos en varios tipos:

- **Regresión** (regression). Cuando la etiqueta o el valor objetivo es un número continuo (como el precio de una casa), el problema se considera un problema de regresión. El modelo debe devolver un número en una escala infinita.
- **Clasificación** (classification). Cuando la etiqueta es categórica (como predecir si un correo es spam o no), el problema es de clasificación. El modelo debe devolver una etiqueta según se corresponda a una clase u otra.

Algunos ejemplos de modelos fundamentados en este tipo de aprendizaje son:

- Regresión logística y lineal.
- Árboles de decisión.
- Clasificador "Naive Bayes".
- Máquinas de Vectores de Soporte.

#### Aprendizaje no supervisado

En este tipo de aprendizaje, en contraposición a lo que sucedía con el anterior, los modelos se entrenan usando un conjunto de datos sin etiquetas. En este tipo de aprendizaje, el objetio es encontrar patrones o estructuras ocultas en los datos.

Puesto que en este tipo de aprendizaje no hay etiquetas, los modelos deben descubrir por sí mismos las relaciones en los datos.

Algunos ejemplos de modelos fundamentados en este tipo de aprendizaje son:

- Agrupamiento (*clustering*).
- Reducción de la dimensionalidad.

#### Aprendizaje por refuerzo

En este aprendizaje el modelo (también llamado agente) aprende a tomar decisiones óptimas a través de la interacción con su entorno. El objetivo es maximizar alguna noción de recompensa acumulativa.

En el aprendizaje por refuerzo, el agente toma acciones, las cuales afectan el estado del ambiente, y recibe retroalimentación en forma de recompensas o penalizaciones. La meta es aprender una estrategia para maximizar su recompensa a largo plazo.

Un ejemplo de este tipo de aprendizaje es un programa que aprenda a jugar al ajedrez. El agente (el programa) decide qué movimiento hacer (las acciones de mover ficha) en diferentes posiciones del tablero de ajedrez (los estados) para maximizar la posibilidad de ganar el juego (la recompensa).

Este tipo de aprendizaje es diferente de los dos anteriores. En lugar de aprender a partir de un conjunto de datos con o sin etiquetas, el aprendizaje por refuerzo está centrado en tomar decisiones óptimas y aprender a partir de la retroalimentación de esas decisiones.

### Conjuntos de datos

Los datos son una pieza fundamental en cualquier algoritmo de Machine Learning. Sin ellos, independientemente del tipo de aprendizaje o modelo, no existe forma de iniciar ningún proceso de aprendizaje.

Un conjunto de datos es una colección que normalmente se representa en forma de tabla. En esta tabla, cada fila representa una observación o instancia y cada columna representa una característica, atributo o variable de esa obversación. Este conjunto de datos es utilizado para entrenar y evaluar modelos:

1. Entrenamiento del modelo. Un modelo de Machine Learning aprende a partir de un conjunto de datos de entrenamiento. El modelo entrena ajustando sus parámetros internamente.
2. Evaluación del modelo. Una vez que el modelo ha sido entrenado, se utiliza un conjunto de datos de prueba independiente para evaluar su rendimiento. Este dataset contiene observaciones que no se utilizaron durante el entrenamiento, lo que permite obtener una evaluación imparcial de cómo se espera que el modelo realice predicciones sobre nuevos datos.

En algunas situaciones, también se utiliza un conjunto de validación, que se utiliza para evaluar el rendimiento de un modelo durante el entrenamiento. Una vez que se entrenan los modelos, se evalúan en el conjunto de validación para seleccionar el mejor modelo posible.

#### División del conjunto de datos

El paso previo de entrenar un modelo, además del EDA, es dividir los datos en un conjunto de entrenamiento (`train dataset`) y un conjunto de prueba (`test dataset`), en un procedimiento como el siguiente:

1. Asegúrate de que tus datos estén organizados en un formato aceptable. Si trabajamos con ficheros de texto deberían ser formato de tabla, y si trabajamos con imágenes, los propios docmentos en sí.
2. Divide el conjunto de datos en dos partes: un conjunto de entrenamiento y un conjunto de prueba. Seleccionaremos aleatoriamente un 80% (puede variar) de las filas y las colocaremos en el conjunto de entrenamiento y el 20% restante en el conjunto de prueba. Además, debemos dividir las predictoras de las clases, conformando 4 elementos: `X_train`, `y_train`, `X_test`, `y_test`.
3. Entrena el modelo usando el conjunto de entrenamiento (`X_train`, `y_train`).
3. Prueba el modelo usando el conjunto de prueba (`X_test`, `y_test`).

![train_test_split](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/train_test_split.jpg?raw=true)


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

![cross_validation](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/cross_validation.jpg?raw=true)

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

Dividimos nuestro conjunto de datos en K-pliegues. K representa el número de pliegues en los que desea dividir sus datos. Si usamos 5 pliegues, el conjunto de datos se divide en cinco secciones. En diferentes iteraciones, una parte se convierte en el conjunto de validación.

*Implementación de validación cruzada de K-fold*

```py
from sklearn.model_selection import cross_validate

# Ejemplo con una función:

def cross_validation(model, _X, _y, _cv=5):

    scoring = ['accuracy', 'precision']

    results = cross_validate(estimator=model,
                                X=_X,
                                y=_y,
                                cv=_cv,
                                scoring=_scoring,
                                return_train_score=True)

    return {"Training Accuracy scores": results['train_accuracy'],
              "Mean Training Accuracy": results['train_accuracy'].mean()*100,
              "Training Precision scores": results['train_precision'],
              "Mean Training Precision": results['train_precision'].mean(),
              "Validation Accuracy scores": results['test_accuracy'],
              "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
              "Validation Precision scores": results['test_precision'],
              "Mean Validation Precision": results['test_precision'].mean(),
              }
        
```

La función cross_validation personalizada en el código anterior realizará una validación cruzada de 5 veces. Devuelve los resultados de las métricas especificadas anteriormente.

El parámetro estimador de la función cross_validate recibe el algoritmo que queremos usar para el entrenamiento. El parámetro X toma la matriz de características. El parámetro y toma la variable objetivo. La puntuación de parámetros toma las métricas que queremos usar para la evaluación. Pasamos una lista que contiene las métricas que queremos usar para verificar nuestro modelo.

Con cualquier procedimiento de validación de modelos es importante tener en cuenta algunas ventajas y desventajas que en el caso de train test split son:

Algunas ventajas:

- Relativamente simple y más fácil de entender que otros métodos como la validación cruzada K-fold.

- Ayuda a evitar modelos demasiado complejos que no se generalizan bien a nuevos datos.

Algunas desventajas:

- Elimina datos que podrían haberse usado para entrenar un modelo de machine learning (los datos de prueba no se usan para entrenar)

- Con el cambio en el estado aleatorio de la división, la precisión del modelo también cambia, por lo que no podemos lograr una precisión fija para el modelo. Los datos de prueba deben mantenerse independientes de los datos de entrenamiento para que no se produzcan fugas de datos.

## ¿Qué es el sobreajuste?

El **sobreajuste** (*overfitting*) se da cuando el modelo se entrena con muchos datos. Cuando un modelo se entrena con tantos datos, comienza a aprender del ruido y las entradas de datos inexactas de nuestro dataset. Debido a esto, el modelo no devuelve una salida acertada. Combatir el sobreajuste es una tarea iterativa y que deriva en la experiencia del desarrollador, y podemos empezar con:

- Realizar un correcto EDA, seleccionando valores y variables significativas para el modelo siguiendo la regla del "menos es más".
- Simplificar o cambiar el modelo que estamos utilizando.
- Usar más o menos datos de entrenamiento.

Detectar si el modelo está sobreajustando los datos es también una ciencia, y lo podemos determinar si la métrica del modelo en el conjunto de datos de entrenamiento es muy alta, y la métrica del conjunto de prueba es baja.

En cambio, si no hemos entrenado suficiente el modelo también podemos verlo simplemente comparando la métrica de entrenamiento y prueba, de tal forma que si son relativamente iguales y muy altos, lo más probable es que nuestro modelo no se ajuste bien a nuestros datos de entrenamiento.