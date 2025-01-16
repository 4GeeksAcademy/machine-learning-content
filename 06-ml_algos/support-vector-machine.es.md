## Máquina de Vectores de Soporte

Las **Máquinas de Vectores de Soporte** (*SVM*, *Support Vector Machines*) son una clase de algoritmos de aprendizaje supervisado ampliamente utilizados en clasificación y regresión. Estas se introdujeron en la década de 1990 y desde entonces han sido una herramienta importante en el campo del aprendizaje automático. 

Imaginemos un conjunto de datos bidimensional donde los datos de dos clases son claramente separables. Una SVM tratará de encontrar una línea recta (o en términos más generales, un hiperplano) que separe las dos clases. Esta línea no es única, pero la SVM intenta encontrar la que tenga el mayor **margen** entre los puntos más cercanos de ambas clases. Sin embargo, no siempre es posible separar las clases con un hiperplano lineal. En estos casos, una SVM utiliza el truco del kernel. Esencialmente, transforma el espacio de entrada a un espacio de dimensiones más altas donde las clases se vuelven linealmente separables.

El **margen** es la distancia entre el hiperplano de separación y los vectores de soporte más cercanos de cada clase. Una SVM busca maximizar este margen, ya que a mayor valor, mayor es el aumento de la robustez y la capacidad de generalización del modelo.

![Lógica del SVM](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/svm-logical.PNG?raw=true)

Un **kernel** es una función que toma dos entradas y las transforma en un único valor de salida. Esta función se utiliza para transformar datos que no son linealmente separables en un espacio de dimensiones más altas donde sí lo sean.

Supongamos que tenemos dos tipos de frutas en una mesa: manzanas y bananas. Si todas las manzanas están en un lado y todas las bananas en otro, podemos dibujar fácilmente una línea recta para separarlas. Pero, ¿qué sucede si están mezcladas y no podemos separarlas con una línea recta? Aquí es donde entra el kernel: imaginemos que usamos nuestra mano para golpear suavemente el centro de la mesa haciendo que las frutas salten en el aire. Mientras están en el aire (añadimos una nueva dimensión: la altura) podríamos dibujar un plano (en lugar de una línea) para separar manzanas y bananas. Después, cuando las frutas vuelvan a caer en la mesa, ese plano se traduciría en una línea curva o en una forma más compleja en la mesa que separa las frutas. El kernel es esa mano que hace saltar las frutas: transforma los datos originales a un espacio donde pueden ser separados con más facilidad.

Las SVM son herramientas poderosas y han sido utilizadas en una variedad de aplicaciones, desde clasificación de texto y reconocimiento de imágenes hasta bioinformática, incluida la clasificación de proteínas y la detección de enfermedades genéticamente predispuestas.

### Hiperparametrización del modelo

Podemos construir un SVM fácilmente en Python utilizando la librería `scikit-learn` y la función `SVC`. Algunos de sus hiperparámetros más importantes y los primeros en los que debemos centrarnos son:

- `C`: Este es el hiperparámetro de regularización. Controla el compromiso entre maximizar el margen y minimizar la clasificación errónea. Un valor pequeño para C permite un margen más amplio a expensas de algunos errores de clasificación. Un valor alto para C exige una clasificación correcta, posiblemente a expensas de un margen más estrecho.
- `kernel`: Define la función kernel que se utilizará en el algoritmo. Puede ser lineal, polinómico, RBF, etc.
- `gamma`: Solo se usa para el kernel RBF y otros. Define cuán lejos llega la influencia de un solo ejemplo de entrenamiento. Valores bajos significan influencia lejana y valores altos significan influencia cercana.
- `degree`: Se utiliza para el kernel polinómico. Es el grado del polinomio utilizado.

Otro hiperparámetro muy importante es el `random_state`, que controla la semilla de generación aleatoria. Este atributo es crucial para asegurar la replicabilidad.

### Uso del modelo en Python

Puedes fácilmente utilizando `scikit-learn` programar estos métodos posterior al EDA:

```py
from sklearn.svm import SVC

# Carga de los datos de train y test
# Estos datos deben haber sido normalizados y correctamente tratados en un EDA completo

model = SVC(kernel = "rbf", C = 1.0, gamma = 0.5)
model.fit(X_train, y_train)

y_pred = model.predict(y_test)
```
