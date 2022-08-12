# Regresión Lineal

La regresión lineal es un algoritmo de Machine Learning básico pero súper poderoso. A medida que adquiera más y más experiencia con Machine Learning, notará que lo simple es mejor que lo complejo la mayor parte del tiempo. La regresión lineal se usa ampliamente en diferentes problemas de Machine Learning supervisado y se enfoca en el problema de regresión (el valor que deseamos que la predicción sea continua).

Modela la relación entre una sola variable dependiente (variable objetivo) y una (regresión simple) o más (regresión múltiple) variables independientes. El modelo de regresión lineal asume una relación lineal entre las variables de entrada y salida. Si esta relación está presente, podemos estimar los coeficientes requeridos por el modelo para hacer predicciones sobre nuevos datos.

¿Cuáles son los cinco supuestos de regresión lineal y cómo puede verificarlos?

1. Linealidad: el objetivo (y) y las características (xi) tienen una relación lineal. Para verificar la linealidad, podemos graficar los errores contra la 'Y' pronosticada y buscar que los valores se distribuyan simétricamente alrededor de una línea horizontal con varianza constante.

2. Independencia: los errores no están correlacionados entre sí. Para verificar la independencia, podemos trazar errores a lo largo del tiempo y buscar patrones no aleatorios (para datos de series temporales).

3. Normalidad: los errores se distribuyen normalmente. Podemos verificar la normalidad trazando los errores con un histograma.

4. Homoscedasticidad: la varianza del término de error es constante a través de los valores del objetivo y las características. Para verificarlo podemos graficar los errores contra la Y predicha.

5. Sin multicolinealidad: busca correlaciones superiores a ~0,8 entre características.

## Regresión lineal simple

La regresión lineal simple es un enfoque lineal para modelar la relación entre una variable dependiente y una variable independiente, obteniendo una línea que mejor se ajuste a los datos.

$y = a + bx$

Donde X es el vector de características, y a, b son los coeficientes que deseamos aprender. a es la intersección y b es la pendiente. La intersección representa el valor de y cuando x es 0 y la pendiente indica la inclinación de la línea. El objetivo es obtener la recta que mejor se ajuste a nuestros datos (la recta que minimice la suma de los cuadrados de los errores). El error es la diferencia entre el valor real y y el valor predicho y_hat, que es el valor obtenido usando la ecuación lineal calculada.

error = y(real) - y(predicted) = y(real) - (a+bx)

En la regresión lineal, cada valor dependiente tiene una sola variable independiente correspondiente que impulsa su valor. Por ejemplo, en la fórmula de regresión lineal de $y = 3x + 7$, solo hay un resultado posible de 'y' si 'x' se define como 2.

Para los curiosos, veamos cómo podríamos implementar la regresión lineal entre dos variables desde cero:

```py

import numpy as np
import matplotlib.pyplot as plt

# Regresión lineal desde cero
import random

# Crear datos de regresión
xs = np.array(range(1,20))
ys = [0,8,10,8,15,20,26,29,38,35,40,60,50,61,70,75,80,88,96]

# Poner datos en el diccionario
data = dict()
for i in list(xs):
    data.update({xs[i-1] : ys[i-1]})

# Pendiente
m = 0
# Y interceptar
b = 0
# Tasa de aprendizaje
lr = 0.0001
# Número de épocas
epochs = 100000   #----->Especifica el número de pases completos del conjunto de datos de entrenamiento a través del algoritmo.

# Fórmula para recta lineal
def lin(x):
    return m * x + b

# Algoritmo de regresión lineal
for i in range(epochs):
    # Elejir un punto aleatorio y calcular la distancia vertical y la distancia horizontal
    rand_point = random.choice(list(data.items()))
    vert_dist = abs((m * rand_point[0] + b) - rand_point[1])
    hor_dist = rand_point[0]

    if (m * rand_point[0] + b) - rand_point[1] < 0:
        # Ajustar la línea hacia arriba
        m += lr * vert_dist * hor_dist
        b += lr * vert_dist   
    else:
        # Ajustar la línea hacia abajo
        m -= lr * vert_dist * hor_dist
        b -= lr * vert_dist
        
# Trazar puntos de datos y línea de regresión
plt.figure(figsize=(15,6))
plt.scatter(data.keys(), data.values())
plt.plot(xs, lin(xs))
plt.title('Linear Regression result')  
print('Slope: {}\nIntercept: {}'.format(m, b))

```

 Slope (pendiente): 5.096261164108786
 Intercept (Interceptar): -8.549690202482191
    
![linear-regression-result](../assets/linear-regression-result.jpg)

La regresión lineal de Scikit-learn nos facilita la implementación y optimización de modelos de regresión lineal. Puedes ver la documentación de regresión lineal de scikit-learn aquí: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

## Regresión lineal múltiple

La regresión lineal múltiple es un cálculo más específico que la regresión lineal simple. Para relaciones directas, la regresión lineal simple puede capturar fácilmente la relación entre las dos variables. Para relaciones más complejas que requieren más consideración, la regresión lineal múltiple suele ser mejor.

Una fórmula de regresión múltiple tiene varias pendientes (una para cada variable) y una intersección con el eje y. Se interpreta igual que una fórmula de regresión lineal simple, excepto que hay múltiples variables que afectan la pendiente de la relación. Debe usarse cuando múltiples variables independientes determinan el resultado de una sola variable dependiente.

$y =b₀+b₁x₁+b₂x₂+b₃x₃+…+bₙxₙ$

Las regresiones múltiples pueden ser lineales y no lineales. Las regresiones múltiples se basan en el supuesto de que existe una relación lineal entre las variables dependientes e independientes. También supone que no existe una correlación importante entre las variables independientes.

Fuente:

https://towardsdatascience.com/simple-and-multiple-linear-regression-with-python-c9ab422ec29c
