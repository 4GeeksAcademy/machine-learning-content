# Regresión

En la regresión, se utilizan una o más variables (predictores) para predecir un resultado (criterio). 

Regresión lineal:

¿Cuáles son los cinco supuestos de regresión lineal y cómo puede verificarlos?

1. Linealidad: el objetivo (y) y las características (xi) tienen una relación lineal. Para verificar la linealidad, podemos graficar los errores contra la 'Y' pronosticada y buscar que los valores se distribuyan simétricamente alrededor de una línea horizontal con varianza constante.

2. Independencia: los errores no están correlacionados entre sí. Para verificar la independencia, podemos trazar errores a lo largo del tiempo y buscar patrones no aleatorios (para datos de series temporales).

3. Normalidad: los errores se distribuyen normalmente. Podemos verificar la normalidad trazando los errores con un histograma.

4. Homoscedasticidad: la varianza del término de error es constante a través de los valores del objetivo y las características. Para verificarlo podemos graficar los errores contra la y predicha.

5. Sin multicolinealidad: busca correlaciones superiores a ~0,8 entre características.


```python
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
# y interceptar
b = 0
# Tasa de aprendizaje
lr = 0.0001
# Número de épocas
epochs = 100000

# Fórmula para línea lineal
def lin(x):
    return m * x + b

# Algoritmo de regresión lineal
for i in range(epochs):
    # Elije un punto aleatorio y calcula la distancia vertical y la distancia horizontal
    rand_point = random.choice(list(data.items()))
    vert_dist = abs((m * rand_point[0] + b) - rand_point[1])
    hor_dist = rand_point[0]

    if (m * rand_point[0] + b) - rand_point[1] < 0:
        # Ajusta la línea hacia arriba
        m += lr * vert_dist * hor_dist
        b += lr * vert_dist   
    else:
        # Ajusta la línea hacia abajo
        m -= lr * vert_dist * hor_dist
        b -= lr * vert_dist
        
# Traza puntos de datos y línea de regresión
plt.figure(figsize=(15,6))
plt.scatter(data.keys(), data.values())
plt.plot(xs, lin(xs))
plt.title('Linear Regression result')  
print('Slope: {}\nIntercept: {}'.format(m, b))
```

Fuente:

https://towardsdatascience.com/simple-and-multiple-linear-regression-with-python-c9ab422ec29c
