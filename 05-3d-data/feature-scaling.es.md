# Escalado de características

Siguiendo los pasos de preprocesamiento de datos, el escalado de datos es un método de estandarización que es útil cuando se trabaja con un conjunto de datos que contiene características continuas que están en diferentes escalas, y estás usando un modelo que opera en algún tipo de espacio lineal (como lineal regresión o K-vecinos más cercanos).

¿Qué significa con diferentes escalas?

La mayoría de las veces, nuestro conjunto de datos contendrá características que varían mucho en magnitudes, unidades y rango. Pero dado que la mayoría de los algoritmos de Machine Learning usan la distancia Euclidiana entre dos puntos de datos en sus cálculos, esto es un problema. Si se dejan solos, estos algoritmos solo toman en cuenta la magnitud de las características y descuidan las unidades. Los resultados variarían mucho entre diferentes unidades, 5 kg y 5000 g. Las características con magnitudes altas pesarán mucho más en los cálculos de distancia que las características con magnitudes bajas.

Así que imagina que estamos viendo los precios de algunos productos tanto en Yenes como en Dólares Estadounidenses. Un Dólar Estadounidense vale aproximadamente 100 Yenes, pero si no escalamos nuestros precios, algoritmos como SVM o KNN considerarán una diferencia de precio de 1 Yen tan importante como una diferencia de 1 Dólar Estadounidense.

Recuerde, escalar significa transformar los datos para que se ajusten a una escala específica, como 0-100 o 0-1. Generalmente 0-1. Queremos escalar los datos, especialmente cuando usamos métodos basados ​​en medidas de qué tan separados están los puntos de datos.

## Métodos para escalar los datos

Aprenderemos cómo implementar diferentes transformaciones de escala usando funciones integradas que vienen con el paquete scikit-learn.

Además de admitir funciones de biblioteca, otras funciones que se utilizarán para lograr la funcionalidad son:

- El método de ajuste (datos) se usa para calcular la media y la desviación estándar para una función determinada, de modo que se pueda usar más para escalar.

- El método transform(data) se utiliza para realizar el escalado utilizando la media y la desviación estándar calculadas mediante el método .fit().

- El método fit_transform() ajusta y transforma.

#### 1. Normalización o escalado

Se refiere a transformar nuestros datos de entrada a una nueva escala. Esta distribución tendrá valores entre -1 y 1 con μ=0. El normalizador escala cada valor dividiendo cada valor por su magnitud en un espacio n-dimensional para n número de características.

**Ejemplo:**

```py
#import 
from sklearn import preprocessing

#scale the data
scaler = preprocessing.Normalizer()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=df.columns)

```

**Drawback:** Es sensible a los valores atípicos, ya que la presencia de valores atípicos comprimirá la mayoría de los valores y hará que aparezcan extremadamente juntos.

#### 2. Min-Max

Otra forma de escalar datos, que es una transformación lineal de datos que asigna el valor mínimo a 0 y el valor máximo a 1. El mínimo de característica se hace igual a cero y el máximo de característica igual a uno. MinMax Scaler reduce los datos dentro del rango dado, generalmente de 0 a 1. Transforma los datos escalando las características a un rango determinado. Escala los valores a un rango de valores específico sin cambiar la forma de la distribución original.

El escalado MinMax se realiza usando:

```py
x_std = (x – x.min(axis=0)) / (x.max(axis=0) – x.min(axis=0))

x_scaled = x_std * (max – min) + min
```

Donde:

- min, max = feature_range.

- x.min(axis=0) : Valor mínimo de característica.

- x.max(axis=0): Valor máximo de característica.

Sklearn preprocessing defines MinMaxScaler() method to achieve this.

**Ejemplo:**

```py

# Otra forma de importar el módulo
from sklearn.preprocessing import MinMaxScaler
 
# Funciones de escala
scaler = MinMaxScaler()
model=scaler.fit(data)
scaled_data=model.transform(data)

```

Esencialmente reduce el rango de tal manera que el rango ahora está entre 0 y 1 (o -1 a 1 si hay valores negativos).

Este escalador funciona mejor para los casos en los que el escalador estándar podría no funcionar tan bien. Si la distribución no es gaussiana o la desviación estándar es muy pequeña, el escalador min-max funciona mejor.

#### 3. RobustScaler

El RobustScaler usa un método similar al escalador Min-Max, pero en su lugar usa el rango intercuartílico, en lugar del mínimo-máximo, por lo que es resistente a los valores atípicos. Por lo tanto, sigue la siguiente fórmula para cada característica:

```py
xi–Q1(x) / Q3(x)–Q1(x)
```

Por supuesto, esto significa que está utilizando menos datos para escalar, por lo que es más adecuado cuando hay valores atípicos en los datos.

**Ejemplo:**

```py

# Importar
from sklearn import preprocessing

# Escala de datos

scaler = preprocessing.RobustScaler()
robust_scaled_df = scaler.fit_transform(x)
robust_scaled_df = pd.DataFrame(robust_scaled_df, columns=['x1', 'x2'])

```

#### 4. Estandarización (o transformación de puntuación Z)

Transforma cada característica en una distribución normal con una media de 0 y una desviación estándar de 1. Reemplaza los valores por su puntaje z.

Standard Scaler ayuda a obtener una distribución estandarizada, con una media cero y una desviación estándar de uno (varianza unitaria). Estandariza las características restando el valor medio de la característica y luego dividiendo el resultado por la desviación estándar de la característica.

La escala estándar se calcula como:

```py
z = (x - u) / s
```

Donde:

- Z son datos escalados.

- X se va a escalar datos.

- U es la media de las muestras de entrenamiento

- S es la desviación estándar de las muestras de entrenamiento.

El preprocesamiento de Sklearn es compatible con el método StandardScaler() para lograr esto directamente en solo 2 o 3 pasos.

**Ejemplo:**

```py
# Módulo de importación
from sklearn.preprocessing import StandardScaler
 
# Escala de datos
scaler = StandardScaler()
model = scaler.fit(data)
scaled_data = model.transform(data)

```

**Drawback:** Se vuelve a escalar a un intervalo ilimitado que puede ser problemático para ciertos algoritmos. Por ejemplo, algunas redes neuronales que esperan que los valores de entrada estén dentro de un rango específico.

La Estandarización y la Normalización Media se pueden usar para algoritmos que asumen datos centrados en cero como el Análisis de Componentes Principales (PCA).

## ¿Cuándo debemos escalar los datos? ¿Por qué?

Cuando nuestro algoritmo ponderará cada entrada. Por ejemplo, el descenso de gradiente utilizado por muchas redes neuronales o el uso de métricas de distancia como KNN.

El rendimiento del modelo a menudo se puede mejorar mediante la normalización, estandarización o, de otro modo, el escalado de los datos para que cada característica tenga un peso relativamente igual.

También es importante cuando las características se miden en diferentes unidades. Por ejemplo, la característica A se mide en pulgadas, la característica B se mide en pies y la característica C se mide en dólares, que están escaladas de manera que se ponderan y/o representan por igual.

En algunos casos, la eficacia no cambiará, pero la importancia de la característica percibida puede cambiar. Por ejemplo, los coeficientes en una regresión lineal.

Por lo general, escalar nuestros datos no cambia el rendimiento ni la importancia de las características de los modelos basados ​​en árboles, ya que los puntos de división simplemente cambiarán para compensar los datos escalados.

> Regla general: ¡cualquier algoritmo que calcule la distancia o asuma normalidad, escala tus características!

**Algunos algoritmos donde el escalado de características es importante:**

- K-vecinos más cercanos: deben escalarse para que todas las características tengan el mismo peso.

- Análisis de Componentes Principales.

- Descenso de gradiente.

- Los modelos basados ​​en árboles no son modelos basados ​​en la distancia y pueden manejar diferentes rangos de características. No es necesario escalar al modelar árboles.

- Naive Bayes y Linear Discriminant Analysis otorgan pesos a las características en consecuencia, por lo que la escala de características puede no tener mucho efecto.

Fuentes:

https://towardsai.net/p/data-science/scaling-vs-normalizing-data-5c3514887a84

https://medium.com/greyatom/why-how-and-when-to-scale-your-features-4b30ab09db5e

https://benalexkeen.com/feature-scaling-with-scikit-learn/

https://www.kaggle.com/code/poonaml/titanic-survival-prediction-end-to-end-ml-pipeline
