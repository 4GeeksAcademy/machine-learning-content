# Ingeniería de Características

La ingeniería de características es una parte esencial de la construcción de cualquier sistema inteligente. Esta es la razón por la que los Científicos de Datos y los Ingenieros de Machine learning suelen dedicar el 70 % de su tiempo a la fase de preparación de datos antes del modelado.

![density_curve.jpg](../assets/ml_pipeline.jpg)

*Una pipeline (tubería) de Machine Learning estándar (fuente: Machine Learning práctico con Python, Apress/Springer)*

Específicamente, aprenderemos cómo modificar las variables del conjunto de datos para extraer información significativa con el fin de capturar la mayor cantidad de información posible, dejando los conjuntos de datos y sus variables listos para usarse en algoritmos de  Machine Learning.

**¿Qué es una característica?**

Una característica suele ser una representación específica sobre los datos sin procesar, que es un atributo individual y medible, generalmente representado por una columna en un conjunto de datos. Considerando un conjunto de datos bidimensional genérico, cada observación se representa mediante una fila y cada característica mediante una columna, que tendrá un valor específico para una observación. Dicho esto, una característica es una variable, que podemos explicar como cualquier característica, número o cantidad que se pueda medir o contar. Las llamamos variables porque los valores que toman pueden variar.

Ejemplos de variables pueden ser:

Edad ( 10,15,17,21,30,...).

País (USA, Thailand, Japan, Argentina,...).

Uso de energía (220, 50, 130, 88,...).

Clasificamos las variables en un conjunto de datos en uno de estos tipos principales:
 
- Variables numéricas.

- Variables categóricas.

- Variables de fecha y hora.

- Variables mixtas.

Las características pueden ser de dos tipos principales según el conjunto de datos. Las características sin procesar inherentes se obtienen directamente del conjunto de datos sin manipulación o ingeniería de datos adicionales. Las características derivadas generalmente se obtienen de la ingeniería de características, donde extraemos características de los atributos de datos existentes. Un ejemplo simple sería crear una nueva característica "IMC" a partir de un conjunto de datos de empleados que contenga “Weight” (Peso) y "Height"(Altura), simplemente usando la fórmula con peso y altura.

**¿Qué es la ingeniería de características?**

La ingeniería de características es el proceso de utilizar el conocimiento del dominio de datos para crear y transformar características o variables que hacen que los algoritmos de Machine Learning funcionen de manera más eficiente. Es una tarea fundamental para mejorar el rendimiento del modelo de Machine Learning y la precisión de la predicción.

La ingeniería de características puede llevar mucho tiempo porque incluye una serie de procesos, como:

- Rellenar valores faltantes dentro de una variable.

- Crear o extraer nuevas características de las disponibles en tu conjunto de datos.

- Codificación de variables categóricas en números.

- Transformación de variables.

**¿Por qué necesitamos hacer ingeniería de características?**

Cada vez que comenzamos un nuevo proyecto de Machine Learning, ya sea que recibamos un conjunto de datos sin procesar o hagamos web scraping para obtener los datos, estos seguramente estarán desordenados y no serán adecuados para entrenar un modelo. Siempre debemos realizar una exploración de datos al principio para encontrar valores vacíos, valores atípicos, tipos de datos, relaciones, etc. Una vez que comprendamos mejor los datos que tenemos, podemos comenzar a realizar tareas de ingeniería de características para crear modelos de alto rendimiento. El éxito de un algoritmo a menudo puede depender de cómo diseñamos las características de entrada.

**No confundas la ingeniería de características con la selección de características**. La selección de características nos permite seleccionar características del grupo de características (incluidas las de nueva ingeniería) que ayudarán a los modelos de Machine Learning a hacer predicciones sobre las variables de destino de manera más eficiente. En una pipeline típica de Machine Learning, realizamos la selección de características después de completar la ingeniería de características.

### Ingeniería de Características en Datos Numéricos

Aunque los datos numéricos se pueden introducir directamente en los modelos de Machine Learning, aún necesitamos diseñar características que sean relevantes para el problema antes de construir un modelo.

Una forma de medidas sin procesar incluye características que representan frecuencias, conteos o ocurrencias de atributos específicos.

Ahora, si estoy creando un sistema de recomendación para recomendaciones de canciones, solo me gustaría saber si una persona está interesada o ha escuchado una canción en particular. Esto no requiere la cantidad de veces que se ha escuchado una canción, ya que estoy más preocupado por las diversas canciones que el/ella ha escuchado. En este caso, se prefiere una función binaria en lugar de una función basada en recuento.

El problema de trabajar con características numéricas continuas sin procesar es que, a menudo, la distribución de valores en estas características estará sesgada. Esto significa que algunos valores ocurrirán con bastante frecuencia, mientras que otros serán bastante raros.

El agrupamiento, también conocido como cuantización, se utiliza para transformar características numéricas continuas en discretas (categorías). Estos valores o números discretos se pueden considerar como categorías o bins (contenedores) en los que se agrupan los valores numéricos continuos sin procesar. Cada bin representa un grado específico de intensidad y, por lo tanto, un rango específico de valores numéricos continuos cae en él.

Un ejemplo de esto sería crear bins por grupos de edad, como el siguiente ejemplo:

![feature_engineering_numerical.jpg](../assets/feature_engineering_numerical.jpg)

El histograma anterior que muestra las edades de los desarrolladores está ligeramente sesgado hacia la derecha como se esperaba (desarrolladores de menor edad).

Código de ejemplo:

```py
data = pd.DataFrame({'Age':[0,2,4,13,35,-1,54]})
​
bins= [0,2,4,13,20,110]
labels = ['Infant','Toddler','Kid','Teen','Adult']
data['AgeGroup'] = pd.cut(data['Age'], bins=bins, labels=labels, right=False)
print (data)
```

### Ingeniería de características en datos categóricos

Por lo general, cualquier atributo de datos de naturaleza categórica representa valores discretos que pertenecen a un conjunto finito específico de categorías o clases. Estos valores discretos pueden ser de naturaleza textual o numérica (¡o incluso datos no estructurados como imágenes!). Hay dos clases principales de datos categóricos, nominales y ordinales.

- Considera un ejemplo simple de categorías de color. Digamos que tenemos cinco clases o categorías principales en este escenario particular sin ningún concepto o noción de orden (el amarillo no siempre aparece antes que el rojo ni es más pequeño o más grande que el rojo). Como no existe el concepto de ordenamiento, este sería un atributo categórico nominal.

- Los atributos categóricos ordinales tienen algún sentido o noción de orden entre sus valores. Los tamaños, el nivel educativo y los roles laborales son otros ejemplos de atributos categóricos ordinales.

Normalmente, las características categóricas deben convertirse en números para que el modelo de Machine Learning las entienda. Veremos las técnicas de codificación de etiquetas en una lección específica.

Si queremos ver la lista de etiquetas únicas en cualquier característica categórica, podemos usar el siguiente código:

```py
music_genres = np.unique(df['Genre'])
music_genres
```

o podemos hacerlo directamente en nuestro dataframe así:

```py
df.column.unique()
```


### ¿Cuáles son algunas técnicas ingenuas de ingeniería de características que mejoran la eficacia del modelo?

1. Estadísticas de resumen (media, mediana, moda, mín, máx, estándar) para cada grupo de registros similares. Por ejemplo, todas las clientas entre 34 y 45 años obtendrían su propio conjunto de estadísticas resumidas.

2. Interacciones o proporciones entre características. Por ejemplo, var1 * var2.

3. Resúmenes de características. Por ejemplo, la cantidad de compras que realizó un cliente en los últimos 30 días (las características sin procesar pueden ser las últimas 10 fechas de compra).

4. Dividir información de características manualmente. Por ejemplo, los clientes que miden más de 1,80 cm serían un dato fundamental a la hora de recomendar un coche frente a un SUV.

5. KNN usa registros en el conjunto de entrenamiento para producir una característica KNN que se alimenta a otro modelo.

Fuente:

https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b

https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63

https://medium.com/@mxcsyounes/hands-on-with-feature-engineering-techniques-broad-introduction-def389c1fc25
