# Aprendizaje no Supervisado

Los modelos de aprendizaje no supervisados ​​se utilizan cuando solo tenemos las variables de entrada (X) y no tenemos las variables de salida correspondientes. Utilizan datos de entrenamiento sin etiquetar para modelar la estructura subyacente de los datos.

## CLUSTERING

**¿Qué es clustering (agrupación)?**

Clustering se utiliza para agrupar muestras de modo que los objetos dentro del mismo grupo sean más similares entre sí que con los objetos de otro grupo.

**¿Cuáles son los objetivos de clustering?**

- Queremos que las observaciones en diferentes grupos sean diferentes entre sí

- Homogeneidad: Queremos que las observaciones en el mismo grupo sean similares entre sí.

- Encontrar agrupaciones naturales.

Queremos minimizar la varianza dentro de un grupo.

**¿Cuáles son las entradas y salidas?**

Las entradas serán un conjunto de entradas numéricas escaladas normalmente y sin valores atípicos.

Los resultados serán un conjunto de etiquetas, una para cada observación, y también un conjunto de centroides, uno para cada conglomerado.

**Para lograr un buen agrupamiento, necesitas:**

- Elijir la métrica de distancia correcta.

- Tener una buena intuición detrás de tus datos.

### K-MEANS

K-means clustering es un algoritmo de agrupamiento no supervisado que toma un grupo de puntos no etiquetados e intenta agruparlos en un número "k" de grupos/clusters, donde cada punto del grupo es similar entre sí.

Cluster es una colección de objetos similares que son diferentes a los demás.

La "k" en k-means denota la cantidad de clusters que deseas tener al final. Si k = 5, tendrás 5 clusters en el conjunto de datos.

Las medias de los clusters generalmente se aleatorizan al principio (a menudo eligiendo observaciones aleatorias de los datos) y luego se actualizan a medida que se observan más registros.

En cada iteración, se asigna una nueva observación a un cluster en función de la media del cluster más cercano y luego se vuelven a calcular o actualizar las medias, con la nueva información de observación incluida.

**¿Cuáles son los casos de uso comunes para la agrupación en clusters de k-means?**

La segmentación de clientes es probablemente el caso de uso más común para la agrupación en clusters de k-means.

También se utiliza para encontrar grupos en tipos de quejas, tipos de comportamientos de los consumidores, resúmenes de datos, encontrar anomalías, por ejemplo, detección de fraude, y la lista continúa.

> Las anomalías pueden considerarse pequeños clusters con puntos muy alejados de cualquier centroide.

**¿Cómo funciona k-means?**

**Paso 1:** Determinar el valor K por el método Elbow y especificar el número de clusters K

**Paso 2:** Asignar aleatoriamente cada punto de datos a un clsuter.

**Paso 3:** Determinar las coordenadas del centroide del grupo.

¿Qué es un centroide?

Simple, es el punto central de un cluster. Por ejemplo, si queremos encontrar 3 clusters, entonces tendríamos 3 centroides (o centros), uno para cada cluster.

**Paso 4:** Determinar las distancias de cada punto de datos a los centroides y reasignar cada punto al centroide del cluster más cercano según la distancia mínima. A menor distancia, mayor similitud. A mayor distancia, menor similitud.

**Paso 5:** Calcular de nuevo los centroides del cluster.

**Paso 6:** Repetir los pasos 4 y 5 hasta que alcancemos un nivel óptimo global en el que no sea posible realizar mejoras ni cambiar los puntos de datos de un cluster a otro.

**Métricas de distancia comunes:**

- Distancia Euclidiana: La distancia se puede definir como una línea recta entre dos puntos (métrica de distancia más común).

- Distancia Manhattan: La distancia entre dos puntos es la suma de las diferencias (absolutas) de sus coordenadas.

- Distancia del coseno

**Código de ejemplo:**

```py
# Importar funciones k-means y vq
from scipy.cluster.vq import kmeans, vq

# Cluster de centros de cómputo
centroids,_ = kmeans(df, 2)

# Asignar etiquetas de cluster
df['cluster_labels'], _ = vq(df, centroids)

# Trazar los puntos con seaborn
sns.scatterplot(x='x', y='y', hue='cluster_labels', data=df)
plt.show()
```

## Reducción de dimensionalidad

**¿Qué es la reducción de dimensionalidad?**

La reducción de dimensionalidad se utiliza para reducir el número de variables de un conjunto de datos y, al mismo tiempo, garantizar que aún se transmita información importante. La reducción de la dimensionalidad se puede realizar mediante métodos de extracción de características y métodos de selección de características. Feature Selection selecciona un subconjunto de las variables originales. La extracción de características realiza la transformación de datos de un espacio de alta dimensión a un espacio de baja dimensión. Ejemplo: el algoritmo PCA es un enfoque de extracción de características.

**¿Por qué querríamos usar técnicas de reducción de dimensionalidad para transformar nuestros datos antes del entrenamiento?**

La reducción de la dimensionalidad nos permite:

- Eliminar la colinealidad del espacio de características.

- Acelerar el entrenamiento al reducir la cantidad de funciones.

- Reducir el uso de memoria al reducir la cantidad de funciones.

- Identificar características latentes subyacentes que impactan múltiples características en el espacio original.

**¿Por qué querríamos evitar las técnicas de reducción de dimensionalidad para transformar nuestros datos antes del entrenamiento?**

La reducción de la dimensionalidad puede:

- Agregar cómputo adicional innecesario.

- Hacer que el modelo sea difícil de interpretar si las características latentes no son fáciles de entender.

- Agregar complejidad al pipeline del modelo.

- Reducir el poder predictivo del modelo si se pierde demasiada señal.

**Algoritmos de reducción de dimensionalidad populares**

1. Análisis de componentes principales (PCA) - utiliza una descomposición propia para transformar los datos de características originales en vectores propios linealmente independientes. A continuación, se seleccionan los vectores más importantes (con valores propios más altos) para representar las características en el espacio transformado.

2. Factorización de matriz no negativa (NMF) - se puede usar para reducir la dimensionalidad de ciertos tipos de problemas y conservar más información que PCA.

3. Técnicas de incrustación - por ejemplo, encontrar vecinos locales como se hace en la incrustación lineal local puede usarse para reducir la dimensionalidad.

4. Técnicas de clustering  o centroide - cada valor se puede describir como un miembro de un cluster, una combinación lineal de clusters o una combinación lineal de centroides de cluster.

Con mucho, el más popular es PCA y variaciones similares basadas en la descomposición propia.

La mayoría de las técnicas de reducción de dimensionalidad tienen transformaciones inversas, pero la señal a menudo se pierde al reducir las dimensiones, por lo que la transformación inversa suele ser solo una aproximación de los datos originales.

### PCA 

El análisis de componentes principales (PCA) es una técnica no supervisada que se utiliza para preprocesar y reducir la dimensionalidad de conjuntos de datos de alta dimensión mientras se preserva la estructura original y las relaciones inherentes al conjunto de datos original para que los modelos de machine learning aún puedan aprender de ellos y usarse para hacer predicciones.

**¿Cómo seleccionamos el número de componentes principales necesarios para PCA?**

La selección del número de características latentes que se van a retener se realiza normalmente mediante la inspección del valor propio de cada vector propio. A medida que disminuyen los valores propios, también disminuye el impacto de la característica latente en la variable objetivo.

Esto significa que los componentes principales con valores propios pequeños tienen un impacto pequeño en el modelo y pueden eliminarse.

Hay varias reglas generales, pero una regla general es incluir los componentes principales más significativos que representen al menos el 95% de la variación en las características.

Fuente:

https://becominghuman.ai/comprehending-k-means-and-knn-algorithms-c791be90883d

https://www.dataquest.io/blog/top-10-machine-learning-algorithms-for-beginners/#:~:text=The%20first%205%20algorithms%20that,are%20examples%20of%20supervised%20learning.