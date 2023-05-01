# K Vecinos más cercanos (KNN)

Antes de comenzar a aprender el algoritmo de K-vecinos más cercanos, primero debemos aclarar la diferencia entre el agrupamiento de K-medias y el algoritmo de k-vecinos más cercanos porque a menudo se confunden entre sí.

Queremos hacerle saber que la 'K' en Agrupación de K-Means no tiene nada que ver con la 'K' en el algoritmo KNN. Agrupación de K-Means es un algoritmo de aprendizaje no supervisado que se utiliza para la agrupación, mientras que KNN es un algoritmo de aprendizaje supervisado que se utiliza para la clasificación.

**¿Qué es K-vecinos más cercanos (KNN)?**

Cuando pienses en KNN, piensa en tus amigos. Eres el promedio de las personas con las que pasas más tiempo.

Cuando piense en su espacio de características, piense en él como su vecindario, donde cada punto de datos tiene un vecino.

KNN es un algoritmo simple pero muy poderoso. Clasifica el punto de datos sobre cómo se clasifica su vecino. Hace predicciones promediando los k vecinos más cercanos a un punto de datos dado. Por ejemplo, si quisiéramos predecir cuánto dinero gastaría un cliente potencial en nuestra tienda, podríamos encontrar los 5 clientes más similares a ella y promediar sus gastos para hacer la predicción.

El promedio podría ponderarse en función de la similitud entre los puntos de datos y también podría definirse la métrica de distancia de similitud.

**¿KNN es un algoritmo paramétrico o no paramétrico? ¿Se utiliza como clasificador o regresor?**

KNN no es paramétrico, lo que significa que no hacemos suposiciones sobre la distribución subyacente de sus datos, y KNN puede usarse como clasificador o regresor.

## ¿Cómo funciona KNN?

KNN hace predicciones por:

- Promedio, para tareas de regresión.

- Voto mayoritario para tareas de clasificación.

**Pasos:**

Paso 1: Determinar el valor de K.

Paso 2: Calcular las distancias entre la nueva entrada (datos de prueba) y todos los datos de entrenamiento. Las métricas más utilizadas para calcular la distancia son Euclidean, Manhattan y Minkowski.

Paso 3: Ordenar la distancia y determinar los k vecinos más cercanos en función de los valores de distancia mínima.

Paso 4: analizar la categoría de esos vecinos y asignar la categoría para los datos de prueba según el voto de la mayoría.

Paso 5: Devolver la clase predicha.

Vamos a explicar más a fondo los siguientes dos pasos:

1. Elegir la métrica de distancia adecuada. La distancia se utiliza mucho en KNN. Solo mide la distancia entre dos puntos de datos.

Pero, ¿qué métrica de distancia debo elegir?

- Distancia de Manhattan para valores continuos. Obtenemos los valores absolutos de las distancias con la fórmula $|x1 - x2| + |y1 - y2|$.

- Distancia Euclidiana para valores continuos. Es la métrica de distancia más corta y una de las más populares de elección.

- Distancia de Hamming para valores categóricos. Si nuestros dos valores están relacionados (ambos tienen 1), obtendríamos 0, lo que significa que son exactamente iguales. Si nuestra métrica de distancia es 1, no son lo mismo.

- Distancia de similitud del coseno (para vectores de palabras). Cuál es el ángulo entre dos puntos diferentes.

2. Elegir el valor de k

![knn](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/knn.jpg?raw=true)

*imagen por helloacm.com*

**¿Cómo seleccionamos el número ideal de vecinos para KNN?**

Elegir el valor correcto de K se denomina ajuste de parámetros y es necesario para obtener mejores resultados. Al elegir el valor de K, sacamos la raíz cuadrada del número total de puntos de datos disponibles en el conjunto de datos.

a. K = sqrt (número total de puntos de datos).

b. El valor impar de K siempre se selecciona para evitar confusiones entre 2 clases.

No existe una solución de forma cerrada para calcular k, por lo que a menudo se utilizan varias heurísticas. Puede ser más fácil simplemente realizar una validación cruzada y probar varios valores diferentes para k y elegir el que produce el error más pequeño durante la validación cruzada.

> A medida que aumenta k, el bias tiende a aumentar y la varianza disminuye.

![error_vs_kvalue](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/error_vs_kvalue.jpg?raw=true)

*imagen por towardsdatascience.com*

## Pros y contras

**Pros:**

- No se requiere tiempo de entrenamiento.

- Es simple y fácil de implementar.

- Se pueden agregar nuevos puntos de datos al conjunto de datos del tren en cualquier momento, ya que no se requiere el entrenamiento del modelo.

- No hay suposiciones sobre los datos, por lo que es bueno para la no linealidad.

**Contras:**

- Requerir escalado de características

- No funciona bien cuando las dimensiones son altas. Tiempo de ejecución pobre en un gran conjunto train.

- Sensible a los valores atípicos

- La predicción es computacionalmente costosa ya que necesitamos calcular la distancia entre el punto bajo consideración y todos los demás puntos.


## Sistemas de recomendación y KNN

Podemos extender este concepto de vecinos a las aplicaciones de los sistemas de recomendación. Las técnicas estadísticas de KNN nos permiten encontrar subconjuntos similares de usuarios (vecinos) para hacer recomendaciones. La idea general es que un nuevo usuario (punto de datos) se compare con todo el espacio de datos para descubrir vecinos similares. Los elementos que gustan a los vecinos se recomiendan al nuevo usuario.

![recommender_system](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/recommender_system.jpg?raw=true)

*Imagen por www.aurigait.com*

Los sistemas de recomendación se pueden dividir libremente en tres categorías: sistemas basados ​​en contenido, sistemas de filtrado colaborativo y sistemas híbridos (que usan una combinación de los otros dos).

El enfoque **basado en el contenido** utiliza una serie de características discretas de un artículo para recomendar artículos adicionales con propiedades similares.

El enfoque de **filtrado colaborativo** crea un modelo a partir de los comportamientos anteriores de un usuario (artículos comprados o seleccionados previamente y/o calificaciones numéricas dadas a esos artículos), así como decisiones similares tomadas por otros usuarios. Luego, este modelo se usa para predecir elementos (o calificaciones de elementos) en los que el usuario puede estar interesado.

![recommender_system_approaches](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/recommender_system_approaches.jpg?raw=true)

La mayoría de las empresas probablemente utilicen una combinación de ambos (enfoque híbrido) en sus sistemas de recomendación de producción.

Los sistemas de recomendación se pueden clasificar en 3 tipos:

- Recomendadores simples: ofrece recomendaciones generalizadas a cada usuario, según la popularidad y/o el género de la película. La idea básica es que las películas que son más populares tendrán una mayor probabilidad de ser del agrado de la audiencia promedio.

- Recomendadores basados ​​en contenido: sugiere artículos similares basados ​​en un artículo en particular. Este sistema utiliza metadatos de elementos, como género, director, descripción, actores, etc. para películas, para hacer estas recomendaciones. La idea general es que si a una persona le gusta un artículo en particular, también le gustará un artículo similar. Y para recomendar eso, hará uso de los metadatos de elementos anteriores del usuario. Un buen ejemplo podría ser YouTube, donde, en función de tu historial, te sugiere nuevos videos que podrías ver.

- Motores de filtrado colaborativo: estos sistemas son muy utilizados e intentan predecir la calificación o preferencia que un usuario le daría a un artículo en base a calificaciones y preferencias pasadas de otros usuarios. Los filtros colaborativos no requieren metadatos de elementos como sus contrapartes basadas en contenido.

Hemos preparado un proyecto guiado para que entiendas cómo construir un sistema de recomendación de películas muy simple.

Fuente:

https://www.dataquest.io/blog/top-10-machine-learning-algorithms-for-beginners/#:~:text=The%20first%205%20algorithms%20that,are%20examples%20of%20supervised%20learning.

https://towardsdatascience.com/how-to-find-the-optimal-value-of-k-in-knn-35d936e554eb

https://becominghuman.ai/comprehending-k-means-and-knn-algorithms-c791be90883d

https://towardsdatascience.com/knn-algorithm-what-when-why-how-41405c16c36f

https://medium.com/analytics-vidhya/a-beginners-guide-to-k-nearest-neighbor-knn-algorithm-with-code-5015ce8b227e

https://www.aurigait.com/blog/recommendation-system-using-knn/#:~:text=Collaborative%20Filtering%20using%20k%2DNearest,of%20top%2Dk%20nearest%20neighbors.
