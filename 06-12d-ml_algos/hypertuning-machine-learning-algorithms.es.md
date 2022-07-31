# Algoritmos de Machine Learning Supermejorados

### ¿Cuáles son dos formas comunes de automatizar el ajuste de hiperparámetros?

1. Grid Search - prueba todas las combinaciones posibles de valores de hiperparámetros predefinidos y seleccione la mejor.

2. Randomized Search - prueba aleatoriamente posibles combinaciones de valores de hiperparámetros predefinidos y selecciona la mejor probada.

**¿Cuáles son los pros y los contras de la búsqueda en cuadrícula?**

**Pros:**

Grid Search es excelente cuando necesitamos ajustar hiperparámetros en un pequeño espacio de búsqueda automáticamente. Por ejemplo, si tenemos 100 conjuntos de datos diferentes que esperamos que sean similares, como resolver el mismo problema repetidamente con diferentes poblaciones. Podemos usar la búsqueda en cuadrícula para ajustar automáticamente los hiperparámetros para cada modelo.

**Contras:** 

Grid Search es computacionalmente costoso e ineficiente, a menudo busca en un espacio de parámetros que tiene muy pocas posibilidades de ser útil, lo que hace que sea extremadamente lento. Es especialmente lento si necesitamos buscar en un espacio grande ya que su complejidad aumenta exponencialmente a medida que se optimizan más hiperparámetros.

**¿Cuáles son los pros y los contras de la búsqueda aleatoria?**

**Pros:**

La búsqueda aleatoria hace un buen trabajo al encontrar hiperparámetros casi óptimos en un espacio de búsqueda muy grande con relativa rapidez y no sufre el mismo problema de escala exponencial que la búsqueda en cuadrícula.

**Contras:**

La búsqueda aleatoria no ajusta los resultados tanto como lo hace la búsqueda en cuadrícula, ya que normalmente no prueba todas las combinaciones posibles de parámetros.
