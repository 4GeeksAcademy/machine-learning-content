# Modelos Lineales Regularizados

Antes de explicar los modelos lineales regularizados, recapitulemos información importante sobre la regresión lineal.

Cada problema de machine learning es básicamente un problema de optimización. Es decir, desea encontrar un máximo o un mínimo de una función específica. La función que desea optimizar generalmente se denomina función de pérdida (o función de costo). La función de pérdida se define para cada algoritmo de machine learning que usa, y esta es la métrica principal para evaluar la precisión de tu modelo entrenado.

Esta es la forma más básica de pérdida para un punto de datos específico, que se usa principalmente para algoritmos de regresión lineal:

$l = ( Ŷi- Yi)^2$

Donde:

- Ŷi es el valor predicho

- Yi es el valor real

La función loss (pérdida) en su conjunto se puede denotar como:

$L = ∑( Ŷi- Yi)^2$

Esta función de pérdida, en particular, se llama pérdida cuadrática o mínimos cuadrados. Deseamos minimizar la función de pérdida (L) tanto como sea posible para que la predicción sea lo más cercana posible a la realidad fundamental.

> Recuerda, cada algoritmo de machine learning define su propia función de pérdida de acuerdo con su objetivo en la vida.

## Superar el sobreajuste con la regularización

Terminamos la última lección hablando de la importancia de evitar el sobreajuste. Uno de los mecanismos más comunes para evitar el sobreajuste se denomina regularización. El modelo de machine learning regularizado es un modelo en el que su función de pérdida contiene otro elemento que también debe minimizarse. Veamos un ejemplo:

$L = ∑( Ŷi- Yi)^2 + λ∑ β2$

1. **Regresión de cresta**: regresión lineal que agrega el término de penalización/regularización de la norma L2 a la función de costo. El parámetro λ es un escalar que también debe aprenderse mediante la validación cruzada. Un hecho muy importante que debemos notar sobre la regresión de cresta es que obliga a que los coeficientes β sean más bajos, pero no obliga a que sean cero. Es decir, no eliminará las características irrelevantes, sino que minimizará su impacto en el modelo entrenado.

2. **Lasso**: regresión lineal que agrega el término de penalización/regularización de la norma L1 a la función de costo. La única diferencia con la regresión de cresta es que el término de regularización está en valor absoluto. Pero esta diferencia tiene un gran impacto en la compensación que hemos discutido antes. El método Lasso supera la desventaja de la regresión de cresta al no solo castigar los valores altos de los coeficientes β, sino también al establecerlos en cero si no son relevantes. Por lo tanto, es posible que termine con menos funciones incluidas en el modelo que las que tenía al principio, lo cual es una gran ventaja.

3. **Red elástica**: regresión lineal que agrega una combinación de términos de penalización de norma L1 y L2 a la función de costo.

## ¿Qué hiperparámetros se pueden ajustar en modelos lineales regularizados?

Puedes ajustar el peso del término de regularización para los modelos regularizados (generalmente indicado como alfa), que afecta la cantidad de características que los modelos comprimirán.

- alpha = 0 ---> El modelo regularizado es idéntico al modelo original.

- alpha = 1 ---> modelo regularizado redujo el modelo original a un valor constante.

**Rendimiento de los modelos regularizados**

Los modelos regularizados tienden a superar a los modelos lineales no regularizados, por lo que se sugiere que al menos intente usar la regresión de cresta.

Lasso puede ser efectivo cuando desea realizar automáticamente la selección de funciones para crear un modelo más simple, pero puede ser peligroso ya que puede ser errático y eliminar funciones que contienen señales útiles.

La red elástica es un equilibrio entre la cresta y el lasso, y se puede utilizar con el mismo efecto que el lasso con un comportamiento menos errático.


Fuente:

https://medium.com/hackernoon/practical-machine-learning-ridge-regression-vs-lasso-a00326371ece