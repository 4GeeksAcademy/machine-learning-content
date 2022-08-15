# Métricas de evaluación

**Métricas de evaluación útiles para problemas de clasificación**

1. Precisión - mide el porcentaje de tiempo en que clasificamos correctamente las muestras: (verdadero positivo + verdadero negativo)/todas las muestras.

2. Precisión - mide el porcentaje de los miembros predichos que se clasificaron correctamente: verdaderos positivos/(verdaderos positivos + falsos positivos).

3. Recall - mide el porcentaje de miembros verdaderos que el algoritmo clasificó correctamente: verdaderos positivos/(verdadero positivo + falso negativo).

4. F1 - medición que equilibra la exactitud y la precisión (o puede considerarla como un equilibrio entre el error de tipo I y el de tipo II).

5. AUC - describe la probabilidad de que un clasificador clasifique una instancia positiva elegida al azar más alta que una negativa elegida al azar.

6. Gini - una versión a escala y centrada de AUC.

7. Log-loss - similar a la precisión pero aumenta la penalización por clasificaciones incorrectas que están más alejadas de su verdadera clase. Para log-loss, los valores más bajos son mejores.

Para ver cómo implementar cada uno de ellos, consulte los ejemplos de documentación de scikit-learn:

https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

**Métricas de evaluación útiles para problemas de regresión**

1. Error cuadrático medio (MSE) - el promedio del error cuadrático de cada predicción.

2. Error cuadrático medio de la raíz (RMSE) - raíz cuadrada de MSE.

3. Error absoluto medio (MAE) - el promedio del error absoluto de cada predicción.

4. Coeficiente de determinación (R^2) - proporción de variación en el objetivo que es predecible a partir de las características.

Para ver cómo implementar cada uno de ellos, consulte los ejemplos de documentación de scikit-learn:

https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

### Evaluación del modelo

**¿Cuáles son los tres tipos de error en un modelo de Machine Learning?**

1. Bias - error causado por elegir un algoritmo que no puede modelar con precisión la señal en los datos. El modelo es demasiado general o se seleccionó incorrectamente. Por ejemplo, seleccionar una regresión lineal simple para modelar datos altamente no lineales daría como resultado un error debido al bias (sesgo).

2. Variance - error de un estimador demasiado específico y relaciones de aprendizaje que son específicas del conjunto de entrenamiento pero que no se generalizan bien a nuevas muestras. La varianza puede provenir de un ajuste demasiado cercano al ruido en los datos, y los modelos con una varianza alta son extremadamente sensibles a las entradas cambiantes. Por ejemplo, crear un árbol de decisión que divida el conjunto de entrenamiento hasta que cada nodo de hoja solo contenga 1 muestra.

3. Irreducible error - error causado por ruido en los datos que no se puede eliminar a través del modelado. Por ejemplo, la inexactitud en la recopilación de datos provoca un error irreductible.

**¿Qué es la compensación bias-variance (sesgo-varianza)?**

**Bias** se refiere a un error de un estimador que es demasiado general y no aprende relaciones de un conjunto de datos que le permitirían hacer mejores predicciones.

**Variance** se refiere a un error debido a que un estimador es demasiado específico y aprende relaciones que son específicas del conjunto de entrenamiento pero que no se generalizarán bien a nuevos registros.

En resumen, bias-variance es una compensación entre el ajuste insuficiente y el ajuste excesivo. A medida que disminuye el bias, tiende a aumentar la varianza.

Nuestro objetivo es crear modelos que minimicen el error general mediante una cuidadosa selección y ajuste del modelo para garantizar que haya un equilibrio entre el sesgo y la varianza, lo suficientemente generales como para hacer buenas predicciones sobre nuevos datos pero lo suficientemente específicos como para captar la mayor cantidad de señales posible.

Fuente:

https://towardsdatascience.com/interpreting-roc-curve-and-roc-auc-for-classification-evaluation-28ec3983f077