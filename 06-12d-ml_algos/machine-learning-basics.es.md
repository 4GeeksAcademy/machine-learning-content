# Básicos de Machine Learning 

¿Qué es Machine Learning? 

Machine learning es el campo de la ciencia que estudia algoritmos que aproximan funciones cada vez mejor a medida que se les dan más observaciones. Los algoritmos de Machine learning a menudo se utilizan para aprender y automatizar procesos humanos, optimizar resultados, predecir resultados, modelar relaciones complejas y aprender patrones en los datos.

**Aprendizaje supervisado vs no supervisado**

Aprendizaje supervisado...

Los datos etiquetados son datos que tienen la información sobre la variable de destino para cada instancia. Los usos más comunes del aprendizaje supervisado son la regresión y la clasificación.

Aprendizaje sin supervisión...

Los usos más comunes del aprendizaje automático no supervisado son la agrupación en clústeres, la reducción de la dimensionalidad y la minería de reglas de asociación.

**Aprendizaje en línea vs fuera de línea**

El aprendizaje en línea se refiere a la actualización gradual de los modelos a medida que obtienen más información.

El aprendizaje fuera de línea se refiere al aprendizaje por datos de procesamiento por lotes. Si ingresan nuevos datos, se debe ingresar un lote nuevo completo (incluidos todos los datos antiguos y nuevos) en el algoritmo para aprender de los nuevos datos.

**Aprendizaje reforzado**

El aprendizaje por refuerzo describe un conjunto de algoritmos que aprenden del resultado de cada decisión. Por ejemplo, un robot podría usar el aprendizaje por refuerzo para aprender que caminar hacia adelante contra una pared es malo, pero alejarse de una pared y caminar es bueno. 

**¿Cuál es la diferencia entre embolsar y aumentar?**

El embolsado y el aumento son métodos de conjunto, lo que significa que combinan muchos predictores débiles para crear un predictor fuerte. Una diferencia clave es que el embolsado construye modelos independientes en paralelo, mientras que el impulso construye modelos secuencialmente, en cada paso enfatizando las observaciones que se perdieron en los pasos anteriores.

### ¿Cómo se dividen los datos?

¿Qué son los datos de entrenamiento y para qué sirven?

Los datos de entrenamiento son un conjunto de ejemplos que se utilizarán para entrenar el modelo de Machine Learning. Para Machine Learning supervisado, estos datos de entrenamiento deben tener una etiqueta. Lo que está tratando de predecir debe ser definido.

Para Machine Learning no supervisado, los datos de entrenamiento contendrán solo características y no usarán objetivos etiquetados. Lo que está tratando de predecir no está definido.

¿Qué es un conjunto de validación y por qué utilizar uno?

Un conjunto de validación es un conjunto de datos que se utiliza para evaluar el rendimiento de un modelo durante el entrenamiento/selección del modelo. Una vez que se entrenan los modelos, se evalúan en el conjunto de validación para seleccionar el mejor modelo posible.

Nunca debe usarse para entrenar el modelo directamente.

Tampoco debe usarse como el conjunto de datos de prueba porque hemos sesgado nuestra selección de modelo para que funcione bien con estos datos, incluso si el modelo no se entrenó directamente en él.

¿Qué es un equipo de prueba y por qué utilizar uno?

Un conjunto de prueba es un conjunto de datos que no se utilizan durante el entrenamiento o la validación. El rendimiento del modelo se evalúa en el conjunto de prueba para predecir qué tan bien se generalizará a nuevos datos.

### Técnicas de entrenamiento y validación

**Train-test-split**

**Validación Cruzada**

La validación cruzada es una técnica para entrenar y validar modelos con mayor precisión. Rota qué datos se retienen del entrenamiento del modelo para usarlos como datos de validación.

Se entrenan y evalúan varios modelos, y cada pieza de datos se obtiene de un modelo. A continuación, se calcula el rendimiento medio de todos los modelos.

Es una forma más confiable de validar modelos, pero es más costosa desde el punto de vista computacional. Por ejemplo, la validación cruzada de 5 veces requiere entrenar y validar 5 modelos en lugar de 1.

**¿Qué es el sobreajuste?**

Sobreajuste cuando un modelo hace predicciones mucho mejores sobre datos conocidos (datos incluidos en el conjunto de entrenamiento) que sobre datos desconocidos (datos no incluidos en el conjunto de entrenamiento).

¿Cómo podemos combatir el sobreajuste?

Algunas formas de combatir el sobreajuste son:

- Simplificar el modelo (a menudo se hace cambiando).

- Seleccionar un modelo diferente.

- Usar más datos de entrenamiento.

- Recopilar datos de mejor calidad para combatir el sobreajuste.

¿Cómo podemos saber si nuestro modelo está sobreajustando los datos?

Si nuestro error de entrenamiento es bajo y nuestro error de validación es alto, lo más probable es que nuestro modelo esté sobreajustando nuestros datos de entrenamiento.

¿Cómo podemos saber si nuestro modelo no se ajusta bien a los datos?

Si nuestro error de entrenamiento y validación son relativamente iguales y muy altos, lo más probable es que nuestro modelo no se ajuste bien a nuestros datos de entrenamiento.

**¿Qué son los datos de pipelines?**

Cualquier colección de transformaciones ordenadas en datos.

Fuente:
    
https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

https://www.kdnuggets.com/2020/09/understanding-bias-variance-trade-off-3-minutes.html

https://medium.com/@ranjitmaity95/7-tactics-to-combat-imbalanced-classes-in-machine-learning-datase-4266029e2861

