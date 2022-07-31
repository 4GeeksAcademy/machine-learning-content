# Reducción de dimensionalidad

**¿Qué es la reducción de dimensionalidad?**

...

**¿Por qué querríamos usar técnicas de reducción de dimensionalidad para transformar nuestros datos antes del entrenamiento?**

La convocatoria de reducción de dimensionalidad nos permite:

- Eliminar la colinealidad del espacio de características.

- Acelerar el entrenamiento al reducir el número de características.

- Reducir el uso de la memoria al reducir la cantidad de características.

- Identificar las características latentes subyacentes que impactan múltiples características en el espacio original.

**¿Por qué querríamos evitar las técnicas de reducción de dimensionalidad para transformar nuestros datos antes del entrenamiento?**

La reducción de la dimensionalidad puede:

- Agregar cálculo adicional innecesario.

- Hacer que el modelo sea difícil de interpretar si las características latentes no son fáciles de entender.

- Agregar complejidad al modelo pipeline.

- Reducir el poder predictivo del modelo si se pierde demasiada señal.

Algunos algoritmos de reducción de dimensionalidad populares son:

1. Análisis de componentes principales (PCA) - utiliza una descomposición propia para transformar los datos de características originales en vectores propios linealmente independientes. A continuación, se seleccionan los vectores más importantes (con valores propios más altos) para representar las características en el espacio transformado.

2. Factorización de matriz no negativa (NMF) - se puede utilizar para reducir la dimensionalidad de ciertos tipos de problemas y conservar más información que PCA.

3. Técnicas de incrustación - por ejemplo, encontrar vecinos locales como se hace en la incrustación lineal local puede usarse para reducir la dimensionalidad.

4. Técnicas de agrupamiento o centroide - cada valor se puede describir como un miembro de un grupo, una combinación lineal de grupos o una combinación lineal de centroides de grupo.

Por mucho, el más popular es PCA y variaciones similares basadas en la descomposición propia.

La mayoría de las técnicas de reducción de dimensionalidad tienen transformaciones inversas, pero la señal a menudo se pierde al reducir las dimensiones, por lo que la transformación inversa suele ser solo una aproximación de los datos originales.

**¿Cómo seleccionamos el número de componentes principales necesarios para PCA?**

La selección del número de características latentes que se van a retener se realiza normalmente inspeccionando el valor propio de cada vector propio. A medida que disminuyen los valores propios, también disminuye el impacto de la característica latente en la variable objetivo.

Esto significa que los componentes principales con valores propios pequeños tienen un impacto pequeño en el modelo y pueden eliminarse.

Hay varias reglas generales, pero una regla general es incluir los componentes principales más significativos que representen al menos el 95 % de la variación en las características.
