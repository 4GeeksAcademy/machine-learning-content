---
description: >-
  Descubre el mundo del Deep Learning y las Redes de Neuronas Artificiales.
  Aprende cómo estas tecnologías transforman la IA y resuelven problemas
  complejos.
---
## Introducción al Deep Learning

Uno de los modelos más utilizados en el Machine Learning son las **Redes de Neuronas Artificiales** (*ANN*, *Artificial Neural Networks*), que son modelos inspirados en la estructura y función del cerebro humano. 

### Definición

El **Aprendizaje Profundo** (*Deep Learning*) es una rama de la Inteligencia Artificial que se centra en la construcción de sistemas basados en **Redes de Neuronas Profundas** (*DNN*, *Deep Neural Networks*). Estas redes neuronales se denominan "profundas" porque tienen muchas capas de neuronas artificiales, o "nodos", que pueden aprender y representar patrones de datos muy complejos. Las redes más sencillas, con menos capas, pueden ser capaces de aprender patrones más simples, pero las redes profundas son capaces de aprender patrones que son demasiado complejos para que los humanos los diseñen manualmente.

Las técnicas de deep learning han impulsado muchos avances en la IA en la última década, particularmente en áreas como el reconocimiento de voz, el reconocimiento de imágenes, el procesamiento del lenguaje natural, y el juego autónomo. Por ejemplo, las técnicas de deep learning son la base de los sistemas de reconocimiento de voz como Siri y Alexa, los sistemas de recomendación de Netflix y Amazon, y el software de conducción autónoma de Tesla.

El deep learning requiere grandes cantidades de datos y capacidad de cálculo para entrenar modelos eficazmente. Esto es debido a que las redes neuronales profundas tienen muchos parámetros que necesitan ser ajustados, y estos parámetros se ajustan iterativamente a través de un proceso llamado retropropagación, que requiere grandes cantidades de cálculos matemáticos.

A pesar de su complejidad y requerimientos de recursos, el deep learning ha demostrado ser una herramienta extremadamente poderosa para resolver problemas de IA complejos y se espera que siga impulsando muchos avances en la IA en el futuro.

### Redes de Neuronas Artificiales

Una **Red de Neuronas Artificiales** (*ANN*, *Artificial Neural Networks*) es un modelo de aprendizaje automático inspirado en la estructura y función del cerebro humano. Consta de un gran número de unidades de procesamiento llamadas **neuronas** (*neuron*) interconectadas. Estas neuronas se organizan en capas: una capa de entrada que recibe los datos, una o más capas ocultas que procesan los datos, y una capa de salida que produce la predicción o clasificación final.

#### Neurona

Una **neurona** (*neuron*) es una unidad de procesamiento básica de la red. También se le conoce como "nodo" o "unidad". Su nombre proviene de las neuronas en el cerebro humano, que fueron la inspiración para el concepto de redes neuronales.

![Estructura de neuronas](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/neuron-structure.PNG?raw=true)

Una neurona en una red neuronal toma un conjunto de entradas ($x_1, x_2, ..., x_n$), realiza un cálculo con ellas y produce una salida ($y_1$). El cálculo que realiza una neurona normalmente implica tomar una combinación ponderada de las entradas (es decir, multiplicar cada entrada por un peso y sumar los resultados), y luego aplicar una función de activación al resultado. Los pesos ($w_i1, w_i2, ..., w_in$) son los principales componentes que la red cambia durante el entrenamiento para aprender de los datos. Se ajustan de manera que la salida de la red se acerque lo más posible al resultado deseado.

La función de activación introduce la no linealidad en el modelo. Esto permite que la red neuronal modele relaciones complejas entre las entradas y las salidas, más allá de lo que podría hacer con solo combinaciones lineales de las entradas. Ejemplos comunes de funciones de activación incluyen la función sigmoide, la función tangente hiperbólica y la unidad lineal rectificada (ReLU).

Una de las técnicas más utilizadas para optimizar los hiperparámetros de una red (que al fin y al cabo son los pesos de las conexiones de las neuronas) es uno que mencionamos al inicio del curso, el **descenso del gradiente** (*gradient descent*). Aquí lo analizamos con más detalle.

##### Descenso del gradiente

El **descenso de gradiente** (*gradient descent*) es un algoritmo de optimización que se utiliza en las redes neuronales (y en muchos otros algoritmos de aprendizaje automático) para minimizar una función de coste o pérdida. La función de coste o pérdida mide cuán lejos está la predicción del modelo del valor real para un conjunto de datos de entrenamiento. El objetivo de entrenar una red neuronal es encontrar los valores de los pesos que minimicen esta función de coste.

El descenso de gradiente hace esto de manera iterativa. Comienza con valores iniciales aleatorios para los parámetros y luego, en cada iteración, calcula el gradiente de la función de coste con respecto a cada parámetro. El gradiente en un punto es un vector que apunta en la dirección de la mayor pendiente en ese punto, por lo que moverse en la dirección opuesta (es decir, el "descenso de gradiente") reduce la función de coste. 

El algoritmo entonces actualiza los parámetros moviéndose una pequeña cantidad en la dirección del gradiente negativo. Este proceso se repite hasta que el algoritmo converge a un mínimo de la función de coste, es decir, un punto donde la función de coste no puede ser reducida más moviendo los parámetros en ninguna dirección.

#### Estructura

Las redes neuronales artificiales generalmente consisten en tres tipos principales de capas: la capa de entrada, las capas ocultas y la capa de salida.

![Estructura de red neuronal](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/neural-network-structure.PNG?raw=true)

1. **Capa de entrada** (*input layer*): Esta es la capa que recibe la información que será procesada por la red neuronal. Cada neurona en esta capa corresponde a una característica o atributo del conjunto de datos. Por ejemplo, si estás procesando imágenes de 28x28 píxeles, tendrías 784 neuronas de entrada, una para cada píxel de la imagen. Solo puede haber una única capa de entrada en una red.
2. **Capas ocultas** (*hidden layers*): Estas son las capas entre la capa de entrada y la capa de salida. La "profundidad" de una red neuronal se refiere al número de estas capas ocultas. Cada neurona en una capa oculta recibe entradas de todas las neuronas de la capa anterior, las procesa aplicando funciones aritméticas y lógicas y pasa el resultado a la siguiente capa. Toda red debe tener al menos una de estas capas.
3. **Capa de salida** (*output layer*): Esta es la última capa de la red y produce el resultado final. La naturaleza de esta capa depende del tipo de tarea que la red esté diseñada para realizar. Si la red está diseñada para la clasificación binaria (por ejemplo, decidir si una imagen es un perro o un gato), entonces esta capa tendría una sola neurona con una función de activación sigmoide que produce un número entre 0 y 1. Si la red está diseñada para la clasificación multiclase, entonces esta capa tendría tantas neuronas como clases y se utilizaría una función de activación softmax para producir una distribución de probabilidad sobre las clases. Si la red está diseñada para una tarea de regresión, entonces esta capa tendría una sola neurona y su salida sería el valor predicho.

Normalmente, se dice que cuando utilizamos redes con una única capa oculta estamos haciendo Machine Learning, y cuando utilizamos más de una capa oculta estamos haciendo Deep Learning.

### Modelos de aprendizaje profundo

Debido a la variedad de problemas y retos a resolver, existe una gran cantidad de modelos, la mayoría de ellos son redes de neuronas, que permiten abordarlos y resolverlos. Algunos de los modelos más conocidos y versátiles son:

| Modelo | Descripción | Uso típico |
|:-------|:------------|:-----------|
| Redes Neuronales Totalmente Conectadas (*FCNN*) | Modelo básico de deep learning con todas las neuronas de una capa conectadas a todas las neuronas de la capa siguiente. | Clasificación, regresión |
| Redes Neuronales Convolucionales (*CNN*) | Modelo diseñado para procesar datos con una estructura de cuadrícula topológica, como una imagen. | Procesamiento y clasificación de imágenes |
| Redes Neuronales Recurrentes (*RNN*) | Modelo diseñado para procesar secuencias de datos, teniendo en cuenta el orden de los datos. | Procesamiento del lenguaje natural, series temporales |
| Autoencoders | Modelo que aprende a copiar sus entradas a sus salidas. Se utiliza para aprender la representación de los datos. | Reducción de la dimensionalidad, generación de nuevas imágenes |
| Redes Generativas Adversativas (*GAN*) | Sistema de dos redes neuronales que compiten entre sí: una red genera nuevos datos y la otra evalúa su autenticidad. | Generación de nuevas imágenes, superresolución |
| Transformers | Modelo basado en la atención que procesa los datos de entrada en paralelo en lugar de secuencialmente, mejorando la eficiencia. | Procesamiento del lenguaje natural (por ejemplo, BERT, GPT) |

### Implementación

Los modelos de aprendizaje profundo se pueden implementar utilizando diversas bibliotecas y marcos de trabajo. Algunas de las más populares son `TensorFlow` y `Keras`.

- **TensorFlow**: Es una biblioteca de código abierto para el aprendizaje automático desarrollada por Google. Proporciona un conjunto de herramientas para desarrollar y entrenar modelos de Machine Learning y Deep Learning. TensorFlow ofrece APIs de alto y bajo nivel y soporta una amplia gama de algoritmos y técnicas. Además, puede ser ejecutado en múltiples CPUs y GPUs, así como en dispositivos móviles.
- **Keras**: Es una API de redes neuronales de alto nivel, escrita en Python y capaz de correr sobre TensorFlow, CNTK, o Theano. Fue desarrollada con el objetivo de permitir una experimentación rápida y ser amigable, modular y extensible. Keras ofrece la capacidad de definir y entrenar modelos de deep learning con solo unas pocas líneas de código, y es una excelente opción para los principiantes.

TensorFlow y Keras son dos herramientas populares y poderosas para la implementación de modelos de aprendizaje profundo. TensorFlow ofrece más flexibilidad y control, mientras que Keras es más fácil de usar y rápido para prototipar. Desde la versión 2.0 de TensorFlow, Keras está oficialmente integrado en TensorFlow, lo que significa que puedes usar las características de alto nivel de Keras mientras mantienes las poderosas capacidades de TensorFlow.

Además de TensorFlow y Keras, existen otras bibliotecas y marcos de trabajo como PyTorch, Caffe, MXNet, etc., que también son muy populares para la implementación de modelos de deep learning.
