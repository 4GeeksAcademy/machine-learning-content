## Introducción al Deep Learing

Uno de los modelos más utilizados en el Machine Learning son las **Redes de Neuronas Artificiales** (*ANN*, *Artificial Neural Networks*), que son modelos inspirados en la estructura y función del cerebro humano. 

### Definición

El **Aprendizaje Profundo** (*Deep Learning*) es una rama de la Inteligencia Artificial que se centra en la construcción de sistemas basados en **Redes de Neuronas Porfundas** (*DNN*, *Deep Neural Networks*). Estas redes neuronales se denominan "profundas" porque tienen muchas capas de neuronas artificiales, o "nodos", que pueden aprender y representar patrones de datos muy complejos. Las redes más sencillas, con menos capas, pueden ser capaces de aprender patrones más simples, pero las redes profundas son capaces de aprender patrones que son demasiado complejos para que los humanos los diseñen manualmente.

Las técnicas de deep learning han impulsado muchos avances en la IA en la última década, particularmente en áreas como el reconocimiento de voz, el reconocimiento de imágenes, el procesamiento del lenguaje natural, y el juego autónomo. Por ejemplo, las técnicas de deep learning son la base de los sistemas de reconocimiento de voz como Siri y Alexa, los sistemas de recomendación de Netflix y Amazon, y el software de conducción autónoma de Tesla.

El deep learning requiere grandes cantidades de datos y capacidad de cálculo para entrenar modelos eficazmente. Esto es debido a que las redes neuronales profundas tienen muchos parámetros que necesitan ser ajustados, y estos parámetros se ajustan iterativamente a través de un proceso llamado retropropagación, que requiere grandes cantidades de cálculos matemáticos.

A pesar de su complejidad y requerimientos de recursos, el deep learning ha demostrado ser una herramienta extremadamente poderosa para resolver problemas de IA complejos y se espera que siga impulsando muchos avances en la IA en el futuro.

### Redes de Neuronas Artificiales

Una **Red de Neuronas Artificiales** (*ANN*, *Artificial Neural Networks*) es un modelo de aprendizaje automático inspirado en la estructura y función del cerebro humano. Consta de un gran número de unidades de procesamiento llamadas **neuronas** (*neuron*) interconectadas. Estas neuronas se organizan en capas: una capa de entrada que recibe los datos, una o más capas ocultas que procesan los datos, y una capa de salida que produce la predicción o clasificación final.

#### Neurona

Una **neurona** (*neuron*) es una unidad de procesamiento básica de la red. También se le conoce como "nodo" o "unidad". Su nombre proviene de las neuronas en el cerebro humano, que fueron la inspiración para el concepto de redes neuronales.

![neuron-structure](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/neuron-structure.PNG?raw=true)

Una neurona en una red neuronal toma un conjunto de entradas ($x_1, x_2, ..., x_n$), realiza un cálculo con ellas y produce una salida ($y_1$). El cálculo que realiza una neurona normalmente implica tomar una combinación ponderada de las entradas (es decir, multiplicar cada entrada por un peso y sumar los resultados), y luego aplicar una función de activación al resultado. Los pesos ($w_i1, w_i2, ..., w_in$) son los principales componentes que la red cambia durante el entrenamiento para aprender de los datos. Se ajustan de manera que la salida de la red se acerque lo más posible al resultado deseado.

La función de activación introduce la no linealidad en el modelo. Esto permite que la red neuronal modele relaciones complejas entre las entradas y las salidas, más allá de lo que podría hacer con solo combinaciones lineales de las entradas. Ejemplos comunes de funciones de activación incluyen la función sigmoide, la función tangente hiperbólica y la unidad lineal rectificada (ReLU).

Una de las técnicas más utilizadas para optimizar los hiperparámetros de una red (que al fin y al cabo son los pesos de las conexiones de las neuronas) es uno que mencionamos al inicio del curso, el **descenso del gradiente** (*gradient descent*). Aquí lo analizamos con más detalle.

##### Descenso del gradiente

El **descenso de gradiente** (*gradient descent*) es un algoritmo de optimización que se utiliza en las redes neuronales (y en muchos otros algoritmos de aprendizaje automático) para minimizar una función de coste o pérdida. La función de coste o pérdida mide cuán lejos está la predicción del modelo del valor real para un conjunto de datos de entrenamiento. El objetivo de entrenar una red neuronal es encontrar los valores de los pesos que minimicen esta función de coste.

El descenso de gradiente hace esto de manera iterativa. Comienza con valores iniciales aleatorios para los parámetros y luego, en cada iteración, calcula el gradiente de la función de coste con respecto a cada parámetro. El gradiente en un punto es un vector que apunta en la dirección de la mayor pendiente en ese punto, por lo que moverse en la dirección opuesta (es decir, el "descenso de gradiente") reduce la función de coste. 

El algoritmo entonces actualiza los parámetros moviéndose una pequeña cantidad en la dirección del gradiente negativo. Este proceso se repite hasta que el algoritmo converge a un mínimo de la función de coste, es decir, un punto donde la función de coste no puede ser reducida más moviendo los parámetros en ninguna dirección.

#### Estructura

Las redes neuronales artificiales generalmente consisten en tres tipos principales de capas: la capa de entrada, las capas ocultas y la capa de salida.

![neural-network-structure](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/neural-network-structure.PNG?raw=true)

1. **Capa de entrada** (*input layer*): Esta es la capa que recibe la información que será procesada por la red neuronal. Cada neurona en esta capa corresponde a una característica o atributo del conjunto de datos. Por ejemplo, si estás procesando imágenes de 28x28 píxeles, tendrías 784 neuronas de entrada, una para cada píxel de la imagen. Sólo puede haber una única capa de entrada en una red.
2. **Capas ocultas** (*hidden layers*): Estas son las capas entre la capa de entrada y la capa de salida. La "profundidad" de una red neuronal se refiere al número de estas capas ocultas. Cada neurona en una capa oculta recibe entradas de todas las neuronas de la capa anterior, las procesa aplicando funciones aritméticas y lógicas y pasa el resultado a la siguiente capa. Toda red debe tener al menos una de estas capas.
3. **Capa de salida** (*output layer*): Esta es la última capa de la red y produce el resultado final. La naturaleza de esta capa depende del tipo de tarea que la red esté diseñada para realizar. Si la red está diseñada para la clasificación binaria (por ejemplo, decidir si una imagen es un perro o un gato), entonces esta capa tendría una sola neurona con una función de activación sigmoide que produce un número entre 0 y 1. Si la red está diseñada para la clasificación multiclase, entonces esta capa tendría tantas neuronas como clases y se utilizaría una función de activación softmax para producir una distribución de probabilidad sobre las clases. Si la red está diseñada para una tarea de regresión, entonces esta capa tendría una sola neurona y su salida sería el valor predicho.

Normalmente se dice que cuando utilizamos redes con una única capa oculta estamos haciendo Machine Learning, y cuando utilizamos más de una capa oculta estamos haciendo Deep Learning.






## Deep learning techniques

There are various Deep Learning models which are used to solve complicated tasks. Each of them is used for specific tasks, with certain processes and limitations. The following deep learning techniques is a very good summary taken from Heartbeat (https://heartbeat.comet.ml/), an online publication community that provides educational resources for data science, machine learning, and deep learning practitioners.

### Multilayer Perceptrons (MLPs)

Multilayer Perceptrons is a feedforward artificial neural network, where a set of inputs are fed into the Neural Network to generate a set of outputs. MLPs are made up of input layers and an output layer that is fully connected.

When to use it?

- When your dataset is in a tabular format consisting of rows and columns. Typically CSV files

- Can be used for both Classification and Regression tasks where a set of ground truth values are given as the input.

### Convolutional Neural Network (CNN)

Convolutional Neural Network (CNN), also known as a ConvNet is a feed-forward Neural Network. It is typically used in image recognition, by processing pixelated data to detect and classify objects in an image. The model has been built to solve complex tasks, preprocessing, and data compilation.

When to use it?

- It works very well with Image Datasets An example of this is OCR document analysis, which recognizes text within a digital image.

- Ideally, the input data is a 2-dimensional field. However, it can also be converted into a 1-dimensional to make the process faster.

- This technique should also be used if the model requires high complexity in calculating the output.

### Recurrent Neural Networks (RNNs)

A Recurrent Neural Network is used to work for time series data or data that involve sequences. RNN uses the previous state’s knowledge and uses it as an input value for the current prediction.

Therefore, RNN can memorize previous inputs using its internal memory. They are used for time-series analysis, handwriting recognition, Natural Language Processing, and more.

When to use it? There are 4 different ways that you can use Recurrent Neural Networks:

1. One to one: a single input that produces a single output. An example of this is Image Classification

2. One to many: a single input that produces a sequence of outputs. An example of this is Image captioning, where a variety of words are detected from a single image

3. Many to one: a sequence of inputs that produces a single output. An example of this is Sentiment Analysis

4. Many to many: a sequence of inputs that produces a sequence of outputs. An example of this is Video classification, where you split the video into frames and label each frame separately

**Long Short Term Memory Networks (LSTMs)**

Long Short Term Memory Networks is a type of Recurrent Neural Network which can learn and memorize long-term dependencies. Its default behavior and the aim of an LSTM is to remember past information for long periods.

### Generative Adversarial Networks (GANs)

Generative Adversarial Networks use two neural networks which compete with one another, hence the “adversarial” in the name.

The two neural networks used to build a GAN are called ‘the generator’ and ‘the discriminator’. The Generator learns to generate fake data whilst the Discriminator learns from that fake information. They are used to ensure accuracy in the model’s predictions.

When to use it?

- Image inpainting — you can do this by restoring missing parts of images.

- Image super-resolution — you can do this by upscaling low-resolution images to high resolution.

- If you want to create data from images to texts.

### Restricted Boltzmann Machines (RBMs)

Restricted Boltzmann machine is a type of Recurrent Neural Network where the nodes make binary decisions with some bias. It was invented by Geoffrey Hinton and is used generally for dimensionality reduction, classification, regression, feature learning, and topic modeling.

RBMs uses two layers:

- Visible units

- Hidden units

The visible and hidden units have biases connected. The visible units are connected to the hidden units, and they do not have any output nodes.

When to use it?

- As the Boltzmann Machine will learn to regulate, this technique will be good to use when monitoring a system.

- It is efficient when you are building a binary recommendation system

- It is also used specifically when using a very specific dataset.

## Top applications of deep learning in 2022

Deep Learning has recently outperformed humans in solving particular tasks, for example, Image Recognition. The level of accuracy achieved by Deep Learning has made it become so popular, and everybody is figuring out ways to implement it into their business. Different applications include:

- Self driving cars

- News aggregation and fraud news detection

- Natural Language Processing

- Virtual Assistants

- Visual Recognition

- Fraud detection

- Personalized experiences in e-commerce

- Detecting developmental delay in children

- Healthcare: from medical imaging, discovering new drugs, clinical research.

- Colourisation of Black and White images

- Adding sounds to silent movies

- Automatic Machine Translation

- Automatic Handwriting Generation

- Automatic Game Playing

- Language Translations

- Pixel Restoration

- Photo Descriptions

- Demographic and Election Predictions

- Deep Dreaming

## Software tools

The following graph shows the top 16 open-source deep learning libraries used by Github contributors.

![deep_learning_tools](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/deep_learning_tools.jpg?raw=true)

*Image by KDnuggets*

By all measures, TensorFlow is the undisputed leader, followed by Keras.

**TensorFlow** 

Tensorflow is a Python open-source library for fast numerical computing created and released by Google. It was designed for use both in research and development and in production systems. It can run on single CPU systems and GPUs, as well as mobile devices and large-scale distributed systems of hundreds of machines.

Computation is described in terms of data flow and operations in the structure of a directed graph.

- Nodes: Nodes perform computation and have zero or more inputs and outputs. Data that moves between nodes are known as **tensors**, which are multi-dimensional arrays of real values.

- Edges: The graph defines the flow of data, branching, looping, and updates to state. Special edges can be used to synchronize behavior within the graph, for example, waiting for computation on a number of inputs to complete.

- Operation: An operation is a named abstract computation that can take input attributes and produce output attributes. For example, you could define an add or multiply operation.

**Keras** 

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. Keras is designed for human beings, not machines, by minimizing the number of user actions required for common use cases. It is also the most used deep learning framework among top-5 winning teams on Kaggle.

To sum up the relationship between Tensorflow and Keras, TensorFlow is an open-sourced end-to-end platform, a library for multiple machine learning tasks, while Keras is a high-level neural network library that runs on top of TensorFlow. Both provide high-level APIs used for easily building and training models, but Keras is more user-friendly because it's built-in Python.

The best place to start is with the user-friendly Keras sequential API. Let's see the following example taken from the Tensorflow website:

```bash
pip install tensorflow
```

```py
#import Tensorflow

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

#load the MNIST dataset. Convert the sample data from integers to floating-point numbers.

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#build a tf.keras.Sequential model by stacking layers

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

# For each example, the model returns a vector of logits or log-odds scores, one for each class.

predictions = model(x_train[:1]).numpy()
predictions

# The tf.nn.softmax function converts these logits to probabilities for each class:

tf.nn.softmax(predictions).numpy()

# Define a loss function for training using losses.SparseCategoricalCrossentropy, which takes a vector of logits and a True index and returns a scalar loss for each example.

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# This loss is equal to the negative log probability of the true class: The loss is zero if the model is sure of the correct class.
# This untrained model gives probabilities close to random (1/10 for each class), so the initial loss should be close to -tf.math.log(1/10) ~= 2.3.

loss_fn(y_train[:1], predictions).numpy()

# Before you start training, configure and compile the model using Keras Model.compile. Set the optimizer class to adam, set the loss to the loss_fn function you defined earlier, and specify a metric to be evaluated for the model by setting the metrics parameter to accuracy.

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Use the Model.fit method to adjust your model parameters and minimize the loss:

model.fit(x_train, y_train, epochs=5)

# The Model.evaluate method checks the models performance, usually on a "Validation-set" or "Test-set".

model.evaluate(x_test,  y_test, verbose=2)

```

The image classifier is now trained to ~98% accuracy on this dataset. If you want your model to return a probability, you can wrap the trained model, and attach the softmax to it:

```py
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])
```

You have trained a machine learning model using a prebuilt dataset using the Keras API! See the exploring-neural-networks notebook to see another example in Keras!


Source: 

https://www.deeplearningbook.org/

https://machinelearningmastery.com/inspirational-applications-deep-learning/

https://heartbeat.comet.ml/deep-learning-techniques-you-should-know-in-2022-94f33e62d922

https://machinelearningmastery.com/introduction-python-deep-learning-library-tensorflow/

https://medium.com/intro-to-artificial-intelligence/deep-learning-series-1-intro-to-deep-learning-abb1780ee20

https://medium.com/@genvill/9-ways-to-become-the-macgyver-of-deep-learning-4f0133bc1f14

https://becominghuman.ai/face-detection-with-opencv-and-deep-learning-90b84735f421

https://machinelearningmastery.com/linear-algebra-cheat-sheet-for-machine-learning/

https://www.mygreatlearning.com/blog/deep-learning-applications/

http://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.88210&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false

https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/quickstart/beginner.ipynb#scrollTo=4mDAAPFqVVgn

https://www.tensorflow.org/tutorials/load_data/images

https://www.tensorflow.org/tutorials

https://heartbeat.comet.ml/deep-learning-techniques-you-should-know-in-2022-94f33e62d922
