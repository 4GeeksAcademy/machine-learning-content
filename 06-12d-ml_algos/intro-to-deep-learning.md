# Intro to Deep Learning

Deep learning is a sub-field of machine learning dealing with algorithms inspired by the structure and function of the brain called artificial neural networks. Deep learning algorithms are similar to how nervous system is structured, where each neuron is connected to each other and passing information.

Deep learning models tend to perform well with amount of data wheras old machine learning models stop improving after a saturation point. One of the differences between machine learning and deep learning models is on the feature extraction area. Feature extraction is done by human in machine learning whereas deep learning model figure out by itself.

![why_deep_learning](../assets/why_deep_learning.jpg)

*Image by Andrew Ng at https://www.slideshare.net/ExtractConf*

**What are activation functions?**

Let's imagine an example about a model that decides if a student is accepted to a university based on its grades. Activation functions are functions that decide, given the inputs into the node, what should be the node’s output, so in this case the activation function decides, given the grades, if the student is accepted or not. 

One way in which neurons process inputs is by using linear combinations of weighted inputs (i.e., linear neuron). Another one is using a logistic function which returns a value between 0 and 1. Other activation functions you’ll see are tanh, and softmax functions.

**What are weights?**

When input data comes into a neuron, it gets multiplied by a weight value that is assigned to this particular input. These weights start out as random values, and as the neural network learns more about what kind of input data leads to a student being accepted into a university, the network adjusts the weights based on any errors in categorization that the previous weights resulted in. This is called training the neural network. Training neural networks can be viewed as a sort of optimization problem. The goal is to minimize the error of neural nets through constant training. 

To figure out how we’re going to find these weights, start by thinking about the goal. We want the network to make predictions as close as possible to the real values. To measure this, we need a metric of how wrong the predictions are, the error. A common metric is the sum of the squared errors (SSE), where y^ is the prediction and y is the true value, and you take the sum over all output units j and another sum over all data points μ.

The SSE is a good choice for a few reasons. The square ensures the error is always positive and larger errors are penalized more than smaller errors. Our goal is to find weights wij that minimize the squared error. To do this with a neural network, typically we use gradient descent. 

Do you remember our 'Algorithm Optimization' module? We saw a little bit of Gradient Descent.

Gradient descent is an optimization algorithm used to find the values of parameters (coefficients) of a function (f) that minimizes a cost function (cost). Gradient descent is best used when the parameters cannot be calculated analytically (e.g. using linear algebra) and must be searched for by an optimization algorithm. It uses derivatives and calculus techniques through back propagation, resilient propagation and Manhattan propagation and other optimization techniques based on gradient calculations.

Algebra and calculus are the foundation of the deep neural nets. 

**Neural networks**

The diagram below shows a simple network. Weights and biases are the learnable parameters. In the following equation: $y = mx+b$ 'm' (the slope) would be the weight and 'b' would represent bias. The linear combination of the weights, inputs, and bias form the input h, which passes through the activation function f(h), giving the final output, labeled y.

![neural_network](../assets/neural_network.jpg)

The activation function, f(h) can be any function. If we let f(h)=h, the output will be the same as the input. Now the output of the network is $y=Σi WiXi + b$

**Forward and back propagation**

By propagating values from the first layer (the input layer) through all the mathematical functions represented by each node, the network outputs a value. This process is called a **forward pass**.

You forward propagateto get the output and compare it with the real value to get the error. Now, to minimise the error, you propagate backwards by finding the derivative of error with respect to each weight and then subtracting this value from the weight value. This is called **back propagation**.

Now let's see the code for implementing the entire propagation:

```py
import numpy as np
from data_prep import features, targets, features_test, targets_test

np.random.seed(21)

def sigmoid(x):
    """
    Calculate sigmoid
    """
    return 1 / (1 + np.exp(-x))


# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 900
learnrate = 0.005

n_records, n_features = features.shape
last_loss = None
# Initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** .5,
                                         size=n_hidden)

for e in range(epochs):
    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # TODO: Calculate the output
        hidden_input = np.dot(x, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)

        output = sigmoid(np.dot(hidden_output,
                                weights_hidden_output))

        ## Backward pass ##
        # TODO: Calculate the network's prediction error
        error = y - output

        # TODO: Calculate error term for the output unit
        output_error_term = error * output * (1 - output)

        ## propagate errors to hidden layer

        # TODO: Calculate the hidden layer's contribution to the error
        hidden_error = np.dot(output_error_term, weights_hidden_output)

        # TODO: Calculate the error term for the hidden layer
        hidden_error_term = hidden_error * hidden_output * (1 - hidden_output)

        # TODO: Update the change in weights
        del_w_hidden_output += output_error_term * hidden_output
        del_w_input_hidden += hidden_error_term * x[:, None]

    # TODO: Update weights
    weights_input_hidden += learnrate * del_w_input_hidden / n_records
    weights_hidden_output += learnrate * del_w_hidden_output / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        hidden_output = sigmoid(np.dot(x, weights_input_hidden))
        out = sigmoid(np.dot(hidden_output,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
```

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

![deep_learning_tools](../assets/deep_learning_tools.jpg)

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
