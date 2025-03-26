---
description: >-
  Discover the fundamentals of deep learning and artificial neural networks.
  Learn how these powerful models drive AI advancements today!
---
## Introduction to Deep Learning

One of the most widely used models in Machine Learning are the **Artificial Neural Networks** (*ANN*), which are models inspired by the structure and function of the human brain. 

### Definition

**Deep Learning** is a branch of Artificial Intelligence that focuses on building systems based on **Deep Neural Networks** (*DNN*). These neural networks are called "deep" because they have many layers of artificial neurons, or "nodes", that can learn and represent very complex data patterns. Simpler networks, with fewer layers, may be able to learn simpler patterns, but deep networks are capable of learning patterns that are too complex for humans to design manually.

Deep learning techniques have driven many advances in AI over the past decade, particularly in areas such as speech recognition, image recognition, natural language processing, and autonomous gaming. For example, deep learning techniques are the basis for voice recognition systems such as Siri and Alexa, Netflix and Amazon's recommendation systems, and Tesla's autonomous driving software.

Deep learning requires large amounts of data and computational power to train models efficiently. This is because deep neural networks have many parameters that need to be tuned, and these parameters are iteratively tuned through a process called backpropagation, which requires large amounts of mathematical computation.

Despite its complexity and resource requirements, deep learning has proven to be an extremely powerful tool for solving complex AI problems and is expected to continue to drive many advances in AI in the future.

### Artificial Neural Networks

An **Artificial Neural Network** (*ANN*) is a machine learning model inspired by the structure and function of the human brain. It consists of a large number of interconnected processing units called **neurons**. These neurons are organized in layers: an input layer that receives the data, one or more hidden layers that process the data, and an output layer that produces the final prediction or classification.

#### Neuron

A **neuron** is a basic processing unit of the network. It is also known as a "node" or "unit". Its name comes from the neurons in the human brain, which were the inspiration for the concept of neural networks.

![Neuron structure](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/neuron-structure.PNG?raw=true)

A neuron in a neural network takes a set of inputs ($x_1, x_2, ..., x_n$), performs a computation on them, and produces an output ($y_1$). The computation a neuron performs typically involves taking a weighted combination of the inputs (i.e., multiplying each input by a weight and summing the results), and then applying an activation function to the result. The weights ($w_i1, w_i2, ..., w_in$) are the main components that the network changes during training to learn from the data. They are adjusted so that the output of the network is as close as possible to the desired result.

The activation function introduces nonlinearity into the model. This allows the neural network to model complex relationships between inputs and outputs, beyond what it could do with only linear combinations of the inputs. Common examples of activation functions include the sigmoid function, the hyperbolic tangent function and the rectified linear unit (ReLU).

One of the most commonly used techniques for optimizing the hyperparameters of a network (which after all are the weights of the neuron connections) is one that we mentioned at the beginning of the course, the **gradient descent**. Here we analyze it in more detail.

##### Gradient descent

**Gradient descent** is an optimization algorithm used in neural networks (and many other machine learning algorithms) to minimize a cost or loss function. The cost or loss function measures how far the model's prediction is from the true value for a training data set. The goal of training a neural network is to find the values of the weights that minimize this cost function.

Gradient descent does this iteratively. It starts with random initial values for the parameters and then, at each iteration, calculates the gradient of the cost function with respect to each parameter. The gradient at a point is a vector pointing in the direction of the largest slope at that point, so moving in the opposite direction (i.e., "gradient descent") reduces the cost function. 

The algorithm then updates the parameters by moving a small amount in the direction of the negative gradient. This process is repeated until the algorithm converges to a minimum of the cost function, i.e., a point where the cost function cannot be further reduced by moving the parameters in either direction.

#### Structure

Artificial neural networks generally consist of three main types of layers: the input layer, the hidden layers and the output layer.

![Neural network structure](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/neural-network-structure.PNG?raw=true)

1. **Input layer**: This is the layer that receives the information to be processed by the neural network. Each neuron in this layer corresponds to a characteristic or attribute of the data set. For example, if you are processing 28x28 pixel images, you would have 784 input neurons, one for each pixel in the image. There can only be a single input layer in a network.
2. **Hidden layers**: These are the layers between the input layer and the output layer. The "depth" of a neural network refers to the number of these hidden layers. Each neuron in a hidden layer receives inputs from all neurons in the previous layer, processes them by applying arithmetic and logical functions, and passes the result to the next layer. Every network must have at least one of these layers.
3. **Output layer**: This is the last layer of the network and produces the final result. The nature of this layer depends on the type of task the network is designed to perform. If the network is designed for binary classification (e.g., deciding whether an image is a dog or a cat), then this layer would have a single neuron with a sigmoid activation function that produces a number between 0 and 1. If the network is designed for multiclass classification, then this layer would have as many neurons as classes, and a softmax activation function would be used to produce a probability distribution over the classes. If the network is designed for a regression task, then this layer would have a single neuron, and its output would be the predicted value.

It is usually said that when we use networks with a single hidden layer, we are doing Machine Learning, and when we use more than one hidden layer, we are doing Deep Learning.

### Deep Learning Models

Due to the variety of problems and challenges to be solved, there are a large number of models; most of them are networks of neurons that allow us to address and solve them. Some of the best known and most versatile models are:

| Model | Description | Typical use |
|:------|:------------|:------------|
| Fully Connected Neural Networks (*FCNN*) | Basic deep learning model with all neurons in one layer connected to all neurons in the next layer. | Classification, regression |
| Convolutional Neural Networks (*CNN*) | Model designed to process data with a topological grid structure, such as an image. | Image processing and classification |
| Recurrent Neural Networks (*RNN*) | Model designed to process sequences of data, taking into account the order of the data. | Natural language processing, time series | 
| Autoencoders | Model that learns to copy its inputs to its outputs. It is used to learn the representation of the data. | Dimensionality reduction, generation of new images. |
| Generative adversarial networks (*GAN*) | System of two competing neural networks: one network generates new data and the other evaluates its authenticity. | Generation of new images, super-resolution |
| Transformers | Attention-based model that processes input data in parallel rather than sequentially, improving efficiency. | Natural Language Processing (e.g., BERT, GPT) |

### Implementation

Deep learning models can be implemented using various libraries and frameworks. Some of the most popular are `TensorFlow` and `Keras`.

- **TensorFlow**: Is an open-source library for machine learning developed by Google. It provides a set of tools for developing and training Machine Learning and Deep Learning models. TensorFlow offers high and low level APIs and supports a wide range of algorithms and techniques. In addition, it can be run on multiple CPUs and GPUs, as well as on mobile devices.
- **Keras**: It is a high-level neural network API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with the goal of enabling rapid experimentation and being user-friendly, modular, and extensible. Keras offers the ability to define and train deep learning models with only a few lines of code, and is an excellent choice for beginners.

TensorFlow and Keras are two popular and powerful tools for implementing deep learning models. TensorFlow offers more flexibility and control, while Keras is easier to use and faster to prototype. Since TensorFlow version 2.0, Keras is officially integrated into TensorFlow, which means you can use the high-level features of Keras while maintaining the powerful capabilities of TensorFlow.

In addition to TensorFlow and Keras, there are other libraries and frameworks such as PyTorch, Caffe, MXNet, etc., which are also very popular for implementing deep learning models.
