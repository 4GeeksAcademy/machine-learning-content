# Exploring KERAS

This notebook is taken from the Machine Learning Mastery tutorials (machinelearningmastery.com). It covers step by step how to model your first neural network using Keras!

We are going to use the Pima Indians diabetes dataset. This is a standard machine learning dataset from the UCI Machine Learning repository. It describes patient medical record data for Pima Indians and whether they had an onset of diabetes within five years.

As such, it is a binary classification problem (onset of diabetes as 1 or not as 0). All of the input variables that describe each patient are numerical. This makes it easy to use directly with neural networks that expect numerical input and output values, and ideal for our first neural network in Keras.

**Understanding the data**

Input Variables (X):

-Number of times pregnant

-Plasma glucose concentration a 2 hours in an oral glucose tolerance test

-Diastolic blood pressure (mm Hg)

-Triceps skin fold thickness (mm)

-2-Hour serum insulin (mu U/ml)

-Body mass index (weight in kg/(height in m)^2)

-Diabetes pedigree function

-Age (years)

Output Variables (y):

Class variable (0 or 1)

**Requirements:**

- Python 2 or 3 installed.

- SciPy (including NumPy) installed.

- Keras and a backend (Theano or TensorFlow) installed.

**Step 1: Install libraries**


```python
pip install scipy tensorflow
```

    Requirement already satisfied: scipy in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (1.8.1)
    Collecting tensorflow
      Downloading tensorflow-2.9.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (511.7 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m511.7/511.7 MB[0m [31m3.7 MB/s[0m eta [36m0:00:00[0mm eta [36m0:00:01[0m[36m0:00:01[0m
    [?25hRequirement already satisfied: numpy<1.25.0,>=1.17.3 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from scipy) (1.23.1)
    Collecting tensorflow-io-gcs-filesystem>=0.23.1
      Downloading tensorflow_io_gcs_filesystem-0.26.0-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (2.4 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.4/2.4 MB[0m [31m119.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting opt-einsum>=2.3.2
      Downloading opt_einsum-3.3.0-py3-none-any.whl (65 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m65.5/65.5 kB[0m [31m24.2 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: six>=1.12.0 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from tensorflow) (1.16.0)
    Collecting protobuf<3.20,>=3.9.2
      Downloading protobuf-3.19.4-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m135.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting astunparse>=1.6.0
      Downloading astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
    Collecting termcolor>=1.1.0
      Downloading termcolor-1.1.0.tar.gz (3.9 kB)
      Preparing metadata (setup.py) ... [?25ldone
    [?25hRequirement already satisfied: typing-extensions>=3.6.6 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from tensorflow) (4.2.0)
    Requirement already satisfied: setuptools in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from tensorflow) (62.3.2)
    Collecting flatbuffers<2,>=1.12
      Downloading flatbuffers-1.12-py2.py3-none-any.whl (15 kB)
    Requirement already satisfied: packaging in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from tensorflow) (21.3)
    Collecting google-pasta>=0.1.1
      Downloading google_pasta-0.2.0-py3-none-any.whl (57 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m57.5/57.5 kB[0m [31m21.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting h5py>=2.9.0
      Downloading h5py-3.7.0-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (4.5 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.5/4.5 MB[0m [31m147.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting libclang>=13.0.0
      Downloading libclang-14.0.1-py2.py3-none-manylinux1_x86_64.whl (14.5 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m14.5/14.5 MB[0m [31m110.3 MB/s[0m eta [36m0:00:00[0mm eta [36m0:00:01[0m0:01[0m
    [?25hCollecting keras<2.10.0,>=2.9.0rc0
      Downloading keras-2.9.0-py2.py3-none-any.whl (1.6 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m105.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting grpcio<2.0,>=1.24.3
      Downloading grpcio-1.47.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.5 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.5/4.5 MB[0m [31m139.8 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting tensorboard<2.10,>=2.9
      Downloading tensorboard-2.9.1-py3-none-any.whl (5.8 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.8/5.8 MB[0m [31m141.3 MB/s[0m eta [36m0:00:00[0mm eta [36m0:00:01[0m
    [?25hCollecting tensorflow-estimator<2.10.0,>=2.9.0rc0
      Downloading tensorflow_estimator-2.9.0-py2.py3-none-any.whl (438 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m438.7/438.7 kB[0m [31m107.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting gast<=0.4.0,>=0.2.1
      Downloading gast-0.4.0-py3-none-any.whl (9.8 kB)
    Collecting absl-py>=1.0.0
      Downloading absl_py-1.2.0-py3-none-any.whl (123 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m123.4/123.4 kB[0m [31m35.7 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: wrapt>=1.11.0 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from tensorflow) (1.14.1)
    Collecting keras-preprocessing>=1.1.1
      Downloading Keras_Preprocessing-1.1.2-py2.py3-none-any.whl (42 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m42.6/42.6 kB[0m [31m15.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: wheel<1.0,>=0.23.0 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from astunparse>=1.6.0->tensorflow) (0.37.1)
    Collecting tensorboard-data-server<0.7.0,>=0.6.0
      Downloading tensorboard_data_server-0.6.1-py3-none-manylinux2010_x86_64.whl (4.9 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.9/4.9 MB[0m [31m163.7 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: requests<3,>=2.21.0 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from tensorboard<2.10,>=2.9->tensorflow) (2.27.1)
    Collecting tensorboard-plugin-wit>=1.6.0
      Downloading tensorboard_plugin_wit-1.8.1-py3-none-any.whl (781 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m781.3/781.3 kB[0m [31m134.7 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting markdown>=2.6.8
      Downloading Markdown-3.4.1-py3-none-any.whl (93 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m93.3/93.3 kB[0m [31m30.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting werkzeug>=1.0.1
      Downloading Werkzeug-2.2.1-py3-none-any.whl (232 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m232.4/232.4 kB[0m [31m50.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting google-auth<3,>=1.6.3
      Downloading google_auth-2.9.1-py2.py3-none-any.whl (167 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m167.8/167.8 kB[0m [31m33.4 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting google-auth-oauthlib<0.5,>=0.4.1
      Downloading google_auth_oauthlib-0.4.6-py2.py3-none-any.whl (18 kB)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from packaging->tensorflow) (3.0.9)
    Collecting cachetools<6.0,>=2.0.0
      Downloading cachetools-5.2.0-py3-none-any.whl (9.3 kB)
    Collecting pyasn1-modules>=0.2.1
      Downloading pyasn1_modules-0.2.8-py2.py3-none-any.whl (155 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m155.3/155.3 kB[0m [31m55.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting rsa<5,>=3.1.4
      Downloading rsa-4.9-py3-none-any.whl (34 kB)
    Collecting requests-oauthlib>=0.7.0
      Downloading requests_oauthlib-1.3.1-py2.py3-none-any.whl (23 kB)
    Requirement already satisfied: importlib-metadata>=4.4 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from markdown>=2.6.8->tensorboard<2.10,>=2.9->tensorflow) (4.11.3)
    Requirement already satisfied: certifi>=2017.4.17 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (2022.5.18.1)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (1.26.9)
    Requirement already satisfied: idna<4,>=2.5 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (3.3)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (2.0.12)
    Requirement already satisfied: MarkupSafe>=2.1.1 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from werkzeug>=1.0.1->tensorboard<2.10,>=2.9->tensorflow) (2.1.1)
    Requirement already satisfied: zipp>=0.5 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.10,>=2.9->tensorflow) (3.8.0)
    Collecting pyasn1<0.5.0,>=0.4.6
      Downloading pyasn1-0.4.8-py2.py3-none-any.whl (77 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m77.1/77.1 kB[0m [31m28.5 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting oauthlib>=3.0.0
      Downloading oauthlib-3.2.0-py3-none-any.whl (151 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m151.5/151.5 kB[0m [31m52.8 MB/s[0m eta [36m0:00:00[0m
    [?25hBuilding wheels for collected packages: termcolor
      Building wheel for termcolor (setup.py) ... [?25ldone
    [?25h  Created wheel for termcolor: filename=termcolor-1.1.0-py3-none-any.whl size=4832 sha256=8eab63bca203b568831e2860e46d14f29fadf95538b229aa5da7d86d0c95a2e7
      Stored in directory: /home/gitpod/.cache/pip/wheels/a0/16/9c/5473df82468f958445479c59e784896fa24f4a5fc024b0f501
    Successfully built termcolor
    Installing collected packages: termcolor, tensorboard-plugin-wit, pyasn1, libclang, keras, flatbuffers, werkzeug, tensorflow-io-gcs-filesystem, tensorflow-estimator, tensorboard-data-server, rsa, pyasn1-modules, protobuf, opt-einsum, oauthlib, keras-preprocessing, h5py, grpcio, google-pasta, gast, cachetools, astunparse, absl-py, requests-oauthlib, markdown, google-auth, google-auth-oauthlib, tensorboard, tensorflow
    Successfully installed absl-py-1.2.0 astunparse-1.6.3 cachetools-5.2.0 flatbuffers-1.12 gast-0.4.0 google-auth-2.9.1 google-auth-oauthlib-0.4.6 google-pasta-0.2.0 grpcio-1.47.0 h5py-3.7.0 keras-2.9.0 keras-preprocessing-1.1.2 libclang-14.0.1 markdown-3.4.1 oauthlib-3.2.0 opt-einsum-3.3.0 protobuf-3.19.4 pyasn1-0.4.8 pyasn1-modules-0.2.8 requests-oauthlib-1.3.1 rsa-4.9 tensorboard-2.9.1 tensorboard-data-server-0.6.1 tensorboard-plugin-wit-1.8.1 tensorflow-2.9.1 tensorflow-estimator-2.9.0 tensorflow-io-gcs-filesystem-0.26.0 termcolor-1.1.0 werkzeug-2.2.1
    [33mWARNING: There was an error checking the latest version of pip.[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.


To use Keras, you will need to have the TensorFlow package installed. 

Once TensorFlow is installed, just import Keras. We will use the NumPy library to load our dataset and we will use two classes from the Keras library to define our model.

**Step 2: Import libraries**


```python
# first neural network with keras

from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

    2022-07-28 15:00:42.975159: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2022-07-28 15:00:42.975210: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.


**Step 3: Load the data**

We can now load the file as a matrix of numbers using the NumPy function loadtxt().


```python
# load the dataset
dataset = loadtxt('../assets/pima-indians-diabetes.csv', delimiter=',')

# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
```

**Step 4: Define KERAS model**

Models in Keras are defined as a sequence of layers.

We create a Sequential model and add layers one at a time until we are happy with our network architecture.

First, ensure the input layer has the right number of input features. This can be specified when creating the first layer with the input_shape argument and setting it to (8,) for presenting the 8 input variables as a vector.

How do we know the number of layers and their types? Often, the best network structure is found through a process of trial and error experimentation.

In this example, we will use a fully-connected network structure with three layers.

Fully connected layers are defined using the Dense class. We can specify the number of neurons or nodes in the layer as the first argument, and specify the activation function using the activation argument.

We will use the rectified linear unit activation function referred to as ReLU on the first two layers and the Sigmoid function in the output layer. We use a sigmoid on the output layer to ensure our network output is between 0 and 1 and easy to map to either a probability of class 1 or snap to a hard classification of either class with a default threshold of 0.5.

To sum up:

-The model expects rows of data with 8 variables (the input_shape=(8,) argument)

-The first hidden layer has 12 nodes and uses the relu activation function.

-The second hidden layer has 8 nodes and uses the relu activation function.

-The output layer has one node and uses the sigmoid activation function.


```python
# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

    2022-07-28 15:00:51.975352: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
    2022-07-28 15:00:51.975396: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
    2022-07-28 15:00:51.975424: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (4geeksacade-machinelear-xbdllitey6k): /proc/driver/nvidia/version does not exist
    2022-07-28 15:00:51.975703: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


**Step 5: Compile KERAS model**

Now that the model is defined, we can compile it. The backend automatically chooses the best way to represent the network for training and making predictions to run on your hardware, such as CPU or GPU or even distributed.

When compiling, we must specify some additional properties required when training the network. Remember training a network means finding the best set of weights to map inputs to outputs in our dataset.

We must specify the loss function to use to evaluate a set of weights, the optimizer is used to search through different weights for the network and any optional metrics we would like to collect. In this case, we will use cross entropy as the loss argument. This loss is for a binary classification problem and is defined in Keras as “binary_crossentropy“. 

We will define the optimizer as the efficient stochastic gradient descent algorithm “adam“. This is a popular version of gradient descent because it automatically tunes itself and gives good results in a wide range of problems. we will collect and report the classification accuracy, defined via the metrics argument.


```python
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

**Step 6: Fit KERAS model**

We have defined our model and compiled it ready for efficient computation. Now let's execute the model on some data.

Training occurs over epochs and each epoch is split into batches.

-Epoch: One pass through all of the rows in the training dataset.

-Batch: One or more samples considered by the model within an epoch before weights are updated

The training process will run for a fixed number of iterations through the dataset called epochs, that we must specify using the epochs argument. We must also set the number of dataset rows that are considered before the model weights are updated within each epoch, called the batch size and set using the batch_size argument. 

For this problem, we will run for a small number of epochs (150) and use a relatively small batch size of 10.


```python
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)
```

    Epoch 1/150
    77/77 [==============================] - 1s 1ms/step - loss: 14.0649 - accuracy: 0.3529
    Epoch 2/150
    77/77 [==============================] - 0s 1ms/step - loss: 4.0506 - accuracy: 0.4792
    Epoch 3/150
    77/77 [==============================] - 0s 1ms/step - loss: 1.2412 - accuracy: 0.5885
    Epoch 4/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.7737 - accuracy: 0.6289
    Epoch 5/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.7071 - accuracy: 0.6419
    Epoch 6/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6600 - accuracy: 0.6706
    Epoch 7/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6517 - accuracy: 0.6628
    Epoch 8/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6465 - accuracy: 0.6797
    Epoch 9/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6343 - accuracy: 0.6862
    Epoch 10/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6259 - accuracy: 0.6953
    Epoch 11/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6256 - accuracy: 0.6836
    Epoch 12/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6200 - accuracy: 0.6888
    Epoch 13/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6193 - accuracy: 0.6810
    Epoch 14/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6145 - accuracy: 0.6901
    Epoch 15/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6147 - accuracy: 0.6888
    Epoch 16/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6097 - accuracy: 0.6940
    Epoch 17/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6079 - accuracy: 0.6901
    Epoch 18/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6040 - accuracy: 0.6966
    Epoch 19/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6003 - accuracy: 0.7148
    Epoch 20/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6041 - accuracy: 0.6940
    Epoch 21/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5983 - accuracy: 0.7109
    Epoch 22/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5977 - accuracy: 0.6888
    Epoch 23/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5946 - accuracy: 0.7031
    Epoch 24/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5997 - accuracy: 0.6914
    Epoch 25/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5933 - accuracy: 0.7005
    Epoch 26/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5967 - accuracy: 0.6953
    Epoch 27/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5914 - accuracy: 0.7031
    Epoch 28/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5884 - accuracy: 0.7018
    Epoch 29/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5853 - accuracy: 0.7031
    Epoch 30/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5830 - accuracy: 0.6992
    Epoch 31/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5831 - accuracy: 0.7083
    Epoch 32/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5823 - accuracy: 0.7083
    Epoch 33/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5830 - accuracy: 0.7109
    Epoch 34/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5785 - accuracy: 0.7109
    Epoch 35/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5706 - accuracy: 0.7148
    Epoch 36/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5735 - accuracy: 0.7174
    Epoch 37/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5763 - accuracy: 0.7174
    Epoch 38/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5687 - accuracy: 0.7148
    Epoch 39/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5708 - accuracy: 0.7109
    Epoch 40/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5671 - accuracy: 0.7279
    Epoch 41/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5645 - accuracy: 0.7266
    Epoch 42/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5653 - accuracy: 0.7253
    Epoch 43/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5629 - accuracy: 0.7188
    Epoch 44/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5671 - accuracy: 0.7240
    Epoch 45/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5604 - accuracy: 0.7344
    Epoch 46/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5624 - accuracy: 0.7292
    Epoch 47/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5621 - accuracy: 0.7279
    Epoch 48/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5588 - accuracy: 0.7227
    Epoch 49/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5579 - accuracy: 0.7279
    Epoch 50/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5576 - accuracy: 0.7214
    Epoch 51/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5568 - accuracy: 0.7318
    Epoch 52/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5548 - accuracy: 0.7266
    Epoch 53/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5538 - accuracy: 0.7331
    Epoch 54/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5568 - accuracy: 0.7305
    Epoch 55/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5578 - accuracy: 0.7227
    Epoch 56/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5560 - accuracy: 0.7292
    Epoch 57/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5540 - accuracy: 0.7279
    Epoch 58/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5498 - accuracy: 0.7357
    Epoch 59/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5494 - accuracy: 0.7279
    Epoch 60/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5522 - accuracy: 0.7435
    Epoch 61/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5508 - accuracy: 0.7292
    Epoch 62/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5502 - accuracy: 0.7357
    Epoch 63/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5509 - accuracy: 0.7318
    Epoch 64/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5494 - accuracy: 0.7461
    Epoch 65/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5420 - accuracy: 0.7409
    Epoch 66/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5498 - accuracy: 0.7266
    Epoch 67/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5522 - accuracy: 0.7435
    Epoch 68/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5567 - accuracy: 0.7253
    Epoch 69/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5463 - accuracy: 0.7487
    Epoch 70/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5434 - accuracy: 0.7370
    Epoch 71/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5457 - accuracy: 0.7409
    Epoch 72/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5400 - accuracy: 0.7461
    Epoch 73/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5455 - accuracy: 0.7396
    Epoch 74/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5483 - accuracy: 0.7227
    Epoch 75/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5420 - accuracy: 0.7435
    Epoch 76/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5438 - accuracy: 0.7305
    Epoch 77/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5453 - accuracy: 0.7370
    Epoch 78/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5387 - accuracy: 0.7383
    Epoch 79/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5454 - accuracy: 0.7422
    Epoch 80/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5361 - accuracy: 0.7500
    Epoch 81/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5511 - accuracy: 0.7305
    Epoch 82/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5429 - accuracy: 0.7344
    Epoch 83/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5380 - accuracy: 0.7487
    Epoch 84/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5357 - accuracy: 0.7474
    Epoch 85/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5394 - accuracy: 0.7344
    Epoch 86/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5336 - accuracy: 0.7552
    Epoch 87/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5348 - accuracy: 0.7604
    Epoch 88/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5360 - accuracy: 0.7331
    Epoch 89/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5391 - accuracy: 0.7370
    Epoch 90/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5350 - accuracy: 0.7513
    Epoch 91/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5321 - accuracy: 0.7487
    Epoch 92/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5318 - accuracy: 0.7461
    Epoch 93/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5310 - accuracy: 0.7552
    Epoch 94/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5321 - accuracy: 0.7474
    Epoch 95/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5283 - accuracy: 0.7578
    Epoch 96/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5326 - accuracy: 0.7383
    Epoch 97/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5366 - accuracy: 0.7305
    Epoch 98/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5248 - accuracy: 0.7565
    Epoch 99/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5248 - accuracy: 0.7539
    Epoch 100/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5293 - accuracy: 0.7552
    Epoch 101/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5253 - accuracy: 0.7578
    Epoch 102/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5219 - accuracy: 0.7513
    Epoch 103/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5262 - accuracy: 0.7487
    Epoch 104/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5223 - accuracy: 0.7591
    Epoch 105/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5275 - accuracy: 0.7500
    Epoch 106/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5207 - accuracy: 0.7500
    Epoch 107/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5249 - accuracy: 0.7565
    Epoch 108/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5223 - accuracy: 0.7474
    Epoch 109/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5266 - accuracy: 0.7461
    Epoch 110/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5197 - accuracy: 0.7539
    Epoch 111/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5182 - accuracy: 0.7604
    Epoch 112/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5176 - accuracy: 0.7591
    Epoch 113/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5139 - accuracy: 0.7682
    Epoch 114/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5172 - accuracy: 0.7500
    Epoch 115/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5126 - accuracy: 0.7734
    Epoch 116/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5157 - accuracy: 0.7513
    Epoch 117/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5167 - accuracy: 0.7604
    Epoch 118/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5120 - accuracy: 0.7578
    Epoch 119/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5126 - accuracy: 0.7552
    Epoch 120/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5156 - accuracy: 0.7526
    Epoch 121/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5119 - accuracy: 0.7539
    Epoch 122/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5110 - accuracy: 0.7656
    Epoch 123/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5078 - accuracy: 0.7539
    Epoch 124/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5092 - accuracy: 0.7669
    Epoch 125/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5111 - accuracy: 0.7552
    Epoch 126/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5050 - accuracy: 0.7708
    Epoch 127/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5079 - accuracy: 0.7721
    Epoch 128/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5110 - accuracy: 0.7565
    Epoch 129/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5075 - accuracy: 0.7604
    Epoch 130/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5063 - accuracy: 0.7487
    Epoch 131/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5055 - accuracy: 0.7604
    Epoch 132/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5063 - accuracy: 0.7695
    Epoch 133/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5038 - accuracy: 0.7604
    Epoch 134/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5131 - accuracy: 0.7578
    Epoch 135/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5047 - accuracy: 0.7708
    Epoch 136/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5083 - accuracy: 0.7591
    Epoch 137/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5042 - accuracy: 0.7591
    Epoch 138/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5048 - accuracy: 0.7617
    Epoch 139/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5047 - accuracy: 0.7513
    Epoch 140/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5028 - accuracy: 0.7604
    Epoch 141/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5097 - accuracy: 0.7526
    Epoch 142/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5025 - accuracy: 0.7578
    Epoch 143/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5027 - accuracy: 0.7617
    Epoch 144/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5002 - accuracy: 0.7656
    Epoch 145/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.4977 - accuracy: 0.7669
    Epoch 146/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5013 - accuracy: 0.7656
    Epoch 147/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.4970 - accuracy: 0.7773
    Epoch 148/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5107 - accuracy: 0.7526
    Epoch 149/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.4965 - accuracy: 0.7721
    Epoch 150/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.4940 - accuracy: 0.7604





    <keras.callbacks.History at 0x7f9eb47c0fd0>



This is where the work happens on your CPU or GPU.

**Step 7: Evaluate KERAS model**

We have trained our neural network on the entire dataset and we can evaluate the performance of the network on the same dataset. You can ideally separate your data into train and test sets. The evaluate() function will return a list with two values. The first will be the loss of the model on the dataset and the second will be the accuracy of the model on the dataset. Here, we are noly interested in the accuracy so we'll ignore the loss value.


```python
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
```

    24/24 [==============================] - 0s 1ms/step - loss: 0.4906 - accuracy: 0.7786
    Accuracy: 77.86


Put all the previous code together in a .py file, for example named 'my_first_neural_network.py'. If you try running this example in an IPython or Jupyter notebook you may get an error.

You can then run the Python file as a script from your command line as follows:

```bash
python my_first_neural_network.py
```

Running this example, you should see a message for each of the 150 epochs printing the loss and accuracy, followed by the final evaluation of the trained model on the training dataset.

We would love the loss to go to zero and accuracy to go to 1.0. This is not possible for any but the most trivial machine learning problems. Instead, we will always have some error in our model. The goal is to choose a model configuration and training configuration that achieve the lowest loss and highest accuracy possible for a given dataset.

>Neural networks are a stochastic algorithm, meaning that the same algorithm on the same data can train a different model with different skill each time the code is run. This is a feature, not a bug. 

>The variance in the performance of the model means that to get a reasonable approximation of how well your model is performing, you may need to fit it many times and calculate the average of the accuracy scores.

**Step 8: Make predictions**

So after training my model a few times and getting an average of all the accuracies obtained, how do I make predictions?

Making predictions is as easy as calling the predict() function on the model. We are using a sigmoid activation function on the output layer, so the predictions will be a probability in the range between 0 and 1. We can easily convert them into a crisp binary prediction for this classification task by rounding them.


```python
# make probability predictions with the model. In this case we are using again the same dataset as if it was new data.
predictions = model.predict(X)
# round predictions 
rounded = [round(x[0]) for x in predictions]
```

    24/24 [==============================] - 0s 917us/step


We can convert the probability into 0 or 1 to predict crisp classes directly:


```python
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
```

    24/24 [==============================] - 0s 1ms/step


The complete example below makes predictions for each example in the dataset, then prints the input data, predicted class and expected class for the first 5 examples in the dataset.


```python
# first neural network with keras make predictions
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# load the dataset
dataset = loadtxt('../assets/pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]
# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10, verbose=0)
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
```

    24/24 [==============================] - 0s 953us/step
    [6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0] => 1 (expected 1)
    [1.0, 85.0, 66.0, 29.0, 0.0, 26.6, 0.351, 31.0] => 0 (expected 0)
    [8.0, 183.0, 64.0, 0.0, 0.0, 23.3, 0.672, 32.0] => 1 (expected 1)
    [1.0, 89.0, 66.0, 23.0, 94.0, 28.1, 0.167, 21.0] => 0 (expected 0)
    [0.0, 137.0, 40.0, 35.0, 168.0, 43.1, 2.288, 33.0] => 1 (expected 1)


The reason why you could get errors in a jupyter notebook is because of the output progress bars during training. You can easily turn these off by setting verbose=0 in the call to the fit() and evaluate() functions, as we just did in the example.

We can see that most rows are correctly predicted. In fact, we would expect about 76.9% of the rows to be correctly predicted based on our estimated performance of the model in the previous section.

**Step 9: Save your model**

Saving models requires that you have the h5py library installed. It is usually installed as a dependency with TensorFlow. You can also install it easily as follows:

```bash
sudo pip install h5py
```

Keras separates the concerns of saving your model architecture and saving your model weights.

The model structure can be described and saved using two different formats: JSON and YAML. Both of them save the model architecture and weights separately. The model weights are saved into a HDF5 format file in all cases.

Keras also supports a simpler interface to save both the model weights and model architecture together into a single H5 file.

Saving the model in this way includes everything we need to know about the model, including:

Model weights.
Model architecture.
Model compilation details (loss and metrics).
Model optimizer state.

This means that we can load and use the model directly, without having to re-compile it.

>Note: this is the preferred way for saving and loading your Keras model.

You can save your model by calling the save() function on the model and specifying the filename.

The example below demonstrates this by first fitting a model, evaluating it and saving it to the file model.h5.


```python

# MLP for Pima Indians Dataset saved to single file
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# load pima indians dataset
dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# define model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# save model and architecture to single file
model.save("model.h5")
print("Saved model to disk")
```

An equivalent code to save the model is the following:

```py
# equivalent to: model.save("model.h5")
from tensorflow.keras.models import save_model
save_model(model, "model.h5")
```

>If you want to know how to save your model using JSON or YAML go to https://machinelearningmastery.com/save-load-keras-deep-learning-models/

**Step 10: Load your model**

Your saved model can then be loaded later by calling the load_model() function and passing the filename. The function returns the model with the same architecture and weights.

In the following code, we load the model, summarize the architecture and evaluate it on the same dataset to confirm the weights and architecture are the same.


```python

# load and evaluate a saved model
from numpy import loadtxt
from tensorflow.keras.models import load_model
 
# load model
model = load_model('model.h5')
# summarize model.
model.summary()
# load dataset
dataset = loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# evaluate the model
score = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
```

Source:

https://keras.io/examples/

https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

https://keras.io/examples/vision/image_classification_from_scratch/

https://machinelearningmastery.com/save-load-keras-deep-learning-models/
