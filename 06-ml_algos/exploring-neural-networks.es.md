# Explorando KERAS

¡En este cuaderno vamos a cubrir paso a paso cómo modelar tu primera red neuronal!

Vamos a utilizar el conjunto de datos de inicio de diabetes de los indios Pima. Este es un conjunto de datos de Machine Learning estándar del repositorio de Machine Learning de UCI. Describe los datos de los registros médicos de los pacientes de los indios Pima y si tuvieron un inicio de diabetes dentro de los cinco años.

Como tal, es un problema de clasificación binaria (inicio de diabetes como 1 o no como 0). Todas las variables de entrada que describen a cada paciente son numéricas. Esto hace que sea fácil de usar directamente con redes neuronales que esperan valores numéricos de entrada y salida, y es ideal para nuestra primera red neuronal en Keras.

**Entender los datos**

Variables de Entrada (X):

- Número de veces embarazada.

- Concentración de glucosa plasmática a las 2 horas en una prueba de tolerancia oral a la glucosa.

- Presión arterial diastólica (mm Hg).

- Grosor del pliegue cutáneo del tríceps (mm).

- Insulina sérica de 2 horas (mu U/ml).

- Índice de masa corporal (peso en kg/(altura en m)^2).

- Función de pedigrí de diabetes.

- Años de edad.

Variables de salida (y):

Variable de clase (0 o 1)

**Requisitos:**

- Python 2 ó 3 instalado.

- SciPy (incluido NumPy) instalado.

- Keras y un backend (Theano o TensorFlow) instalado.

**Paso 1: instalar bibliotecas**


```python
pip install scipy tensorflow
```

    Requirement already satisfied: scipy in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (1.8.1)
    Collecting tensorflow
      Downloading tensorflow-2.9.1-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (511.7 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m511.7/511.7 MB[0m [31m3.0 MB/s[0m eta [36m0:00:00[0mm eta [36m0:00:01[0m[36m0:00:01[0m
    [?25hRequirement already satisfied: numpy<1.25.0,>=1.17.3 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from scipy) (1.23.1)
    Collecting grpcio<2.0,>=1.24.3
      Downloading grpcio-1.47.0-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.5 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.5/4.5 MB[0m [31m52.0 MB/s[0m eta [36m0:00:00[0m0m eta [36m0:00:01[0m
    [?25hCollecting keras-preprocessing>=1.1.1
      Downloading Keras_Preprocessing-1.1.2-py2.py3-none-any.whl (42 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m42.6/42.6 kB[0m [31m1.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting absl-py>=1.0.0
      Downloading absl_py-1.2.0-py3-none-any.whl (123 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m123.4/123.4 kB[0m [31m6.0 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting termcolor>=1.1.0
      Downloading termcolor-1.1.0.tar.gz (3.9 kB)
      Preparing metadata (setup.py) ... [?25ldone
    [?25hCollecting astunparse>=1.6.0
      Downloading astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
    Collecting protobuf<3.20,>=3.9.2
      Downloading protobuf-3.19.4-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m27.6 MB/s[0m eta [36m0:00:00[0m0m eta [36m0:00:01[0m
    [?25hCollecting opt-einsum>=2.3.2
      Downloading opt_einsum-3.3.0-py3-none-any.whl (65 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m65.5/65.5 kB[0m [31m3.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: wrapt>=1.11.0 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from tensorflow) (1.14.1)
    Collecting h5py>=2.9.0
      Downloading h5py-3.7.0-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (4.5 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.5/4.5 MB[0m [31m50.8 MB/s[0m eta [36m0:00:00[0m0m eta [36m0:00:01[0m[36m0:00:01[0m
    [?25hCollecting tensorboard<2.10,>=2.9
      Downloading tensorboard-2.9.1-py3-none-any.whl (5.8 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m5.8/5.8 MB[0m [31m62.5 MB/s[0m eta [36m0:00:00[0m0m eta [36m0:00:01[0m0:01[0m
    [?25hCollecting flatbuffers<2,>=1.12
      Downloading flatbuffers-1.12-py2.py3-none-any.whl (15 kB)
    Collecting libclang>=13.0.0
      Downloading libclang-14.0.1-py2.py3-none-manylinux1_x86_64.whl (14.5 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m14.5/14.5 MB[0m [31m62.5 MB/s[0m eta [36m0:00:00[0mm eta [36m0:00:01[0m[36m0:00:01[0m
    [?25hCollecting google-pasta>=0.1.1
      Downloading google_pasta-0.2.0-py3-none-any.whl (57 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m57.5/57.5 kB[0m [31m2.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: packaging in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from tensorflow) (21.3)
    Collecting tensorflow-estimator<2.10.0,>=2.9.0rc0
      Downloading tensorflow_estimator-2.9.0-py2.py3-none-any.whl (438 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m438.7/438.7 kB[0m [31m15.4 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: six>=1.12.0 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from tensorflow) (1.16.0)
    Requirement already satisfied: setuptools in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from tensorflow) (62.3.2)
    Collecting gast<=0.4.0,>=0.2.1
      Downloading gast-0.4.0-py3-none-any.whl (9.8 kB)
    Requirement already satisfied: typing-extensions>=3.6.6 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from tensorflow) (4.2.0)
    Collecting keras<2.10.0,>=2.9.0rc0
      Downloading keras-2.9.0-py2.py3-none-any.whl (1.6 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m41.5 MB/s[0m eta [36m0:00:00[0m0m eta [36m0:00:01[0m
    [?25hCollecting tensorflow-io-gcs-filesystem>=0.23.1
      Downloading tensorflow_io_gcs_filesystem-0.26.0-cp38-cp38-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (2.4 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.4/2.4 MB[0m [31m53.3 MB/s[0m eta [36m0:00:00[0m0m eta [36m0:00:01[0m
    [?25hRequirement already satisfied: wheel<1.0,>=0.23.0 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from astunparse>=1.6.0->tensorflow) (0.37.1)
    Collecting google-auth<3,>=1.6.3
      Downloading google_auth-2.9.1-py2.py3-none-any.whl (167 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m167.8/167.8 kB[0m [31m6.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting tensorboard-plugin-wit>=1.6.0
      Downloading tensorboard_plugin_wit-1.8.1-py3-none-any.whl (781 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m781.3/781.3 kB[0m [31m25.7 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: requests<3,>=2.21.0 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from tensorboard<2.10,>=2.9->tensorflow) (2.27.1)
    Collecting tensorboard-data-server<0.7.0,>=0.6.0
      Downloading tensorboard_data_server-0.6.1-py3-none-manylinux2010_x86_64.whl (4.9 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m4.9/4.9 MB[0m [31m59.4 MB/s[0m eta [36m0:00:00[0m0m eta [36m0:00:01[0m
    [?25hCollecting werkzeug>=1.0.1
      Downloading Werkzeug-2.2.1-py3-none-any.whl (232 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m232.4/232.4 kB[0m [31m13.3 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting google-auth-oauthlib<0.5,>=0.4.1
      Downloading google_auth_oauthlib-0.4.6-py2.py3-none-any.whl (18 kB)
    Collecting markdown>=2.6.8
      Downloading Markdown-3.4.1-py3-none-any.whl (93 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m93.3/93.3 kB[0m [31m4.7 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from packaging->tensorflow) (3.0.9)
    Collecting cachetools<6.0,>=2.0.0
      Downloading cachetools-5.2.0-py3-none-any.whl (9.3 kB)
    Collecting rsa<5,>=3.1.4
      Downloading rsa-4.9-py3-none-any.whl (34 kB)
    Collecting pyasn1-modules>=0.2.1
      Downloading pyasn1_modules-0.2.8-py2.py3-none-any.whl (155 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m155.3/155.3 kB[0m [31m6.2 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting requests-oauthlib>=0.7.0
      Downloading requests_oauthlib-1.3.1-py2.py3-none-any.whl (23 kB)
    Requirement already satisfied: importlib-metadata>=4.4 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from markdown>=2.6.8->tensorboard<2.10,>=2.9->tensorflow) (4.11.3)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (1.26.9)
    Requirement already satisfied: charset-normalizer~=2.0.0 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (2.0.12)
    Requirement already satisfied: idna<4,>=2.5 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (3.3)
    Requirement already satisfied: certifi>=2017.4.17 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard<2.10,>=2.9->tensorflow) (2022.5.18.1)
    Requirement already satisfied: MarkupSafe>=2.1.1 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from werkzeug>=1.0.1->tensorboard<2.10,>=2.9->tensorflow) (2.1.1)
    Requirement already satisfied: zipp>=0.5 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.10,>=2.9->tensorflow) (3.8.0)
    Collecting pyasn1<0.5.0,>=0.4.6
      Downloading pyasn1-0.4.8-py2.py3-none-any.whl (77 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m77.1/77.1 kB[0m [31m2.9 MB/s[0m eta [36m0:00:00[0m
    [?25hCollecting oauthlib>=3.0.0
      Downloading oauthlib-3.2.0-py3-none-any.whl (151 kB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m151.5/151.5 kB[0m [31m5.7 MB/s[0m eta [36m0:00:00[0m
    [?25hBuilding wheels for collected packages: termcolor
      Building wheel for termcolor (setup.py) ... [?25ldone
    [?25h  Created wheel for termcolor: filename=termcolor-1.1.0-py3-none-any.whl size=4832 sha256=e841deca723b0399f0c0c675f59521ea1dd329e069dbdd7d5820a743fb46009e
      Stored in directory: /home/gitpod/.cache/pip/wheels/a0/16/9c/5473df82468f958445479c59e784896fa24f4a5fc024b0f501
    Successfully built termcolor
    Installing collected packages: termcolor, tensorboard-plugin-wit, pyasn1, libclang, keras, flatbuffers, werkzeug, tensorflow-io-gcs-filesystem, tensorflow-estimator, tensorboard-data-server, rsa, pyasn1-modules, protobuf, opt-einsum, oauthlib, keras-preprocessing, h5py, grpcio, google-pasta, gast, cachetools, astunparse, absl-py, requests-oauthlib, markdown, google-auth, google-auth-oauthlib, tensorboard, tensorflow
    Successfully installed absl-py-1.2.0 astunparse-1.6.3 cachetools-5.2.0 flatbuffers-1.12 gast-0.4.0 google-auth-2.9.1 google-auth-oauthlib-0.4.6 google-pasta-0.2.0 grpcio-1.47.0 h5py-3.7.0 keras-2.9.0 keras-preprocessing-1.1.2 libclang-14.0.1 markdown-3.4.1 oauthlib-3.2.0 opt-einsum-3.3.0 protobuf-3.19.4 pyasn1-0.4.8 pyasn1-modules-0.2.8 requests-oauthlib-1.3.1 rsa-4.9 tensorboard-2.9.1 tensorboard-data-server-0.6.1 tensorboard-plugin-wit-1.8.1 tensorflow-2.9.1 tensorflow-estimator-2.9.0 tensorflow-io-gcs-filesystem-0.26.0 termcolor-1.1.0 werkzeug-2.2.1
    [33mWARNING: There was an error checking the latest version of pip.[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.


Para usar Keras, deberá tener instalado el paquete TensorFlow.

Una vez que TensorFlow esté instalado, solo importa Keras. Usaremos la biblioteca NumPy para cargar nuestro conjunto de datos y usaremos dos clases de la biblioteca Keras para definir nuestro modelo.

**Paso 2: importar bibliotecas**


```python
# Primera red neuronal con keras

from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
```

    2022-07-28 05:13:18.859773: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
    2022-07-28 05:13:18.859912: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.


**Paso 3: Cargar los datos**

Ahora podemos cargar el archivo como una matriz de números usando la función NumPy loadtxt().


```python
# Cargar el conjunto de datos
dataset = loadtxt('../assets/pima-indians-diabetes.csv', delimiter=',')

# Dividir en variables de entrada (X) y salida (y)
X = dataset[:,0:8]
y = dataset[:,8]
```

**Paso 2: Definir el modelo KERAS**

Los modelos en Keras se definen como una secuencia de capas.

Creamos un modelo secuencial y agregamos capas una a la vez hasta que estemos satisfechos con nuestra arquitectura de red.

Primero, asegúrate de que la capa de entrada tenga la cantidad correcta de entidades de entrada. Esto se puede especificar al crear la primera capa con el argumento input_shape y establecerlo en (8,) para presentar las 8 variables de entrada como un vector.

¿Cómo sabemos el número de capas y sus tipos? A menudo, la mejor estructura de red se encuentra a través de un proceso de experimentación de prueba y error.

En este ejemplo, utilizaremos una estructura de red completamente conectada con tres capas.

Las capas totalmente conectadas se definen mediante la clase Dense. Podemos especificar el número de neuronas o nodos en la capa como primer argumento y especificar la función de activación usando el argumento de activación.

Usaremos la función de activación de la unidad lineal rectificada denominada ReLU en las dos primeras capas y la función Sigmoid en la capa de salida. Usamos un sigmoide en la capa de salida para garantizar que la salida de nuestra red esté entre 0 y 1 y sea fácil de asignar a una probabilidad de clase 1 o ajustarse a una clasificación rígida de cualquier clase con un umbral predeterminado de 0,5.

Para resumir:

- El modelo espera filas de datos con 8 variables (el argumento input_shape=(8,).

- La primera capa oculta tiene 12 nodos y utiliza la función de activación relu.

- La segunda capa oculta tiene 8 nodos y utiliza la función de activación relu.

- La capa de salida tiene un nodo y utiliza la función de activación sigmoidea.


```python
# Definir el modelo de keras
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

    2022-07-28 05:34:23.313842: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcuda.so.1'; dlerror: libcuda.so.1: cannot open shared object file: No such file or directory
    2022-07-28 05:34:23.314107: W tensorflow/stream_executor/cuda/cuda_driver.cc:269] failed call to cuInit: UNKNOWN ERROR (303)
    2022-07-28 05:34:23.314184: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (4geeksacade-machinelear-646p8lv2dr4): /proc/driver/nvidia/version does not exist
    2022-07-28 05:34:23.314906: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
    To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.


**Paso 3: Compilar el modelo KERAS**

Ahora que el modelo está definido, podemos compilarlo. El backend elige automáticamente la mejor manera de representar la red para entrenar y hacer predicciones para ejecutar en su hardware, como CPU o GPU o incluso distribuido.

Al compilar, debemos especificar algunas propiedades adicionales requeridas al entrenar la red. Recuerde entrenar una red significa encontrar el mejor conjunto de pesos para asignar entradas a salidas en nuestro conjunto de datos.

Debemos especificar la función de pérdida a usar para evaluar un conjunto de pesos, el optimizador se usa para buscar entre diferentes pesos para la red y cualquier métrica opcional que nos gustaría recopilar. En este caso, utilizaremos la entropía cruzada como argumento de pérdida. Esta pérdida es para un problema de clasificación binaria y se define en Keras como "binary_crossentropy". 

Definiremos el optimizador como el algoritmo de descenso de gradiente estocástico eficiente “adam”. Esta es una versión popular del descenso de gradiente porque se sintoniza automáticamente y brinda buenos resultados en una amplia gama de problemas. recopilaremos e informaremos la precisión de la clasificación, definida a través del argumento de las métricas.


```python
# Compilar el modelo de keras
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

**Paso 4: ajuste el modelo KERAS**

Hemos definido nuestro modelo y lo compilamos listo para un cálculo eficiente. Ahora vamos a ejecutar el modelo en algunos datos.

El entrenamiento ocurre en epochs (épocas) y cada época se divide en lotes (batches).

-Epoch: Una pasada por todas las filas del conjunto de datos de entrenamiento.

-Batch: Una o más muestras consideradas por el modelo dentro de una época antes de que se actualicen los pesos.

El proceso de entrenamiento se ejecutará durante un número fijo de iteraciones a través del conjunto de datos llamado epochs, que debemos especificar usando el argumento epochs. También debemos establecer la cantidad de filas del conjunto de datos que se consideran antes de que se actualicen los pesos del modelo dentro de cada epoch, lo que se denomina tamaño de batch y se establece mediante el argumento  batch_size (tamaño_lote).

Para este problema, ejecutaremos una pequeña cantidad de epochs (150) y usaremos un tamaño de batch relativamente pequeño de 10.


```python
# Ajustar el modelo de keras en el conjunto de datos
model.fit(X, y, epochs=150, batch_size=10)
```

    Epoch 1/150
    77/77 [==============================] - 0s 1ms/step - loss: 10.6695 - accuracy: 0.5234
    Epoch 2/150
    77/77 [==============================] - 0s 1ms/step - loss: 2.8471 - accuracy: 0.4792
    Epoch 3/150
    77/77 [==============================] - 0s 1ms/step - loss: 1.5770 - accuracy: 0.5833
    Epoch 4/150
    77/77 [==============================] - 0s 1ms/step - loss: 1.2138 - accuracy: 0.6120
    Epoch 5/150
    77/77 [==============================] - 0s 1ms/step - loss: 1.0117 - accuracy: 0.6341
    Epoch 6/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.9316 - accuracy: 0.6250
    Epoch 7/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.8823 - accuracy: 0.6133
    Epoch 8/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.7788 - accuracy: 0.6276
    Epoch 9/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.7285 - accuracy: 0.6628
    Epoch 10/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.7102 - accuracy: 0.6602
    Epoch 11/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6894 - accuracy: 0.6654
    Epoch 12/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6629 - accuracy: 0.6719
    Epoch 13/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6379 - accuracy: 0.6875
    Epoch 14/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6439 - accuracy: 0.6797
    Epoch 15/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6375 - accuracy: 0.6810
    Epoch 16/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6215 - accuracy: 0.6732
    Epoch 17/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6124 - accuracy: 0.6732
    Epoch 18/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6079 - accuracy: 0.6901
    Epoch 19/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6080 - accuracy: 0.6940
    Epoch 20/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6216 - accuracy: 0.6693
    Epoch 21/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5997 - accuracy: 0.6953
    Epoch 22/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6183 - accuracy: 0.6745
    Epoch 23/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5904 - accuracy: 0.7018
    Epoch 24/150
    77/77 [==============================] - 0s 1000us/step - loss: 0.5891 - accuracy: 0.7122
    Epoch 25/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5954 - accuracy: 0.6979
    Epoch 26/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5957 - accuracy: 0.7005
    Epoch 27/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5969 - accuracy: 0.6862
    Epoch 28/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5912 - accuracy: 0.7083
    Epoch 29/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5947 - accuracy: 0.6914
    Epoch 30/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5872 - accuracy: 0.6979
    Epoch 31/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5696 - accuracy: 0.7148
    Epoch 32/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5849 - accuracy: 0.7057
    Epoch 33/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5822 - accuracy: 0.7044
    Epoch 34/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5677 - accuracy: 0.7292
    Epoch 35/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5790 - accuracy: 0.7174
    Epoch 36/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5644 - accuracy: 0.7370
    Epoch 37/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5683 - accuracy: 0.7357
    Epoch 38/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5534 - accuracy: 0.7266
    Epoch 39/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5704 - accuracy: 0.7188
    Epoch 40/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5777 - accuracy: 0.7018
    Epoch 41/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5637 - accuracy: 0.7148
    Epoch 42/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5711 - accuracy: 0.7122
    Epoch 43/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5645 - accuracy: 0.7174
    Epoch 44/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5430 - accuracy: 0.7357
    Epoch 45/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5654 - accuracy: 0.7135
    Epoch 46/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5773 - accuracy: 0.7148
    Epoch 47/150
    77/77 [==============================] - 0s 999us/step - loss: 0.5402 - accuracy: 0.7318
    Epoch 48/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5501 - accuracy: 0.7227
    Epoch 49/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5680 - accuracy: 0.7188
    Epoch 50/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5444 - accuracy: 0.7409
    Epoch 51/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5596 - accuracy: 0.7266
    Epoch 52/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5690 - accuracy: 0.7240
    Epoch 53/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5686 - accuracy: 0.7253
    Epoch 54/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5399 - accuracy: 0.7500
    Epoch 55/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5571 - accuracy: 0.7083
    Epoch 56/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5607 - accuracy: 0.7135
    Epoch 57/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.6115 - accuracy: 0.6914
    Epoch 58/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5427 - accuracy: 0.7292
    Epoch 59/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5412 - accuracy: 0.7331
    Epoch 60/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5464 - accuracy: 0.7357
    Epoch 61/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5510 - accuracy: 0.7253
    Epoch 62/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5470 - accuracy: 0.7279
    Epoch 63/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5394 - accuracy: 0.7461
    Epoch 64/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5532 - accuracy: 0.7370
    Epoch 65/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5593 - accuracy: 0.7188
    Epoch 66/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5617 - accuracy: 0.7227
    Epoch 67/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5374 - accuracy: 0.7539
    Epoch 68/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5295 - accuracy: 0.7448
    Epoch 69/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5477 - accuracy: 0.7292
    Epoch 70/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5298 - accuracy: 0.7305
    Epoch 71/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5277 - accuracy: 0.7435
    Epoch 72/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5288 - accuracy: 0.7500
    Epoch 73/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5487 - accuracy: 0.7331
    Epoch 74/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5351 - accuracy: 0.7396
    Epoch 75/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5451 - accuracy: 0.7227
    Epoch 76/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5359 - accuracy: 0.7331
    Epoch 77/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5287 - accuracy: 0.7331
    Epoch 78/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5301 - accuracy: 0.7552
    Epoch 79/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5349 - accuracy: 0.7409
    Epoch 80/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5318 - accuracy: 0.7487
    Epoch 81/150
    77/77 [==============================] - 0s 984us/step - loss: 0.5210 - accuracy: 0.7578
    Epoch 82/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5163 - accuracy: 0.7578
    Epoch 83/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5234 - accuracy: 0.7578
    Epoch 84/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5183 - accuracy: 0.7500
    Epoch 85/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5610 - accuracy: 0.7083
    Epoch 86/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5463 - accuracy: 0.7422
    Epoch 87/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5180 - accuracy: 0.7604
    Epoch 88/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5239 - accuracy: 0.7565
    Epoch 89/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5198 - accuracy: 0.7669
    Epoch 90/150
    77/77 [==============================] - 0s 985us/step - loss: 0.5216 - accuracy: 0.7539
    Epoch 91/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5161 - accuracy: 0.7500
    Epoch 92/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5201 - accuracy: 0.7500
    Epoch 93/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5406 - accuracy: 0.7448
    Epoch 94/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5197 - accuracy: 0.7474
    Epoch 95/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5210 - accuracy: 0.7526
    Epoch 96/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5145 - accuracy: 0.7539
    Epoch 97/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5065 - accuracy: 0.7500
    Epoch 98/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5300 - accuracy: 0.7331
    Epoch 99/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5451 - accuracy: 0.7279
    Epoch 100/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5168 - accuracy: 0.7461
    Epoch 101/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5025 - accuracy: 0.7643
    Epoch 102/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5136 - accuracy: 0.7526
    Epoch 103/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5185 - accuracy: 0.7565
    Epoch 104/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5325 - accuracy: 0.7422
    Epoch 105/150
    77/77 [==============================] - 0s 2ms/step - loss: 0.5083 - accuracy: 0.7526
    Epoch 106/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5271 - accuracy: 0.7409
    Epoch 107/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5440 - accuracy: 0.7331
    Epoch 108/150
    77/77 [==============================] - 0s 987us/step - loss: 0.5151 - accuracy: 0.7578
    Epoch 109/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5164 - accuracy: 0.7422
    Epoch 110/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5150 - accuracy: 0.7643
    Epoch 111/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5152 - accuracy: 0.7630
    Epoch 112/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5080 - accuracy: 0.7578
    Epoch 113/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5125 - accuracy: 0.7513
    Epoch 114/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5012 - accuracy: 0.7604
    Epoch 115/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5078 - accuracy: 0.7617
    Epoch 116/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5052 - accuracy: 0.7630
    Epoch 117/150
    77/77 [==============================] - 0s 2ms/step - loss: 0.5057 - accuracy: 0.7552
    Epoch 118/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5246 - accuracy: 0.7448
    Epoch 119/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5009 - accuracy: 0.7409
    Epoch 120/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5055 - accuracy: 0.7591
    Epoch 121/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5194 - accuracy: 0.7461
    Epoch 122/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5080 - accuracy: 0.7526
    Epoch 123/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5165 - accuracy: 0.7630
    Epoch 124/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.4975 - accuracy: 0.7591
    Epoch 125/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5214 - accuracy: 0.7513
    Epoch 126/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5098 - accuracy: 0.7552
    Epoch 127/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5035 - accuracy: 0.7526
    Epoch 128/150
    77/77 [==============================] - 0s 994us/step - loss: 0.5033 - accuracy: 0.7617
    Epoch 129/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5023 - accuracy: 0.7591
    Epoch 130/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.4933 - accuracy: 0.7826
    Epoch 131/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5183 - accuracy: 0.7422
    Epoch 132/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5425 - accuracy: 0.7344
    Epoch 133/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.4949 - accuracy: 0.7708
    Epoch 134/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.4916 - accuracy: 0.7721
    Epoch 135/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5014 - accuracy: 0.7565
    Epoch 136/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5028 - accuracy: 0.7630
    Epoch 137/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5124 - accuracy: 0.7565
    Epoch 138/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.4944 - accuracy: 0.7591
    Epoch 139/150
    77/77 [==============================] - 0s 993us/step - loss: 0.4904 - accuracy: 0.7604
    Epoch 140/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5205 - accuracy: 0.7630
    Epoch 141/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.4927 - accuracy: 0.7721
    Epoch 142/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.4993 - accuracy: 0.7630
    Epoch 143/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5110 - accuracy: 0.7318
    Epoch 144/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.4873 - accuracy: 0.7591
    Epoch 145/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.4911 - accuracy: 0.7682
    Epoch 146/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.4785 - accuracy: 0.7865
    Epoch 147/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5080 - accuracy: 0.7487
    Epoch 148/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.5013 - accuracy: 0.7643
    Epoch 149/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.4974 - accuracy: 0.7604
    Epoch 150/150
    77/77 [==============================] - 0s 1ms/step - loss: 0.4873 - accuracy: 0.7695





    <keras.callbacks.History at 0x7f4d0071b0a0>



Aquí es donde ocurre el trabajo en tu CPU o GPU.

**Paso 5: Evaluar el modelo KERAS**

Hemos entrenado nuestra red neuronal en todo el conjunto de datos y podemos evaluar el rendimiento de la red en el mismo conjunto de datos. Idealmente, puedes separar tus datos en conjuntos de entrenamiento y prueba. La función evaluar() devolverá una lista con dos valores. El primero será la pérdida del modelo en el conjunto de datos y el segundo será la precisión del modelo en el conjunto de datos. Aquí, no estamos interesados ​​en la precisión, por lo que ignoraremos el valor de la pérdida.


```python
# Evaluar el modelo de keras
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))
```

    24/24 [==============================] - 0s 1ms/step - loss: 0.4707 - accuracy: 0.7786
    Accuracy: 77.86


Pon todo junto en un archivo .py. Si intentas ejecutar este ejemplo en un cuaderno IPython o Jupyter, es posible que obtenga un error.

**Paso 7: Hacer predicciones**

Fuente:

https://keras.io/examples/

https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

https://keras.io/examples/vision/image_classification_from_scratch/
