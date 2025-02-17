---
description: >-
  Introducción al aprendizaje automático en AWS: Aprenda a transformar problemas empresariales en soluciones de ML, a utilizar SageMaker y a implementar modelos de IA de manera eficiente.
---

# Primeros pasos con Machine Learning en las herramientas de AWS

Independientemente del problema en el que estés trabajando, normalmente debes realizar los siguientes pasos: 

Comenzamos con nuestro problema comercial y dedicamos un tiempo a convertirlo en un problema de Machine Learning, que no es un proceso simple. Luego comenzamos a recopilar los datos y, una vez que tenemos todos nuestros datos juntos, los visualizamos, los analizamos y hacemos un poco de ingeniería de funciones para finalmente tener datos limpios listos para entrenar nuestro modelo. Probablemente, no tengamos nuestro modelo ideal desde el principio, por lo que con la evaluación del modelo medimos cómo está funcionando el modelo, si nos gusta el rendimiento o no, si es preciso o no, y luego comenzamos a optimizarlo ajustando algunos hiperparámetros.

Una vez que estemos satisfechos con nuestro modelo, debemos verificar si satisface nuestro objetivo comercial inicial; de lo contrario, tendríamos que trabajar en el aumento de funciones o en la recopilación de más datos. Una vez que se cumple nuestro objetivo comercial, podemos implementar nuestro modelo para hacer predicciones en un entorno de producción y no termina ahí porque queremos mantenerlos actualizados y al corriente para seguir entrenándolos con más datos. Mientras que en el software escribes las reglas a seguir, en Machine Learning, el modelo determina la regla en función de los datos con los que se ha entrenado. Entonces, para mantenerse actualizado, debes volver a entrenar tu modelo con los datos actuales.

No es sencillo, pero ya hemos aprendido a hacer todo esto por nuestra cuenta. La buena noticia sobre la computación en la nube es que podemos implementar algunas soluciones de ML sin tener que pasar por cada uno de los pasos anteriores.

Debido a que actualmente es el proveedor de computación en la nube número uno, elegimos AWS para aprender algunas habilidades de computación en la nube. En la siguiente imagen podemos ver la pila (*stack*) de Machine Learning de AWS de tres capas.

![Stack de AWS](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/aws_stack.jpg?raw=true)

En la parte inferior de la pila podemos encontrar 'Marcos e infraestructura de ML', que es lo que AWS llamaría la forma "difícil" de hacer Machine Learning, ejecutando algunas máquinas virtuales en las que podemos usar GPU si las necesitamos e instalamos. Algunos frameworks, por ejemplo TensorFlow, para empezar a hacer todos los pasos mencionados anteriormente.

Hay una forma más fácil, que son los 'servicios ML'. Esto es todo sobre el servicio llamado SageMaker. SageMaker es un servicio que tiene el pipeline anterior listo para que lo uses, pero aún necesitas conocer los algoritmos que deseas usar y aún necesitas codificar un poco si quieres profundizar un poco más.

Ahora veamos la forma más fácil en la parte superior de la imagen de la pila. En 'AI Models' los modelos ya están construidos. Utilizamos, por ejemplo, un servicio de procesamiento de lenguaje natural llamado 'Amazon Translate'.

Los servicios de IA son una excelente manera de probar la IA, especialmente si no tienes experiencia o si estás trabajando en una experimentación rápida, son una forma rápida de obtener valor comercial, y si encuentras dónde está el valor comercial y necesitas algo más personalizado, luego puedes bajar la pila a la siguiente capa.

Lo mejor de estas APIs de servicios de IA es que, como desarrollador, puedes comenzar a experimentar en lugar de tener que aprender muchas cosas antes de comenzar a usarlas, y luego puedes profundizar y personalizarlas.

Tres cosas que el desarrollador debe aprender para aprovechar al máximo estos servicios:

1. Comprende tus datos, no solo en los servicios de IA, sino en todo Machine Learning.

2. Comprende tu caso de uso, prueba el servicio con tu caso de uso particular, no solo con el genérico.

3. Entiende cómo es el éxito. Machine Learning es muy poderoso, pero no será 100% preciso.

## Amazon SageMaker

**¿Qué es Amazon SageMaker?**

Amazon SageMaker proporciona capacidades de Machine Learning para que los científicos y desarrolladores de datos preparen, construyan, entrenen e implementen modelos ML de alta calidad de manera eficiente.

**Flujo de trabajo de SageMaker:**

1. Label data: configura y administra trabajos de etiquetado para conjuntos de datos de entrenamiento de alta precisión dentro de Amazon SageMaker, utilizando el aprendizaje activo y el etiquetado humano.

2. Build: conéctese a otros servicios de AWS y transforma datos en notebooks de Amazon SageMaker.

3. Train: utiliza los marcos y los algoritmos de Amazon SageMaker, o trae los tuyos propios, para el entrenamiento distribuido.

4. Tune: Amazon SageMaker ajusta automáticamente tu modelo ajustando múltiples combinaciones de parámetros de algoritmo.

5. Deploy: Una vez completado el entrenamiento, los modelos se pueden implementar en los endpoints de Amazon SageMaker para realizar predicciones en tiempo real.

6. Discover: Encuentra, compra e implementa paquetes modelo, algoritmos y productos de datos listos para usar en el mercado de AWS.

**Beneficios de SageMaker**

- Para la exploración y el preprocesamiento de datos, proporciona instancias totalmente administradas que ejecutan Jupyter notebooks que incluyen código de ejemplo para ejercicios comunes de alojamiento y entrenamiento de modelos.

- Cuando estés listo para entrenar tus datos, simplemente indica el tipo y la cantidad de instancias que necesitas y comienza el entrenamiento con un solo clic.

- Proporciona algoritmos de Machine Learning altamente optimizados en cuanto a velocidad, precisión y escalado para ejecutarse en conjuntos de datos de entrenamiento extremadamente grandes.

- SageMaker proporciona artefactos de modelo e imágenes de puntuación para su implementación en Amazon EC2 o en cualquier otro lugar.

![Amazon SageMaker](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/sagemaker.jpg?raw=true)

## SageMaker Studio

Cuando inicias sesión en Amazon Web Services y eliges SageMaker, debes realizar algunos primeros pasos importantes:

Primero necesitamos crear un dominio de Studio. Este es un proceso único que realizas en tu consola de AWS solo para SageMaker. 

Una vez creado, AWS administra el servidor por ti, y tú como consumidor, iniciarás sesión en un perfil de usuario. Puedes tener tantos perfiles de usuario como desees por dominio de Studio. Lo que normalmente ocurre en las organizaciones es que dan un dominio de Studio a un equipo de científicos de datos y crean tantos perfiles de usuario como necesitan.

Cada vez que creas un perfil de usuario, es nuevamente el servidor Jupyter el que administra AWS, pero cada vez que crees un nuevo notebook, aplicación o máquina, se ejecutarán en una nueva instancia EC2 cada uno, por lo que debes tener cuidado porque todas tus instancias en ejecución es lo que se cobrará, así que asegúrate de cerrar la sesión o, si no estás utilizando una instancia, apágala para que no se te cobre. Lo bueno es que puedes tener tantas máquinas funcionando como desees y las tienes funcionando en todos los entornos diferentes, por lo que puedes tener una máquina R, una máquina Spark, una máquina PyTorch, una máquina TensorFlow, entre otras.

Razones para elegir SageMaker Studio:

- Tienes más poder de cómputo, puedes tener más máquinas para ejecutar.

- Puedes agregar tantos datos como desees al servidor y simplemente crece.

- Todos los widgets que tienes: Proyectos, Data Wrangler, Feature Store, Pipelines, Experimentos y ensayos, Registro de modelos, Endpoints, etc.

### Código de ejemplo utilizado en SageMaker Studio

En el panel de control izquierdo de SageMaker encontrarás SageMaker Studio. Una vez que hagas clic en él, encontrarás una especie de entorno de JupyterLab y se te presentará una pantalla de inicio.

**Contexto**

En este ejemplo de SageMaker, aprenderás los pasos para crear, entrenar, ajustar e implementar un modelo de detección de fraude.

Los siguientes pasos incluyen la preparación de tu SageMaker notebook, la descarga de datos de Internet a SageMaker, la transformación de los datos, el uso del algoritmo Gradient Boosting para crear un modelo, evaluar su efectividad y configurar el modelo para hacer predicciones.

**Preparación**

Una vez que instales Pandas, debes especificar algunos detalles:

- El depósito de S3 y el prefijo son los que debes usar para entrenar y modelar los datos, porque los datos deben provenir de algún almacenamiento. No se recomienda tener tus datos cargados en un Notebook porque si tienes muchos datos y los estás cargando constantemente, se te están cobrando constantemente y también es más lento. S3 definitivamente es más amigable en términos de carga y aterrizaje de nuestros conjuntos de datos porque puedes almacenar terabytes de datos y puedes entrenar tu modelo en terabytes de datos. Los datos también se pueden almacenar en el sistema de archivos elástico (EFS).

- El rol de IAM se utiliza para brindar capacitación y acceso de hospedaje a tus datos. Esta función es la función que se te asignó cada vez que inició SageMaker Studio. Con `get_execution_role()` buscamos ese rol. Es importante porque proporciona acceso a SageMaker a otros recursos de AWS.

```py
!pip install --upgrade pandas
```

> Es importante mencionar que cuando creas un nuevo Notebook y es hora de seleccionar el kernel, puedes reutilizar el kernel de una sesión existente, para que tengas todos los mismos paquetes, para evitar la reinstalación. Otra forma es crear por separado una imagen de docker base y adjuntarla a tu dominio de Studio.

```py
import sagemaker # importar el sdk de sagemaker python, similar a otros paquetes de python pero con características diferentes
bucket = sagemaker.Session().default_bucket()
prefix = 'sagemaker/fraud-detection'

# Definir el rol de IAM
import boto3
import re
from sagemaker import get_execution_role

role = get_execution_role()

# Importar las librerías de python que necesitaremos
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Ipython.display import Image  # Para mostrar imágenes el notebook
from Ipython.display import display  # Para mostrar las salidas en el notebook
from time import gmtime, strftime  # Para etiquetar modelos de SageMaker
import sys
import math
import json
import zipfile
```

No necesitas estrictamente el SDK de Python de SageMaker para usar SageMaker. Hay otras formas de invocar las API de SageMaker:

- Puedes usar boto3.

- Puedes usar la CLI de AWS.

```py
# Asegúrate de que la versión de Pandas esté configurada en 1.2.4 o posterior
# Siempre es una buena idea comprobar las versiones
# Debido a que SageMaker agrega características constantemente

pd.__version__
```

**Datos**

En este ejemplo, los datos se almacenan en S3, por lo que los descargaremos del depósito público de S3.

```py
!wget https://s3-us-west-2.amazonaws.com/sagemaker-e2e-solutions/fraud-detection/creditcardfraud.zip

# Descomprimiendo los datos
with zipfile.ZipFile('creditcardfraud.zip','r') as zip_ref:
    zip_ref.extractall('.')
```

Ahora llevemos esto a un marco de datos de Pandas y echemos un primer vistazo.

Estos datos están optimizados y ya normalizados, pero los escenarios comunes con los datos son que no tenemos datos o que los datos nunca están limpios. La normalización debe realizarse en algunos casos, pero depende del modelo que utilices. XGBoost tiende a ser muy robusto para los datos no normalizados, pero si estás utilizando K-means, por ejemplo, o cualquier modelo de aprendizaje profundo, se debe realizar la normalización.

```py
data = pd.read_csv('./creditcard.csv')
print(data.columns)
data[['Time','V1','V2','V27','V28','Amount','Class']].describe()
data.head(10)
```

Este conjunto de datos tiene 28 columnas Vi para i = 1...28 de características anonimizadas junto con columnas de tiempo, cantidad y clase. Ya sabemos que las columnas Vi se han normalizado para tener 0 media y desviación estándar unitaria como resultado de PCA.

La columna de clase corresponde a si una transacción es fraudulenta o no. Verás con el siguiente código que la mayoría de los datos no son fraudulentos con solo 492 (.173%) contra 284315 no fraudulentos (99.827%).

```py
nonfrauds, frauds = data.groupby('Class').size()
print('Number of frauds: ', frauds)
print('Number of non-frauds: ', nonfrauds)
print('Percentage of fraudulent data: ', 100.*frauds/(frauds + nonfrauds))
```

Aquí, estamos lidiando con un conjunto de datos desequilibrado, por lo que la métrica de "precisión" sería una métrica engañosa, ya que probablemente predecirá la ausencia de fraude en el 100 % de los casos con una precisión del 99,9 %. Otras métricas como el recuerdo o la precisión nos ayudan a atacar ese problema. Otra técnica es utilizar el sobremuestreo de los datos de la clase minoritaria o el submuestreo de los datos de la clase mayoritaria. Puedes leer sobre SMOTE para comprender ambas técnicas.

Ahora separemos nuestros datos en características y etiquetas.

```py
feature_columns = data.columns[:-1]
label_column = data.columns[-1]

features = data[feature_columns].values.astype('float32')
labels = (data[label_column].values).astype('float32')
```

```py
# XGBoost necesita que la variable objetivo sea la primera
model_data = data
model_data.head()
model_data = pd.concat([model_data['Class'], model_data.drop(['Class'], axis=1)], axis=1)
model_data.head()
```

Separación de datos en conjuntos de entrenamiento y validación.

```py
train_data, validation_data, test_data = np.split(model_data.sample(frac=1, random_state=1729),
                                                [int(0.7 * len(model_data)), int(0.9 * len(model_data))])
train_data.to_csv('train.csv', header = False, index = False)
validation_data.to_csv('validation.csv', header = False, index = False)
```

Necesitamos los datos antes de que pueda comenzar el entrenamiento, en S3. Como hemos descargado los datos de **S3**, tenemos que volver a cargarlos. Podemos crear un script de Python para preprocesar los datos y cargarlos automáticamente por nosotros.

```py
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train/train.csv')) \
                                .upload_file('train.csv')  
boto3.Session().resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'validation/validation.csv')) \
                                .upload_file('validation.csv')
s3_train_data = 's3://{}/{}/train/train.csv'.format(bucket, prefix)
s3_validation_data = 's3://{}/{}/validation/validation.csv'.format(bucket, prefix)
print('Uploaded training data location: {}'.format(s3_train_data))
print('Uploaded training data location: {}'.format(s3_validation_data))

output_location = 's3://{}/{}/output'.format(bucket, prefix)
print('Training artifacts will be uploaded to: {}'.format(output_location))
```

**Entrenamiento**

El algoritmo elegido para este ejemplo es XGBoost, por lo que tenemos dos opciones de entrenamiento. La primera es usar el XGBoost integrado proporcionado por AWS SageMaker o podemos usar el paquete de código abierto para XGBoost. En este ejemplo en particular, utilizaremos el algoritmo integrado proporcionado por AWS.

En SageMaker, detrás de escena todo se basa en contenedores, por lo que debemos especificar las ubicaciones de los contenedores de algoritmo elegidos. Para especificar el algoritmo de aprendizaje lineal, usamos una función de utilidad para obtener su URL.

La lista completa de algoritmos incorporados se puede encontrar aquí: https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html

Así que procedamos con el proceso de entrenamiento. Primero, debemos especificar la ubicación del contenedor ECR para la implementación de XGBoost de Amazon SageMaker.

```py
container = sagemaker.image_uris.retrieve(region=boto3.Session().region_name, framework='xgboost', version='latest')
```

Debido a que estamos entrenando con el formato de archivo CSV, crearemos `s3_inputs` que nuestra función de entrenamiento puede usar como puntero a los archivos en S3, que también especifican que el tipo de contenido es CSV.

```py
s3_input_train = sagemaker.inputs.TrainingInput(s3_data = 's3://{}/{}/train'.format(bucket, prefix), content_type='csv')
s3_input_validation = sagemaker.inputs.TrainingInput(s3_data = 's3://{}/{}/validation'.format(bucket, prefix), content_type='csv')
```

Cuando ejecutamos un trabajo de capacitación en SageMaker, no usamos el poder de cómputo en nuestro notebook. Estamos haciendo este trabajo de entrenamiento en una máquina en la nube, y necesitamos la flexibilidad para especificar cualquier máquina que queramos usar, según el algoritmo que estamos usando. Por ejemplo, si usamos un modelo de aprendizaje profundo, es posible que queramos usar el GPU en lugar de CPU, por lo que necesitamos esa flexibilidad porque también queremos optimizar el costo. Si tuviéramos GPU en nuestra computadora portátil y nos distrajéramos con la codificación, no queremos pagar por ese tiempo porque realmente no estamos usando la máquina para lo que debe ser, por lo que se vuelve muy importante que en el entrenamiento de trabajo proporcionemos un cierto tipo de máquina y cuando se realiza el entrenamiento automáticamente finaliza todos los recursos.

Para resumir la idea, debemos realizar el entrenamiento en una instancia separada porque, según nuestro trabajo, nuestro modelo y el tamaño del conjunto de datos, debemos especificar el estilo de instancia apropiado.

Para hacer eso, necesitamos hacer esa configuración especificando parámetros de entrenamiento para el estimador. Esto incluye:

1. El contenedor del algoritmo XGBoost.

2. El rol de IAM a utilizar.

3. Tipo y conteo de instancias de entrenamiento.

4. Ubicación S3 para datos de salida.

5. Hiperparámetros del algoritmo.

Y luego una función `.fit()` que especifica la ubicación S3 para los datos de salida.

```py
sess = sagemaker.Session()

xgb = sagemaker.estimator.Estimator(container,
                                    role, 
                                    instance_count = 1, # Si proporcionamos más de 1, se hará automáticamente
                                    instance_type = 'ml.m4.xlarge',
                                    output_path = 's3://{}/{}/output'.format(bucket,prefix),
                                    sagemaker_session = sess)
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        silent=0,
                        objective='binary:logistic',
                        num_round=100)

xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})
```

**Alojamiento**

Una vez que el modelo está entrenado, podemos usar el estimador con `.deploy()`.

```py
xgb_predictor = xgb.deploy(initial_instance_count = 1, # La palabra 'initial' es porque puedes actualizarla más tarde para aumentarla
                            instance_type = 'ml.m4.xlarge')
```
 
**Evaluación**

Una vez implementado, puede evaluarlo en el notebook de SageMaker. En este caso, solo estamos prediciendo si es una transacción fraudulenta (1) o no (0), lo que produce una matriz de confusión simple.

Primero, necesitamos determinar cómo pasamos y recibimos datos de nuestro punto final. Nuestros datos se almacenan actualmente como arrays de NumPy en la memoria de nuestra instancia de computadora portátil. Para enviarlo en una solicitud HTTP POST, lo serializaremos como una cadena CSV y luego decodificaremos el CSV resultante.

> Para la inferencia con formato CSV, SageMaker XGBoost requiere que los datos no incluyan la variable de destino.

```py
xgb_predictor.serializer = sagemaker.serializers.CSVSerializer()
```

Ahora, usaremos una función para:

- Bucle sobre el conjunto de datos de prueba.

- Dividir en mini lotes de filas.

- Convertir esos mini lotes en cargas útiles de strings CSV (primero eliminamos la variable de destino del conjunto de datos).

- Recuperar predicciones de mini lotes invocando el endpoint de XGBoost.

- Recopilar predicciones y convertir la salida CSV que proporciona nuestro modelo en un array NumPy.

```py
def predict(data, predictor, rows=500):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = ''
    for array in split_array:
        predictions = '.'.join([predictions, predictor.predict(array).decode('utf-8')])

    return np.fromstring(predictions[1:],sep=',')

predictions = predict(test_data.drop(['Class'], axis=1).to_numpy(), xgb_predictor)
```

Ahora revisemos la matriz de confusión para ver qué tan bien lo hicimos.

```py
pd.crosstab(index=test_data.iloc[:,0], columns=np.round(predictions), rownames=['actual'], colnames=['predictions'])
```

> Debido a los elementos aleatorios del algoritmo, tus resultados pueden diferir ligeramente.

**Hypertuning**

Podemos usar el Ajuste Automático del Modelo (AMT) de SageMaker donde importa los hiperparámetros y luego proporciona el rango de esos hiperparámetros. En el siguiente ejemplo, supongamos que queremos maximizar el área bajo la curva (AUC), pero no sabemos qué valores de los hiperparámetros eta, alpha, min_child_weight y max_depth usar para entrenar el mejor modelo. Para encontrar los mejores valores, podemos especificar un rango de valores y SageMaker buscará la mejor combinación de ellos para obtener el trabajo de entrenamiento con el AUC más alto.

```py
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner
hyperparameter_ranges = {'eta': ContinuousParameter(0,1),
                        'min_child_weight': ContinuousParameter(0,10),
                        'alpha': ContinuousParameter(0,2),
                        'max_depth': IntegerParameter(1,10)}
```

```py
objective_metric_name = 'validation:auc'
```

```py
tuner = HyperparameterTuner(xgb,
                            objective_metric_name,
                            hyperparameter_ranges,
                            max_jobs = 9,
                            max_parallel_jobs = 3)
```

```py
tuner.fit({'train': s3_input_train, 'validation': s3_input_validation})
```

```py
boto3.client('sagemaker').describe_hyper_parameter_tuning_job(
    HyperParameterTuningJobName = tuner.latest_tuning_job.job_name)['HyperParameterTuningJobStatus']
```

```py
# Devuelve el mejor nombre de trabajo de entrenamiento
tuner.best_training_job()
```

```py
# Implementar el modelo mejor capacitado o especificado por el usuario en un endpoint de Amazon SageMaker
tuner_predictor = tuner.deploy(initial_instance_count=1,
                                instance_type='ml.m4.xlarge')
```

```py
# Crear un serializador
tuner_predictor.serializer = sagemaker.serializers.CSVSerializer()
```

**Limpieza opcional**

Cuando hayas terminado con tu Notebook, el siguiente código eliminará el endpoint alojado que creaste y evitará cualquier cargo.

```py
# Almacenar variables que se utilizarán en el próximo Notebook
%store model_data
%store container
%store test_data

xgb_predictor.delete_endpoint(delete_endpoint_config=True)
```

Amazon SageMaker es parte del nivel gratuito de Amazon Web Services, por lo que si deseas profundizar en esta herramienta de Machine Learning en la nube, puedes crear una cuenta en AWS y experimentar con SageMaker. Lee la información importante a continuación.

> Las cuentas cubiertas por la capa gratuita de AWS no están restringidas en cuanto a lo que pueden lanzar. A medida que avanzas con AWS, es posible que comiences a usar más de lo que cubre la capa gratuita de AWS. Debido a que estos recursos adicionales pueden incurrir en cargos, se requiere un método de pago en la cuenta. La capa gratuita de AWS no cubre todos los servicios de AWS.
