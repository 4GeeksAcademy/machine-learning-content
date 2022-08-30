# Almacenes de datos en la nube

Las computadoras portátiles y de escritorio funcionan bien para tareas rutinarias, pero con el aumento reciente en el tamaño de los conjuntos de datos y la potencia informática necesaria para ejecutar modelos de aprendizaje automático, aprovechar los recursos de la nube es una necesidad para la ciencia de datos.

La computación en la nube es la disponibilidad bajo demanda de los recursos del sistema informático, especialmente el almacenamiento de datos y la potencia informática, sin una gestión activa directa por parte del usuario. Las nubes grandes a menudo tienen funciones distribuidas en múltiples ubicaciones, cada ubicación es un centro de datos.

La computación en la nube se basa en el intercambio de recursos para lograr la coherencia y, por lo general, utiliza un modelo de "pago por uso" que puede ayudar a reducir los gastos de capital, pero tenga cuidado porque también puede generar gastos operativos inesperados para usuarios desprevenidos.

Hay muchos beneficios que podríamos mencionar al implementar la computación en la nube en nuestras organizaciones. Por supuesto, la computación en la nube también puede implicar riesgos. Puede ser útil con la reducción de costos y la inmediatez, pero puede implicar riesgos relacionados con la dependencia y la privacidad.

Las principales características de la computación en la nube son:

- Es autoservicio y bajo demanda.

- Accesible desde todo el mundo y asignación transparente.

- Escalable. Puedes agregar más recursos a medida que los necesites, o puedes reducir los recursos.

- Hay una opción para pagar por lo que usas.

Cuando hablamos de IA, siempre hablamos de niveles de responsabilidad. Los niveles de responsabilidad en la computación en la nube siempre son compartidos, por lo que somos responsables de ciertas tareas y el proveedor de la nube es responsable de otras tareas.

![cloud_computing](../assets/cloud_computing.jpg)

Cuando hablamos de un esquema local, básicamente no estamos en la nube y somos responsables de todo. Cuando migramos a un modelo IaaS (Infraestructura como servicio), significa que vamos a alquilar los servicios de infraestructura al proveedor de la nube, por ejemplo máquinas virtuales, servidores, almacenamiento, etc. Cuando migramos a PaaS (Plataforma como servicio), el proveedor no solo nos ofrece una infraestructura, nos ofrece una plataforma que podemos usar. Finalmente, en Saas (Software as a service), el proveedor se encarga de todo.

¿Cuáles son los costos en los servicios de computación en la nube?

Los servicios en la nube tienen modelos de precios basados ​​en 'pago por uso'. Una ventaja de esto es que no hay costos iniciales de infraestructura y no hay necesidad de comprar infraestructura costosa que quizás no usemos por completo. Otra gran ventaja es que podemos pagar recursos adicionales si los necesitamos, esto es lo que llamamos elasticidad, y por supuesto podemos dejar de pagar esos recursos si ya no los necesitamos.

Si eres dueño de una empresa, hay algunas preguntas que debes responder antes de migrar tu empresa a la nube. Ha habido migraciones exitosas a la nube, pero también ha habido migraciones fallidas. Tenemos que entender que estamos hablando de un proceso de transformación digital. Generalmente, las empresas que inician una transformación digital, tienen cierta madurez en sus procedimientos, en su cultura (recursos humanos), y en sus tecnologías. Las empresas que han fracasado en esta transformación, normalmente se enfocan en las nuevas tecnologías, pero se olvidan de rediseñar sus procesos, o no se preocupan en capacitar a sus recursos humanos en el uso de estas nuevas tecnologías. Después de algunos meses o un par de años, se dan cuenta de que perdieron el tiempo y el dinero. A veces, la razón es que ni siquiera comenzaron con objetivos claros de qué lograr con su transformación digital.

Así que antes de migrar a la nube hazte estas preguntas:

1. ¿Qué quiero lograr al migrar a la nube?

2. ¿Qué servicios deseo migrar a la nube?

3. ¿Qué tipo de modelo debo elegir?

4. ¿Qué empresa elegiré para que me preste ese servicio?

Para responder a la última pregunta, necesitamos saber qué proveedores existen en el mercado. Hay muchos proveedores de computación en la nube, por ejemplo, Digital Ocean, IBM Cloud, OpenStack, etc. Sin embargo, hay tres grandes proveedores en el mercado occidental, AWS, Microsoft Azure y la plataforma Google Cloud, de los cuales AWS tiene la mayor participación de mercado. y es el más utilizado. Además de los servicios de infraestructura central, cada proveedor de la nube ofrece sus ofertas patentadas únicas en NoSQL, Big Data, Analytics, ML y otras áreas similares.

> Muchos proveedores tienen niveles gratuitos. AWS, GCP y Azure los ofrecen.

## Conceptos de computación en la nube

**¿Qué es un Data Lakes (lago de datos)?**

Es un servicio que nos permite almacenar datos en un formato crudo, sin ningún tipo de procesamiento previo. Podemos almacenar datos estructurados, por ejemplo, un archivo de Excel, datos semiestructurados, como un archivo json, y datos no estructurados, por ejemplo, un audio.

Estos Data Lakes están destinados a almacenar datos no estructurados, sin embargo, también pueden almacenar los otros dos tipos de datos. Los lagos de datos nos permiten importar cualquier cantidad de datos en tiempo real, almacenar y etiquetar esos datos de forma segura, analizarlos sin necesidad de mover los datos a una herramienta de análisis separada y, por supuesto, podemos usarlos para crear modelos predictivos.

Los Data Lakes ayudan a las empresas a tomar mejores decisiones porque se convierten en una fuente de información centralizada y estandarizada. Se adaptan fácilmente a los cambios, pueden escalar mucho más que una base de datos relacional y permiten cruzar diversas fuentes de datos, no solo relacionales.

Sin embargo, también tienen dificultades, como grandes inversiones iniciales y costoso mantenimiento. Las personas que trabajarán con este servicio deben estar muy capacitadas porque la integración de diferentes fuentes de datos es muy compleja y siempre existe un gran riesgo de seguridad de comprometer la información de la empresa si el Data Lakes está comprometido. 

**¿Cuál es la diferencia entre un Data Lake y un Data Warehouse?**

Un Data Warehouse (almacén de datos) es un repositorio de datos optimizado y centralizado para analizar datos relacionales que provienen de sistemas transaccionales y aplicaciones comerciales. Los datos se limpian, enriquecen y transforman para que puedan actuar como la única fuente verdadera en la que los usuarios pueden confiar.

Un Data Lake es diferente porque almacena datos relacionales de aplicaciones comerciales y no datos relacionales de aplicaciones móviles, dispositivos IoT y redes sociales. La estructura o el esquema de los datos no se define cuando se capturan los datos.

La principal diferencia entre el Data Lake y el Data Warehouse tiene que ver con el formato en el que se procesan y almacenan los datos. En un Data Warehouse siempre encontraremos datos estructurados y preprocesados. Decidir si tener un Data Lake o un Data Warehouse depende del tipo de datos con los que trabajarás y también de la frecuencia con la que se actualizarán y consultarán los datos. Los Data Warehouse son bases de datos analíticas, por lo que no están destinados a consultas y actualizaciones frecuentes.

Según las necesidades, una empresa puede tener un Data Lake y un Data Warehouse. Ambas estructuras de datos son muy valiosas.

**¿Qué es un data mart?**

A Datawarehouse is composed of datamarts. Data marts are small databases oriented to specific topics from the organization. For example a datamart for each department (Marketing, Purchases, etc). The process that extracts, transforms and loads the data from datamarts into the data warehouse is known as **ETL**. This ETL processes are normally done by batches, which are loaded during a specific time of the day, normally at nights when the work loads are not so high.

A data warehouse, compared to a data mart, tries to centralize information, and fight against having multiple truths from having multiple databases.

## Cloud computing providers

Machine learning is a critical element of the data science process, but training ML models is often a time consuming process that requires a lot of resources. Machine learning and deep learning models involve thousands of training iterations. You need these extensive amounts of iterations to produce the most accurate model. For example, if you have a set of training samples with only 1TB of data, 10 iterations of this training set will require 10TB of speed with which the data transfer takes place between the hard disk drive and RAM. In the past, gaining access to ML resources was difficult and expensive. Today, many cloud computing vendors offer resources for data science in the cloud.

Cloud computing enables you to model storage capacity and handle loads at scale, or to scale the processing across nodes. For example, AWS offers Graphics Processing Unit (GPU) instances with 8–256GB memory capacity. These instances are priced at an hourly rate. GPUs are specialized processors designed for complex image processing.

### Amazon Web Services (AWS)

AWS is the most used cloud platform today. This cloud solution allows us to virtually execute any application on the cloud, from web applications to IoT solutions and big data. In AWS, as in any cloud provider, we have to choose the region in the world in which the servers that provide the service, are installed. Remember to always choose a region that is the closest possible to the client that will use the resources. Amazon Web Services has a lot of tools and services but in this lesson, we want to mention some of them that can help you improving your machine learning models.

**S3** is an Amazon data storage service. One of the most important products from Amazon. **S3** is an object based storage service that allows to store as much data as you want because the scaling is automatic. When we talk about object based we mean the minimal unit, in this case, the files. 

All files or objects are stored in buckets. We can think of a bucket as a folder where we store our files. A bucket is a container that can store different types of objects. We can not create a bucket with a name that already exists. Bucket names in S3 are unique and we can configure if buckets are public or private. Inside a bucket we can modify a file and have its versions during time.

S3 storage format: https://[bucket name].s3.amazonaws.com/video.mp4

**Cloud9** is like a Visual Studio Code but from AWS. It allows us to create environments. When we configure them, in the background this creates and executes a virtual machine service called Elastic Compute Cloud (EC2). 

**SageMaker** is a fully-managed machine learning platform for data scientists and developers. The platform also runs on Elastic Compute Cloud (EC2), and enables you to build machine learning models, organize your data, and scale your operations. Machine learning applications on SageMaker range from speech recognition, computer vision, and recommendations. The AWS marketplace offers models to use, instead of starting from scratch. You can start training and optimizing your model. The most common choices are frameworks like Keras, TensorFlow, and PyTorch. SageMaker can optimize and configure these frameworks automatically, or you can train them yourself. You can also develop your own algorithm by building it in a Docker container and you can use a Jupyter notebook to build your machine learning model, and visualize your data.

Link of the Amazon Web Services: `aws.amazon.com/console`

Where to learn AWS skills: `aws.amazon.com/es/training/awsacademy/`

### Google Cloud Platform (GCP)

Google Cloud Platform offers you three ways to carry out machine learning:

**Auto ML** to train state-of-the-art deep learning models on your data without writing any code, for begineers. 

Auto ML is a cloud-based machine learning platform built for inexperienced users. You can upload your datasets, train models, and deploy them on the website. AutoML integrates with all Google’s services and stores data in the cloud. You can deploy trained models via the REST API interface. There are a number of available AutoML products you can access via a graphical interface. This includes training models on structured data, image and video processing services, and a natural language processing and translation engine.

**BigQuery ML** to build custom ML models, train them and make predictions on structured data using just SQL. Use BigQuery ML for quick problem formulation, experimentation, and easy, low-cost machine learning. Once you identify a viable ML problem using BQML, use Auto ML for code-free, state-of-the-art models. Hand-roll your own custom models only for problems where you have lots of data and enough time/effort to devote.

**Cloud ML Engine** to build custom, deep learning models using Keras with a TensorFlow backend, for more experienced data professionals. It can be used to train a complex model by leveraging GPU and Tensor Processing Unit (TPU) infrastructure. You can also use the service to deploy an externally trained model. Cloud ML automates all monitoring and resource provisioning processes for running the jobs. Besides hosting and training, Cloud ML can also perform hyperparameter tuning that influences the accuracy of predictions. Without hyperparameter tuning automation, data scientists need to manually experiment with multiple values while evaluating the accuracy of the results.

TensorFlow is an open source software library that uses data-flow graphs for numerical operations. Mathematical operations in these graphs are represented by nodes, while edges represent data transferred from one node to another. Data in TensorFlow is represented as tensors, which are multidimensional arrays. TensorFlow is usually used for deep learning research and practice. TensorFlow is cross-platform. You can run it on GPUs, CPUs, TPUs, and mobile platforms.

Choose between them based on your skill set, how important additional accuracy is, and how much time/effort you are willing to devote to the problem. 

Check out the Google Cloud Platform free program to discover new tools for your machine learning models.

Link of the Google Cloud Platform: `console.cloud.google.com`

Where to learn GCP skills: `go.qwiklabs.com`, `cloudskillsboost.google`

### Microsoft Azure

Just like the SageMaker of Amazon and ML Engine of Google, Azure AI is the answer of Microsoft to Amazon and Google. Moreover, Azure AI offers a range of open and comprehensive platforms for the building, evaluating, and deployment of machine learning models and many more features supporting various AI frameworks such as PyTorch, TensorFlow, Sci-kit Learn, and more.

Compared to AWS, Azure machine learning offerings are more flexible in terms of out-of-the-box algorithms. 

**Azure ML** is a huge library of pre-trained, pre-packaged machine learning algorithms. Azure ML Service also provides an environment for implementing these algorithms and applying them to real-world applications. The UI of Azure ML enables you to build machine learning pipelines that combine multiple algorithms. You can use the UI to train, test, and evaluate models.

Azure ML also provides solutions for Artificial Intelligence (AI). This includes visualization and other data that can help understand model behavior, and compare algorithms to find the best option.

Link of the Microsoft Azure platform: `azure.microsoft.com`

How to learn Microsoft skills: [Microsoft Learn](docs.microsoft.com/en-us/learn), [Azure fundamentals](docs.microsoft.com/en-us/certifications/azure-fundamentals/).


**Source:**

-https://en.wikipedia.org/wiki/Cloud_computing

-https://www.c-sharpcorner.com/article/aws-vs-azure-vs-google-cloud-comparative-analysis-of-cloud-platforms-for-machi/

-https://towardsdatascience.com/data-science-in-the-cloud-239b795a5792

-https://medium.com/ibm-data-ai/machine-learning-in-google-cloud-with-bigquery-25d40b158f91

-https://medium.com/@rajdeepmondal/predicting-cab-fare-for-chicago-using-bqml-395126343c92

-https://windsor.ai/how-to-get-your-analytics-crm-and-media-data-into-bigquery/

-https://towardsdatascience.com/build-a-useful-ml-model-in-hours-on-gcp-to-predict-the-beatles-listeners-1b2322486bdf

-https://towardsdatascience.com/choosing-between-tensorflow-keras-bigquery-ml-and-automl-natural-language-for-text-classification-6b1c9fc21013

-https://towardsdatascience.com/running-jupyter-notebook-in-google-cloud-platform-in-15-min-61e16da34d52

-https://towardsdatascience.com/automated-machine-learning-on-the-cloud-in-python-47cf568859f
