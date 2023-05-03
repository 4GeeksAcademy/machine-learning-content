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

![cloud_computing](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/cloud_computing.jpg?raw=true)

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

**¿Qué es un Data Mart?**

Un Data Warehouse se compone de Data Marts. Los Data Marts son pequeñas bases de datos orientadas a temas específicos de la organización. Por ejemplo un Data Mart para cada departamento (Marketing, Compras, etc). El proceso que extrae, transforma y carga los datos de los Data Marts en el almacén de datos se conoce como **ETL**. Estos procesos ETL normalmente se realizan por lotes, los cuales se cargan en un momento específico del día, normalmente en las noches cuando las cargas de trabajo no son tan altas.

Un Data Warehouse, en comparación con un Data Mart, intenta centralizar la información y luchar contra tener múltiples verdades por tener múltiples bases de datos.

## Proveedores de computación en la nube

Machine Learning es un elemento crítico del proceso de ciencia de datos, pero entrenar modelos de ML suele ser un proceso lento que requiere una gran cantidad de recursos. Los modelos de Machine Learning y aprendizaje profundo implican miles de iteraciones de entrenamiento. Necesitas estas grandes cantidades de iteraciones para producir el modelo más preciso. Por ejemplo, si tienes un conjunto de muestras de entrenamiento con solo 1 TB de datos, 10 iteraciones de este conjunto de entrenamiento requerirán 10 TB de velocidad con la que se realiza la transferencia de datos entre el disco duro y la RAM. En el pasado, obtener acceso a los recursos de ML era difícil y costoso. Hoy en día, muchos proveedores de computación en la nube ofrecen recursos para la ciencia de datos.

La computación en la nube te permite modelar la capacidad de almacenamiento y manejar cargas a escala, o escalar el procesamiento entre nodos. Por ejemplo, AWS ofrece instancias de unidades de procesamiento de gráficos (GPU) con una capacidad de memoria de 8 a 256 GB. Estas instancias tienen un precio por hora. Las GPU son procesadores especializados diseñados para el procesamiento de imágenes complejas.

### Servicios web de Amazon (AWS)

AWS es la plataforma en la nube más utilizada en la actualidad. Esta solución en la nube nos permite ejecutar virtualmente cualquier aplicación en la nube, desde aplicaciones web hasta soluciones IoT y big data. En AWS, como en cualquier proveedor de nube, tenemos que elegir la región del mundo en la que están instalados los servidores que dan el servicio. Recuerda elegir siempre una región que sea lo más cercana posible al cliente que utilizará los recursos. Amazon Web Services tiene muchas herramientas y servicios, pero en esta lección queremos mencionar algunos de ellos que pueden ayudarte a mejorar tus modelos de Machine Learning.

**S3** es un servicio de almacenamiento de datos de Amazon. Uno de los productos más importantes de Amazon. **S3** es un servicio de almacenamiento basado en objetos que permite almacenar tantos datos como quieras porque el escalado es automático. Cuando hablamos de basado en objetos nos referimos a la unidad mínima, en este caso, los archivos.

Todos los archivos u objetos se almacenan en cubos. Podemos pensar en un depósito como una carpeta donde almacenamos nuestros archivos. Un cubo es un contenedor que puede almacenar diferentes tipos de objetos. No podemos crear un depósito con un nombre que ya existe. Los nombres de los depósitos en S3 son únicos y podemos configurar si los depósitos son públicos o privados. Dentro de un cubo podemos modificar un archivo y tener sus versiones en el tiempo.

Formato de almacenamiento S3: https://[bucket name].s3.amazonaws.com/video.mp4

**Cloud9** es como Visual Studio Code pero de AWS. Nos permite crear ambientes. Cuando los configuramos, en segundo plano crea y ejecuta un servicio de máquina virtual llamado Elastic Compute Cloud (EC2).

**SageMaker** es una plataforma de Machine Learning completamente administrada para desarrolladores y científicos de datos. La plataforma también se ejecuta en Elastic Compute Cloud (EC2) y te permite crear modelos de Machine Learning, organizar tus datos y escalar tus operaciones. Las aplicaciones de Machine Learning en SageMaker van desde reconocimiento de voz, visión artificial y recomendaciones. El mercado de AWS ofrece modelos para usar, en lugar de comenzar desde cero. Puedes comenzar a entrenar y optimizar tu modelo. Las opciones más comunes son marcos como Keras, TensorFlow y PyTorch. SageMaker puede optimizar y configurar estos marcos automáticamente, o puedes entrenarlos tu mismo. También puedes desarrollar tu propio algoritmo al construirlo en un contenedor Docker y puedes usar un cuaderno Jupyter para construir tu modelo de Machine Learning y visualizar tus datos.

Enlace de los Servicios Web de Amazon: `aws.amazon.com/console`

Dónde aprender habilidades de AWS: `aws.amazon.com/es/training/awsacademy/`

### Plataforma en la nube de Google (GCP)

La Plataforma en la nube de Google (GCP) te ofrece tres formas de llevar a cabo Machine Learning:

**Auto ML** para entrenar modelos de aprendizaje profundo de última generación en tus datos sin escribir ningún código, para principiantes.

Auto ML es una plataforma de Machine Learning basada en la nube creada para usuarios sin experiencia. Puedes cargar tus conjuntos de datos, entrenar modelos e implementarlos en el sitio web. AutoML se integra con todos los servicios de Google y almacena datos en la nube. Puedes implementar modelos entrenados a través de la interfaz API REST. Hay una serie de productos de AutoML disponibles a los que puedes acceder a través de una interfaz gráfica. Esto incluye modelos de entrenamiento en datos estructurados, servicios de procesamiento de imágenes y videos, y un motor de procesamiento y traducción de lenguaje natural.

**BigQuery ML** para crear modelos de ML personalizados, entrenarlos y hacer predicciones sobre datos estructurados usando solo SQL. Use BigQuery ML para la formulación rápida de problemas, la experimentación y Machine Learning fácil y de bajo costo. Una vez que identifiques un problema de Machine Learning viable mediante BQML, utiliza Auto Machine Learning para obtener modelos de última generación sin código. Haz rodar a mano tus propios modelos personalizados solo para problemas en los que tengas muchos datos y suficiente tiempo/esfuerzo para dedicar.

**Cloud ML Engine** para crear modelos personalizados de aprendizaje profundo utilizando Keras con un backend de TensorFlow, para profesionales de datos más experimentados. Se puede usar para entrenar un modelo complejo aprovechando la GPU y la infraestructura de la Unidad de procesamiento de tensores (TPU). También puedes usar el servicio para implementar un modelo entrenado externamente. Cloud ML automatiza todos los procesos de supervisión y aprovisionamiento de recursos para ejecutar los trabajos. Además de alojamiento y capacitación, Cloud ML también puede realizar ajustes de hiperparámetros que influyen en la precisión de las predicciones. Sin la automatización de ajuste de hiperparámetros, los científicos de datos necesitan experimentar manualmente con múltiples valores mientras evalúan la precisión de los resultados.

TensorFlow es una biblioteca de software de código abierto que utiliza gráficos de flujo de datos para operaciones numéricas. Las operaciones matemáticas en estos gráficos están representadas por nodos, mientras que los bordes representan datos transferidos de un nodo a otro. Los datos en TensorFlow se representan como tensores, que son arrays multidimensionales. TensorFlow generalmente se usa para la investigación y la práctica del aprendizaje profundo. TensorFlow es multiplataforma. Puede ejecutarlo en GPU, CPU, TPU y plataformas móviles.

Elije entre ellos en función de tu conjunto de habilidades, la importancia de la precisión adicional y la cantidad de tiempo/esfuerzo que estás dispuesto a dedicar al problema.

Consulte el programa gratuito Google Cloud Platform (Plataforma en la nube de Google) para descubrir nuevas herramientas para tus modelos de Machine Learning.

Enlace de la plataforma Google Cloud: `console.cloud.google.com`

Dónde aprender habilidades de GCP: `go.qwiklabs.com`, `cloudskillsboost.google`

### Microsoft Azure

Al igual que SageMaker de Amazon y ML Engine de Google, Azure AI es la respuesta de Microsoft a Amazon y Google. Además, Azure AI ofrece una gama de plataformas abiertas e integrales para la creación, evaluación e implementación de modelos de Machine Learning y muchas más funciones compatibles con varios marcos de IA, como PyTorch, TensorFlow, Sci-kit Learn y más.

En comparación con AWS, las ofertas de Machine Learning de Azure son más flexibles en términos de algoritmos listos para usar. 

**Azure ML** es una biblioteca enorme de algoritmos de Machine Learning empaquetados y entrenados previamente. Azure ML Service también proporciona un entorno para implementar estos algoritmos y aplicarlos a aplicaciones del mundo real. La interfaz de usuario de Azure ML te permite crear pipelines de Machine Learning que combinan varios algoritmos. Puedes usar la interfaz de usuario para entrenar, probar y evaluar modelos.

Azure ML también proporciona soluciones para Inteligencia Artificial (IA). Esto incluye visualización y otros datos que pueden ayudar a comprender el comportamiento del modelo y comparar algoritmos para encontrar la mejor opción.

Enlace de la plataforma Microsoft Azure: `azure.microsoft.com`

Cómo aprender habilidades de Microsoft: [Microsoft Learn](docs.microsoft.com/en-us/learn), [Azure fundamentals](docs.microsoft.com/en-us/certifications/azure-fundamentals/).


**Fuente:**

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
