## Computación en la nube

La **computación en la nube** (*cloud computing*) es un modelo de entrega de servicios de tecnología a través de Internet. En lugar de tener que comprar y mantener servidores y hardware propios, las empresas y/o los usuarios pueden acceder a recursos informáticos, como servidores, almacenamiento, bases de datos, redes y software, a través de proveedores de servicios en la nube.

En esencia, el cloud computing permite a las organizaciones y a los individuos utilizar recursos informáticos de manera flexible y bajo demanda, pagando solo por lo que realmente utilizan. Esto proporciona varias ventajas, como:

- **Escalabilidad**: Los recursos pueden escalarse hacia arriba o hacia abajo según las necesidades, lo que permite un uso eficiente de los recursos y la capacidad de manejar cargas de trabajo variables.
- **Acceso remoto**: Los usuarios pueden acceder a sus aplicaciones y datos desde cualquier lugar con conexión a Internet, lo que facilita la colaboración y el trabajo remoto.
- **Coste reducido**: Al no requerir inversión en infraestructura física ni hardware, las organizaciones pueden evitar costes iniciales y gastos de mantenimiento. Solo pagan por los recursos que consumen.
- **Actualizaciones y mantenimiento simplificados**: Los proveedores de servicios en la nube se encargan de la administración de hardware y software, lo que libera a los usuarios de la carga de mantenimiento.
- **Flexibilidad**: Ofrecen una variedad de opciones y configuraciones para satisfacer las necesidades específicas de diferentes usuarios y empresas.

Hay tres modelos principales de servicios en la nube:

- **Infraestructura como servicio** (*IaaS*, *Infraestructure as a Service*): Proporciona acceso a recursos de infraestructura básicos, como servidores virtuales, almacenamiento y redes. En este tipo de servicio, básicamente obtenemos una parte de una(s) computadora(s) gigante(s) en la nube. Es como alquilar solo el hardware de una computadora. Aquí, nosotros somos responsables de administrar todo, como instalar sistemas operativos y programas. Puede ser muy flexible, como si estuviéramos construyendo nuestra propia casa y eligiendo cada detalle. Ejemplos son servicios como Amazon Web Services (AWS) y Microsoft Azure.
- **Plataforma como servicio** (*PaaS*, *Platform as a Service*): Ofrece un entorno de desarrollo y ejecución para que los desarrolladores construyan, implementen y gestionen aplicaciones sin preocuparse por la infraestructura subyacente. Este servicio es como si nos diesen un lugar en la nube para construir nuestra casa, pero ya tenemos algunas herramientas y materiales específicos listos para usar. Podemos construir aplicaciones usando esas herramientas sin preocuparnos por la infraestructura básica. Es como si alguien nos diera un kit de construcción para hacer nuestra casa en lugar de tener que diseñar cada pieza nosotros mismo. Ejemplos incluyen Google App Engine y Heroku.
- **Software como servicio** (*SaaS*, *Software as a Service*): Ofrece aplicaciones completas a través de la nube. Los usuarios pueden acceder y utilizar el software a través de un navegador web sin necesidad de instalar o mantenerlo localmente. Con SaaS, en realidad no tenemos que construir nada. Es como si ya tuviéramos una casa completamente amueblada y lista para vivir. Solo necesitamos entrar y empezar a vivir. En lugar de instalar programas en nuestro ordenador, accedemos a ellos a través de Internet. Por ejemplo, como sucede en Gmail o Google Docs. No necesitamos preocuparnos por las cosas técnicas, solo usar lo que está ahí.

|   | IaaS (Infraestructura como Servicio) | PaaS (Plataforma como Servicio) | SaaS (Software como Servicio) |
|!--|!-------------------------------------|!--------------------------------|!------------------------------|
| Nivel de Abstracción | Baja | Mediana | Alta |
| Responsabilidad de Gestión | Usuario (Sistemas Operativos, Redes) | Proveedor (Plataforma, Middleware) | Proveedor (Aplicación) |
| Flexibilidad | Alta | Moderada | Baja |
| Escalabilidad | Alta | Moderada | Limitada |
| Desarrollo de aplicaciones | Depende del usuario | Basado en Plataforma | No necesario, solo uso |
| Ejemplos | Máquinas virtuales (AWS, Azure) | Google App Engine, Heroku | Salesforce, Google Workspace |

![cloud_computing](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/cloud_computing.jpg?raw=true)

La computación en la nube, en términos del Machine Learning y, más allá, de la Inteligencia Artificial, hoy en día se utiliza en todas sus formas; desde utilizar herramientas de terceros para desarrollar modelos como entornos de desarrollo completamente integrados en la nube, pasando por desarrollos locales y el despliegue en la nube (éste último el más utilizado).

### Machine learning en la nube

A pesar de que hay un catálogo infinito y muy bien repartido de servicios para trabajar en el ámbito del machine learning, algunos de los más destacados y conocidos son:

- **Amazon Web Services** (*AWS*): AWS ofrece una amplia gama de servicios para machine learning, como *Amazon SageMaker*, que es una plataforma integral para construir, entrenar e implementar modelos de machine learning. También proporciona servicios específicos como Amazon Rekognition para reconocimiento de imágenes y Amazon Polly para síntesis de voz.
- **Microsoft Azure**: Azure Machine Learning es la oferta de machine learning en la nube de Microsoft. Proporciona herramientas y servicios para desarrollar, entrenar y desplegar modelos de machine learning. También incluye Azure Cognitive Services para capacidades de procesamiento de lenguaje natural y visión computarizada.
- **Google Cloud Platform** (*GCP*): Google Cloud ofrece Google Cloud AI Platform, que permite construir, entrenar y desplegar modelos de machine learning utilizando TensorFlow y otros frameworks populares. También ofrece servicios de procesamiento de lenguaje natural a través de la API Cloud Natural Language y el reconocimiento de imágenes a través de la API Cloud Vision.
- **IBM Cloud**: IBM ofrece IBM Watson, una plataforma de inteligencia artificial que abarca machine learning y análisis de datos. Watson Studio es una herramienta para crear y entrenar modelos, y Watson Machine Learning permite implementarlos. También ofrecen servicios de procesamiento de lenguaje natural y análisis de texto.
- **Alibaba Cloud**: El proveedor chino Alibaba Cloud ofrece servicios de machine learning a través de Alibaba Cloud Machine Learning Platform, que incluye herramientas para construir, entrenar y desplegar modelos. También ofrece servicios de procesamiento de lenguaje natural y visión computarizada.
- **Oracle Cloud**: Oracle Cloud proporciona servicios de machine learning y analítica avanzada a través de Oracle Cloud Infrastructure Data Science. Ofrece capacidades de modelado, entrenamiento y despliegue, así como integración con otras herramientas de Oracle.

### Almacenes de datos en la nube

Los almacenes de datos en la nube son sistemas diseñados para almacenar grandes cantidades de información de manera eficiente y escalable. Con el aumento reciente en el tamaño de los conjuntos de datos y la potencia informática necesaria para ejecutar modelos de aprendizaje automático, aprovechar los recursos de la nube es una necesidad para la ciencia de datos.

En la gestión de datos hay conceptos clave, como 




# Almacenes de datos en la nube




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
