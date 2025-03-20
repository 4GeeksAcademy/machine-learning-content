---
description: >-
  Aprende lo esencial del Análisis Exploratorio de Datos (EDA) y su papel en el Machine Learning. ¡Descubre cómo visualizar datos e identificar patrones!
---
## Análisis exploratorio de datos

El **análisis exploratorio de datos** (*EDA*, *Exploratory data analysis*) es el enfoque de partida y fundamental de cualquier análisis de datos, ya que tiene como objetivo comprender las características principales de un conjunto de datos antes de realizar análisis más avanzados o modelados posteriores.

El EDA implica lo siguiente:

- **Visualización de datos**: Usando gráficos como histogramas, box plots, scatter plots y muchos otros para visualizar la distribución de los datos, las relaciones entre las variables y cualquier anomalía o particularidad en los datos.
- **Identificación de anomalías**: Detectando y, a veces, tratando valores atípicos o datos faltantes que podrían afectar posteriores análisis.
- **Formulación de hipótesis**: A partir de la exploración, los analistas pueden comenzar a formular hipótesis que luego se testearán en un análisis más detallado o en el modelado.

El objetivo principal del EDA es ver lo que los datos pueden decirnos más allá de la tarea formal del modelado. Sirve para asegurarnos de que la subsiguiente fase de modelado o análisis se realice de manera adecuada y que cualquier conclusión se base en una correcta comprensión de la estructura de los datos.

### Análisis descriptivo y EDA

El análisis descriptivo y el EDA, dependiendo de dónde se implante, pueden ser equivalentes, pero aquí los distinguiremos por sus principales diferencias:

1. **Análisis descriptivo**: Se centra en describir las características principales de un conjunto de datos mediante estadísticas descriptivas, como la media, la mediana, el rango, etcétera. Su objetivo principal es proporcionar una descripción clara y resumida de los datos.
2. **EDA**: Va un paso más allá, ya que se centra en explorar patrones, relaciones, anomalías, etc., en los datos utilizando gráficos y estadísticas más sofisticadas. Su objetivo principal es entender la estructura de los datos, relaciones entre variables y formular hipótesis o intuiciones para posteriores análisis o modelado.

En el mundo real, tras la captura de información, se puede comenzar con un análisis descriptivo para obtener un sentido básico del conjunto de datos y luego proceder al EDA para una exploración más profunda. Sin embargo, en muchos casos, el término EDA se usa para englobar ambos procesos, ya que el límite entre ambos es algo difuso y depende de cada grupo de trabajo, empresa, etcétera.

### Flujo de Machine Learning

El flujo ideal de Machine Learning debe contener las siguientes fases:

1. **Definición del problema**: Se identifica una necesidad que se trata de solucionar utilizando Machine Learning.
2. **Obtención del conjunto de datos**: Una vez se ha definido el problema a resolver, es necesario un proceso de captura de datos para resolverlo. Para ello, se pueden utilizar fuentes como bases de datos, APIs y datos provenientes de cámaras, sensores, etcétera.
3. **Almacenar la información**: La mejor forma de almacenar la información para que pueda nutrir el proceso de Machine Learning es almacenarla en una base de datos. Debemos evitar los ficheros planos ya que no son seguros ni óptimos. Considera incluirlos en una base de datos.
4. **Análisis descriptivo**: Los datos en crudo almacenados en una base de datos pueden ser una gran y muy valiosa fuente de información. Antes de comenzar a simplificarlos y a explotarlos con el EDA, debemos conocer sus medidas estadísticas fundamentales: medias, modas, distribuciones, desviaciones, etcétera. Conocer las distribuciones de los datos es vital para poder seleccionar un modelo acorde.
5. **EDA**: Este paso es vital para asegurar que nos quedamos con las variables estrictamente necesarias y eliminamos las que no son relevantes o no aportan información. Además, nos permite conocer y analizar las relaciones entre las variables y los valores raros.
6. **Modelado y optimización**: Con los datos listos solo nos queda modelizar el problema. Con las conclusiones de los dos pasos anteriores debemos analizar qué modelo se ajusta mejor a los datos y entrenarlo. Optimizar un modelo después del primer entrenamiento es totalmente necesario a no ser que se ejecute sin errores.
7. **Despliegue**: Para que el modelo sea capaz de aportar valor a nuestros procesos o a nuestros clientes, es necesario que se pueda consumir. El despliegue es la implantación del modelo en un entorno controlado en el que se pueda ejecutar y utilizar para predecir con los datos del mundo real.
