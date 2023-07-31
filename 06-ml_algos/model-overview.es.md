## Resumen de los modelos de Aprendizaje Supervisado

A continuación se facilita un breve repaso sobre los diferentes modelos estudiados y cuándo y por qué se utilizan, a modo de guía práctica para saber elegir siempre la mejor opción:

### Clasificadores y regresores

Dependiendo de la naturaleza del modelo y su definición matemática, se podrían utilizar para clasificar, hacer predicciones (regresión) o para ambas:

| **Modelo** | **Clasificación** | **Regresión** |
|:-----------|:------------------|:--------------|
| Regresión Logística | ✅ | ❌ |
| Regresión Lineal | ❌ | ✅ |
| Regresión Lineal Regularizada | ❌ | ✅ |
| Árbol de Decisión | ✅ | ✅ |
| Random Forest | ✅ | ✅ |
| Boosting | ✅ | ✅ |
| Naive Bayes | ✅ | ❌ |
| K vecinos cercanos | ✅ | ✅ |

Además, podemos implementarlo fácilmente utilizando las siguientes funciones:

| **Modelo** | **Clasificación** | **Regresión** |
|:-----------|:------------------|:--------------|
| Regresión Logística | `sklearn.linear_model.LogisticRegression` | - |
| Regresión Lineal | - | `sklearn.linear_model.LinearRegression` |
| Regresión Lineal Regularizada | - | `sklearn.linear_model.Lasso`<br />`sklearn.linear_model.Ridge` |
| Árbol de Decisión | `sklearn.tree.DecisionTreeClassifier` | `sklearn.tree.DecisionTreeRegressor` |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | `sklearn.ensemble.RandomForestRegressor` |
| Boosting | `sklearn.ensemble.GradientBoostingClassifier`<br />`xgboost.XGBClassifier` | `sklearn.ensemble.GradientBoostingRegressor`<br />`xgboost.XGBRegressor` |
| Naive Bayes | `sklearn.naive_bayes.BernoulliNB`<br />`sklearn.naive_bayes.MultinomialNB`<br />`sklearn.naive_bayes.GaussianNB` | - |
| K vecinos cercanos | `sklearn.neighbors.KNeighborsClassifier` | `sklearn.neighbors.KNeighborsRegressor` |

### Descripción y cuándo utilizar

Saber cuál es el papel de cada modelo y cuándo podemos/debemos utilizarlo es vital para realizar nuestro trabajo de forma eficiente y profesional. A continuación vemos una tabla comparativa que aborda esta información:

| **Modelo** | **Utilidad** | **Uso recomendado** | **Ejemplos de casos de uso** |
|:-----------|:-------------|:--------------------|:-----------------------------|
| Regresión Logística | Se utiliza para clasificar sucesos binarios o multiclase (menos común). | Útil cuando la relación entre las características y la variable objetivo es lineal. Requiere que las características sean linealmente independientes. | Clasificación de correos electrónicos como spam o no spam. Detección de enfermedades basada en síntomas y pruebas médicas. Predicción de la probabilidad de un cliente de comprar un producto. |
| Regresión Lineal | Se utiliza para predecir valores numéricos continuos. | Útil cuando la relación entre las características y la variable objetivo es lineal. Requiere que las características tengan una correlación significativa con la variable objetivo para obtener buenos resultados. | Predicción del precio de una vivienda en función de su tamaño, número de habitaciones y ubicación. Estimación del rendimiento académico de un alumno basado en sus horas de estudio y calificaciones previas. |
| Regresión Lineal Regularizada | Similar a la regresión lineal pero incluyendo un parámetro para evitar el sobreajuste. | Útil cuando hay multicolinealidad entre las características o para evitar el sobreajuste del modelo tradicional. | Predicción del precio de un automóvil en función de características como el año de fabricación, la marca, el modelo, aplicando regularización para evitar el sobreajuste. Estimación del salario de un empleado basado en su experiencia laboral y nivel de educación, con regularización para reducir la influencia de características irrelevantes. |
| Árbol de Decisión | Se utiliza para clasificar o predecir valores numéricos continuos. | Útil cuando las relaciones entre las características y la variable objetivo son no lineales o complejas. Pueden manejar características numéricas y categóricas sin necesidad de estandarización. | Predicción de la lealtad de los clientes en función de su historial de compras. Clasificación de películas según su genéro y características. Detección de fraudes en transacciones financieras. |
| Random Forest | Se utiliza para clasificar o predecir valores numéricos continuos. Combina múltiples árboles de decisión. | Útil cuando el conjunto de datos es grande y complejo, evitando el sobreajuste y mejorando la precisión. | Clasificación de imágenes para reconocimiento de objetivos. Predicción de precios de viviendas basada en múltiples características. Diagnóstico de enfermedades basado en múltiples pruebas médicas. |
| Boosting | Se utiliza para clasificar o predecir valores numéricos continuos. Combina múltiples árboles de decisión creados secuencialmente para corregir los errores de los modelos anteriores. | Útil cuando se desean modelos más precisos que los modelos individuales y se tiene suficiente potencia de cómputo. | Análisis de sentimiento en texto para clasificar opiniones como positivas o negativas. Detección de comportamiento anómalo en sistemas de seguridad. Predicción de ingresos de un cliente en función de múltiples factores. |
| Naive Bayes | Se utiliza para clasificar sucesos binarios o multiclase. | Útil cuando existe independencia condicional entre las características (ya que es el fundamento del modelo). Funciona bien cuando el conjunto de datos contiene características categóricas o representan frecuencias de palabras. | Problemas de clasificación de texto y tareas de categorización. Clasificación de reseñas de productos como positivas o negativas. |
| K vecinos cercanos | Se utiliza para clasificar o predecir valores numéricos continuos. | Útil cuando se tiene un conjunto de datos con relaciones no lineales y cuando la estructura local de los datos es importante. El conjunto de datos debe estar estandarizado. | Recomendación de productos similares en un sitio de comercio electrónico. Clasificación de enfermedades en función de síntomas y antecedentes médicos. Predicción del precio de una casa basado en precios similares de propiedades cercanas. |

Al margen de los casos de ejemplo y de sus definiciones, el criterio del desarrollador y profesional prevalece, y dependiendo del caso de uso, de los datos y de sus características, a veces incluso modelos que no están optimizados para eso pueden ser útiles.