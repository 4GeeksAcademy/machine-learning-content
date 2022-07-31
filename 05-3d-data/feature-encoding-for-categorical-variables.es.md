# Codificación de Características para variables categóricas

Esta es una fase importante en el proceso de limpieza de datos. La codificación de características implica reemplazar clases en una variable categórica con números reales. Por ejemplo, el aula A, el aula B y el aula C podrían codificarse como 2,4,6.

**¿Cuándo debemos codificar nuestras características? ¿Por qué?**

Codificamos características cuando son categóricas. Convertimos datos categóricos numéricamente porque las matemáticas generalmente se hacen usando números. Una gran parte del procesamiento del lenguaje natural es convertir texto en números. Así, nuestros algoritmos no pueden ejecutar y procesar datos si esos datos no son numéricos. Siempre debemos codificar características categóricas para que puedan ser procesadas por algoritmos numéricos, y así los algoritmos de Machine Learning puedan aprender de ellas.

## Tipos de variables categóricas

Hay dos tipos de variables categóricas, nominales y ordinales. Antes de sumergirnos en la codificación de características, es importante que primero contrastemos la diferencia entre una variable nominal y una variable ordinal.

Como ya explicamos en la lección de Ingeniería de Características, una variable nominal es una variable categórica donde sus datos no siguen un orden lógico. Algunos ejemplos son:

- Sexo (masculino o femenino).

- Colores (Rojo, Azul, Verde).

- Partido político (Demócrata o Republicano).

Una variable ordinal, por otro lado, también es una variable categórica excepto que sus datos siguen un orden lógico. Algunos ejemplos de datos ordinales incluyen:

- Nivel socioeconómico (ingresos bajos, ingresos medios o ingresos altos).

- Nivel educativo (bachillerato, licenciatura, maestría o doctorado).

- Calificación de satisfacción (extremadamente disgusto, disgusto, neutral, me gusta o me gusta mucho).

## Métodos para codificar datos categóricos:

#### 1. Codificación Nominal

A cada categoría se le asigna un valor numérico que no representa ningún orden. Por ejemplo: [negro, rojo, blanco] podría codificarse como [3,7,11].

Exploraremos dos formas diferentes de codificar variables nominales, una usando Scikit-learn OneHotEncoder y la otra usando Pandas get_dummies.

**Scikit-learn OneHotEncoder**

Cada categoría se transforma en una nueva característica binaria, con todos los registros marcados con 1 para Verdadero o 0 para Falso. Por ejemplo: [Florida, Virginia, Massachussets] podría codificarse como state_Florida = [1,0,0], state_Virginia = [0,1,0], state_Massachussets = [0,0,1].

En el caso de una característica llamada 'Gender' (Género) con los valores únicos de Hombre y Mujer, OneHotEncoder crea dos columnas para representar las dos categorías en la columna de género, una para hombre y otra para mujer.

Las pasajeras recibirán un valor de 1 en la columna femenina y un valor de 0 en la columna masculina. Por el contrario, los pasajeros masculinos recibirán un valor de 0 en la columna femenina y un valor de 1 en la columna masculina.

```py
from sklearn.preprocessing import OneHotEncoder

# Crear una instancia de OneHotEncoder
ohe = OneHotEncoder()

# Aplicar OneHotEncoder a la columna de gender
ohe.fit_transform(data[['Gender']])

# Verificar categorías creadas en OneHotEncoder
ohe.categories_
```

**Método get_dummies de Pandas**

De manera similar, get_dummies codifica la función 'Sex' (sexo) categórica al crear dos columnas para representar las dos categorías.

```py
pd.get_dummies(data['Gender']).head()
```

**Diferencia entre OneHotEncoder() y Get_dummies**

- Bajo OneHotEncoder, el DataFrame original sigue siendo del mismo tamaño.

- OneHotEncoder se puede incorporar como parte de una pipeline (tubería) de Machine Learning.

- Bajo OneHotEncoder podemos usar la función GridSearch en Scikit-learn para elegir los mejores parámetros de preprocesamiento.

#### 2. Codificación ordinal

A cada categoría se le asigna un valor numérico que representa un orden. Por ejemplo: [pequeño, mediano, grande, extragrande] podría codificarse como [1,2,3,4].

En esta sección, volveremos a considerar dos enfoques para codificar variables ordinales, uno usando Scikit-learn OrdinalEncoder y el otro usando el método de mapa de Pandas.

**Scikit-learn OrdinalEncoder**

OrdinalEncoder asigna valores incrementales a las categorías de una variable ordinal. Esto ayuda a los algoritmos de Machine Learning a detectar una variable ordinal y, posteriormente, utilizar la información que ha aprendido para hacer predicciones más precisas.

Para usar OrdinalEncoder, primero debemos especificar el orden en el que nos gustaría codificar nuestra variable ordinal.

Ejemplo de código:

```py

from sklearn.preprocessing import OrdinalEncoder

# Instanciar codificador ordinal
oe = OrdinalEncoder()

# Aplicar ordinalEncoder a income_status
oe.fit_transform(data[['income_status']])

```

**Método de Mapa de Pandas**

El método de mapa de Pandas es un enfoque más manual para codificar variables ordinales donde asignamos individualmente valores numéricos a las categorías en una variable ordinal.

Aunque replica el resultado de OrdinalEncoder, no es ideal para codificar variables ordinales con una gran cantidad de categorías únicas.

```py
data['income_status'].map({'low': 0,
                            'medium':1,
                            'high':2})
```

## Construyendo una pipeline (tubería)

Imagina que queremos combinar OneHotEncoder y OrdinalEncoder en un transformador de columna de un solo paso. Después de haber separado nuestros predictores de nuestra variable de destino, podemos construir este transformador de columna de esta manera:

```py
# Hacer un codificador de columna

column_encoder = make_column_encoded(
    (ohe, ['Gender','Blood_type', 'Colour']),
    (oe, ['income_status'])
)

# Aplicar codificador a predictores

column_encoder.fit_transform(predictors)
```

Después de separar nuestros datos en conjuntos de entrenamiento y validación, podemos usar Scikit-learn make_pipeline para construir un Machine Learning Pipeline con los pasos de preprocesamiento y el algoritmo elegido para el modelado.

Ejemplo de código:

```py
from sklearn.pipeline import make_pipeline

# Crear una instancia de pipeline con regresión lineal

lm = LinearRegression()
lm_pipeline = make_pipeline(column_encoder, lm)

# Ajustar pipeline al conjunto de entrenamiento y hacer predicciones en el conjunto de prueba

lm_pipeline.fit(X_train, y_train)
lm_predictions = lm_pipeline.predict(X_test)

print('First 5 predictions:', list(lm_predictions[:5]))
```

Para obtener más información sobre make_pipeline de Scikit-learn, consulta la siguiente documentación: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html

Fuente: 

https://towardsdatascience.com/guide-to-encoding-categorical-features-using-scikit-learn-for-machine-learning-5048997a5c79#:~:text=Feature%20encoding%20is%20the%20process,not%20data%20in%20text%20form.

https://towardsdatascience.com/understanding-feature-engineering-part-2-categorical-data-f54324193e63
