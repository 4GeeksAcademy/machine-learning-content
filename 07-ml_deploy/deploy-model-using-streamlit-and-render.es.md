---
description: >-
  Learn how to deploy your Machine Learning model using Streamlit and Render.
  Discover the simple steps to create an interactive web app today!
---
## Despliegue en Render usando Streamlit

En el módulo anterior, aprendimos a cómo crear una aplicación web de Machine Learning con `Flask`, una librería intuitiva y que facilitaba la generación de plantillas HTML dinámicas para poder utilizar nuestros modelos una vez entrenados.

`Streamlit` es otra gran alternativa muy comúnmente utilizada para generar aplicaciones simplemente programando en Python, sin necesidad de tener conocimientos previos de HTML ni de CSS.

Crearemos una aplicación que integraremos nuevamente en Render.

### Integración en Render

Utilizaremos de nuevo el modelo de clasificación que hemos desarrollado en el [módulo de los árboles de decisión](https://4geeks.com/es/lesson/explorando-arboles-de-decision).

El modelo `decision_tree_classifier_default_42.sav` se ha guardado en un objeto `Pickle` de tal forma que pueda ser utilizado, por ejemplo, para desplegarlo en un servicio web como este caso.

#### Paso 1: Crear un repositorio en Git

Para integrar algo en Render primero debemos haber creado un repositorio en Git. El repositorio que vamos a generar en esta lección se encuentra [aquí](https://github.com/4GeeksAcademy/streamlit-render-integration), que deriva del Machine Learning Template de 4Geeks.

#### Paso 2: Crear una aplicación básica

Ahora generaremos una aplicación sencilla utilizando la librería `Streamlit`. En el directorio `src`, creamos un archivo nuevo llamado `app.py` que modificaremos con el siguiente código:

```py
import streamlit as st

st.title("Hello, World!")
```

En comparación con la sintaxis de Flask que vimos en el módulo anterior es muchísimo más simple e intuitiva. En la interfaz podemos esperar una ventana vacía, con un título que muestra "Hello, World!".

Ahora mismo el repositorio luce de la siguiente forma:

![Streamlit paso 1](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step1.png?raw=true)

#### Paso 3: Ejecutar la aplicación

Para ejecutar la aplicación en local podemos utilizar la misma librería, ya que proporciona un mecanismo para ejecutarla de forma muy sencilla. Debemos acceder con la consola al directorio donde se encuentra el script y ejecutar `streamlit run app.py`.

![Streamlit paso 2](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step2.png?raw=true)

Al terminar se disponibilizará una dirección a través de la cual podemos acceder a la aplicación web:

![Streamlit paso 2.1](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step21.png?raw=true)

En este caso, como estamos desarrollándolo en un Codespace, el enlace es distinto al que se generaría en local, que sería `http://172.16.5.4:8501`.

También podemos apreciar que Streamlit proporciona un estilo base y muy diferente a Flask. Este es el potencial de esta herramienta, que con una implementación simple y minimalista podemos conseguir aplicaciones estéticas, atractivas y usables.

A continuación mejoraremos la interfaz con el fin de poder utilizar el modelo a través de ella.

#### Paso 4: Implementar la interfaz web de la aplicación

Como hemos mencionado al inicio de la lección, queremos integrar el árbol de decisión entrenado para el conjunto de datos de Iris del repositorio UCI de Machine Learning. Este conjunto de datos cuenta con 4 variables predictoras: anchura del pétalo (`petal width (cm)`), longitud del pétalo (`petal length (cm)`), anchura del sépalo (`sepal width (cm)`) y longitud del sépalo (`sepal length (cm)`).

La implementación de interfaces web en Streamlit es infinitamente más simple, no necesitando para ello una carpeta `templates` ni crear plantillas concretas para cada página HTML, se hace todo directamente en el mismo script de Python `app.py`:

```py
from pickle import load
import streamlit as st

model = load(open("../models/decision_tree_classifier_default_42.sav", "rb"))
class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}

st.title("Iris - Model prediction")

val1 = st.slider("Petal width", min_value = 0.0, max_value = 4.0, step = 0.1)
val2 = st.slider("Petal length", min_value = 0.0, max_value = 4.0, step = 0.1)
val3 = st.slider("Sepal width", min_value = 0.0, max_value = 4.0, step = 0.1)
val4 = st.slider("Sepal length", min_value = 0.0, max_value = 4.0, step = 0.1)

if st.button("Predict"):
    prediction = str(model.predict([[val1, val2, val3, val4]])[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)
```

Con el código anterior conseguimos generar un formulario compuesto por 4 elementos deslizantes (para facilitar la tarea al usuario de introducir el valor) y un botón que lanza la predicción al modelo y lo muestra por pantalla.

Otra ventaja de Streamlit es que cada vez que se guarda el script se actualiza la aplicación web, y no es necesario reiniciar la ejecución, como sí sucedía con gunicorn y Flask.

Una vez guardados los cambios y actualizada la interfaz veremos lo siguiente:

![Streamlit paso 3](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step3.png?raw=true)

Tras rellenar los valores y hacer clic sobre `Predict`, el resultado se muestra también en la propia interfaz:

![Streamlit paso 4](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step4.png?raw=true)

Al introducir cualquier valor se predice una clase. Además, la efectividad del modelo es la observada en el módulo pasado.

La interfaz web estaría completa y lista para la integración en Render, ya que no sería necesario actualizar su estilo o diseño como sí lo era en Flask. Sin embargo, las opciones que ofrece esta librería son muchas. [Guía y ejemplos](https://blog.streamlit.io/designing-streamlit-apps-for-the-user-part-ii/).

#### Paso 5: Crear servicio en Render y desplegar la aplicación

El último paso es configurar el servicio en Render y conectarlo con nuestro repositorio Git. Debemos ir al Dashboard de Render, seleccionar el apartado de `Web Services` y elegir el repositorio en el que hayamos subido todo el código y las carpetas anteriores.

Una vez lo seleccionemos nos aparecerá un formulario como el siguiente:

![Streamlit paso 5](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step5.png?raw=true)

Deberemos rellenarlo con la siguiente información:

- `Name`: El nombre que queramos que tenga nuestro servicio. En este caso introduciremos `4geeks-streamlit-integration`
- `Branch`: La rama en la que se encuentra nuestro código actualizado, siempre en la última versión. Deberemos dejar el valor por defecto, `main`.
- `Root Directory`: En este caso hemos desarrollado el código dentro de la carpeta `src`, que incluye el script de Python y las librerías del proyecto (archivo `requirements.txt`), por lo que deberemos introducir `src`.
- `Runtime`: El código es Python, así que dejaremos el valor por defecto, `Python 3`.
- `Build Command`: Dejaremos el valor por defecto, `pip install -r requirements.txt`.
- `Start Command`: Aunque podríamos utilizar gunicorn como sucedía con Flask, Streamlit también tiene una interfaz amena para desplegar soluciones en local, así que modificamos el comando y lo sustituimos por `streamlit run app.py`.

Por último, elegiremos la tarifa gratuita. El formulario, una vez relleno, debería tener la siguiente información:

![Streamlit paso 6](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step6.png?raw=true)

En el siguiente paso, tal y como sucedía en el módulo anterior y siempre que despliegue una solución en Render, aparecerá una consola que nos informará del estado del despliegue:

![Streamlit paso 7](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step7.png?raw=true)

#### Resolver fallo de creación

Debido a que el entorno de Render es diferente a nuestro entorno de desarrollo (especialmente en la versión de Python, ya que se utiliza por defecto la 3.7 y en este caso nosotros usamos de la 3.10 hacia arriba), puede que nos arroje un error la build del proyecto. En este caso su resolución es muy simple:

![Streamlit paso 8](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step8.png?raw=true)

Tenemos que acceder, en la misma pantalla donde se abre el log de la ejecución, al apartado `Environment` e introducir una nueva variable de entorno. En este caso nosotros tenemos la versión `3.11.4` de Python, pero se podría introducir cualquier otra (siempre y cuando sea a partir de la 3.7).

Volvemos a lanzar el despliegue y ahora debería funcionar.

***

Una vez el despliegue haya sido satisfactorio, este será el log que se mostrará:

![Streamlit paso 9](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step9.png?raw=true)

#### Paso 6: Uso del servicio en Render

Una vez que el despliegue ha sido satisfactorio, accedemos a la aplicación desde el enlace situado justo debajo del nombre del servicio, y ya podemos utilizar la aplicación y compartírsela a nuestros amigos/compañeros/clientes. La que hemos creado en esta lección está accesible en el siguiente enlace: `https://fourgeeks-streamlit-integration.onrender.com/`.

![Streamlit paso 10](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/streamlit-step10.png?raw=true)

> Nota: Al haber utilizado el plan gratuito, puede que Render tire la aplicación si no se utiliza. Depende de cuando leas esto la aplicación estará operativa o no.
