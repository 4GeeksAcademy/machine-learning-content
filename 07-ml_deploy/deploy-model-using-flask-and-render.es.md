## Despliegue en Render usando Flask

Tras la fase de desarrollo del modelo, tendremos un modelo resolutivo según nuestras espectativas y que satisface nuestras necesidades. Para que este modelo sea útil y cumpla la función para la que ha sido entrenado, debemos disponibilizarlo en algún entorno que nos permita su utilización. Aquí proponemos un entorno gratuito llamado `Render`, pero puede trasladarse a otros entornos, gratuitos o de pago.

### Render

Render es una plataforma de computación en la nube que facilita el despliegue, la hosting y la ejecución de aplicaciones, bases de datos, tareas programadas y otros servicios. A menudo se describe como una plataforma fácil de usar que combina la facilidad de las plataformas como Heroku con el poder y la flexibilidad de los proveedores de nube más tradicionales como AWS.

Algunas características y ofertas clave de Render incluyen:

1. **Despliegue de aplicaciones web**: Render permite desplegar aplicaciones web en varios lenguajes y marcos, incluidos Node.js, Ruby on Rails, Django y muchos otros.
2. **Servicios privados**: Son aplicaciones o trabajos que no están expuestos a internet pero pueden ser usados por otras aplicaciones en Render.
3. **Tareas programadas**: Permite ejecutar trabajos periódicos, similares a los cron jobs en sistemas Unix.
4. **Bases de datos**: Render soporta el despliegue de bases de datos como PostgreSQL, y ofrece una solución de almacenamiento persistente para datos.
5. **Despliegue desde repositorios**: Puedes conectar tu repositorio de GitHub o GitLab y configurar despliegues automáticos cada vez que hagas push a tu repositorio.

Render se ha ganado una reputación positiva por ser una opción atractiva para desarrolladores y startups que buscan una forma rápida y sencilla de desplegar y escalar aplicaciones sin la sobrecarga administrativa de las soluciones más tradicionales.

#### Registro en la plataforma

Para poder acceder a Render debemos tener una cuenta. Para registrarse se debe acceder al siguiente [enlace](https://dashboard.render.com/register). Una vez tenemos una cuenta, se nos habilita el acceso a toda la funcionalidad de Render:

![render-functionalities](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/render-functionalities.PNG?raw=true)

Podemos crear servicios de bases de datos, de despliegue web, tareas programadas...

### Integración en Render

En esta lección integraremos el modelo de clasificación que hemos desarrollado en el [módulo de los árboles de decisión](https://4geeks.com/es/lesson/explorando-arboles-de-decision).

El modelo `decision_tree_classifier_default_42.sav` se ha guardado en un objeto `Pickle` de tal forma que pueda ser utilizado, por ejemplo, para desplegarlo en un servicio web como este caso.

#### Paso 1: Crear un repositorio en Git

Para integrar algo en Render primero debemos haber creado un repositorio en Git. El Git que vamos a generar en esta lección se encuentra [aquí](https://github.com/4GeeksAcademy/flask-render-integration), que deriva del Machine Learning Template de 4Geeks.

#### Paso 2: Crear una aplicación básica

Ahora generaremos una aplicación sencilla utilizando la librería `Flask`. En el directorio `src`, creamos un archivo nuevo llamado `app.py` que modificaremos con el siguiente código:

```py
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"
```

El archivo creado servirá como un ejemplo mínimo de cómo manejar las solicitudes HTTP. En él se importa el objeto `Flask` y se crea una función que devuelve una respuesta HTTP.

Ahora mismo el repositorio luce de la siguiente forma:

![Flask paso 1](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step1.png?raw=true)

#### Paso 3: Ejecutar la aplicación

Para ejecutar la aplicación en local necesitamos la librería de Python `gunicorn`. Simplemente debemos instalarla, acceder con la consola al directorio donde se encuentra el script y ejecutar `gunicorn app:app`.

![Flask paso 2](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step2.png?raw=true)

Al terminar se disponibilizará una dirección a través de la cual podemos acceder a la aplicación web:

![Flask paso 2.1](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step21.png?raw=true)

En este caso como estamos desarrollándolo en un Codespace, el enlace es distinto al que se generaría en local, que sería `http://127.0.0.1:8000`.

En este punto tenemos una pequeña aplicación web Flask con poca o casi ninguna funcionalidad. A continuación, añadiremos archivos HTML para personalizar la aplicación.

#### Paso 4: Implementar la interfaz web de la aplicación

Como hemos mencionado al inicio de la lección, queremos integrar el árbol de decisión entrenado para el conjunto de datos de Iris del repositorio UCI de Machine Learning. Este conjunto de datos cuenta con 4 variables predictoras: anchura del pétalo (`petal width (cm)`), longitud del pétalo (`petal length (cm)`), anchura del sépalo (`sepal width (cm)`) y longitud del sépalo (`sepal length (cm)`).

Crearemos un HTML que permita introducir un valor para cada variable para poder llevar a cabo la predicción:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Iris - Model prediction</title>
</head>
<body>
    <h2>Introduce the values</h2>
    
    <form action="/" method="post">
        Petal width: <input type="number" step="any" name="val1" required><br><br>
        Petal length: <input type="number" step="any" name="val2" required><br><br>
        Sepal width: <input type="number" step="any" name="val3" required><br><br>
        Sepal length: <input type="number" step="any" name="val4" required><br><br>
        <input type="submit" value="Predict">
    </form>
    
    {% if prediction != None %}
        <h3>Prediction: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
```

Este HTML contiene un título y un formulario en el que se deben introducir los valores asociados a cada campo. A continuación, pulsando sobre el botón `Predict` aparecerá un elemento que contiene la predicción del modelo, en función de los valores introducidos. En el HTML hay unas sentencias entre llaves que es código Python puro, una curiosa sintaxis que utiliza Flask para introducir valores de manera dinámica.

Todas las plantillas HTML que generemos deben ir en una carpeta `templates` que se debe crear al mismo nivel que el `app.py`. Llamamos a este fichero `index.html` y lo almacenamos en la carpeta.

Además de crear la plantilla anterior, debemos actualizar el código para que se alimente del HTML, reciba los campos y pueda devolver una predicción. Así, el archivo `app.py` lo actualizaríamos:

```py
from flask import Flask, request, render_template
from pickle import load

app = Flask(__name__)
model = load(open("/workspaces/flask-render-integration/models/decision_tree_classifier_default_42.sav", "rb"))
class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        
        val1 = float(request.form["val1"])
        val2 = float(request.form["val2"])
        val3 = float(request.form["val3"])
        val4 = float(request.form["val4"])
        
        data = [[val1, val2, val3, val4]]
        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]
    else:
        pred_class = None
    
    return render_template("index.html", prediction = pred_class)
```

Hemos creado la función `index`, que reemplaza a la antigua `hello_world` y que se nutre de los valores que se introduzcan en el HTML para desencadenar el proceso de predicción. Esto es así porque cuando se hace click sobre el botón `Predecir`, se envía una petición POST al script y se leen los valores introducidos en el formulario del HTML para realizar la predicción.

En última instancia, el método devuelve el HTML renderizado, en este caso con el valor de la predicción en función de los valores.

Ahora mismo el repositorio luce de la siguiente forma:

![flask-step3](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step3.png?raw=true)

Si guardamos los cambios y ejecutamos de nuevo la aplicación (`gunicorn app:app`), tras navegar a nuestra aplicación web en local veremos lo siguiente:

![flask-step4](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step4.png?raw=true)

Tras rellenar los valores y hacer click sobre `Predict`, el resultado se muestra también en la propia interfaz:

![flask-step5](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step5.png?raw=true)

Al introducir cualquier valor se predice una clase. Además, la efectividad del modelo es la observada en el módulo pasado.

La interfaz web parece muy simple y poco atractiva de cara a los usuarios. El siguiente paso es darle algo de estilo.

#### Paso 5: Estilizar la interfaz web de la aplicación

Una manera fácil de añadir estilos es utilizando CSS. Podemos agregar un bloque `<style>` directamente al HTML anterior para mejorarlo visualmente. El código `CSS` que incluiremos será el siguiente:

```css
body {
    font-family: Arial, sans-serif;
    margin: 40px;
    background-color: #f4f4f4;
}
form {
    background-color: #fff;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
}
input[type="number"] {
    width: 100%;
    padding: 10px;
    margin: 10px 0;
    border-radius: 4px;
    border: 1px solid #ccc;
}
input[type="submit"] {
    background-color: #333;
    color: #fff;
    padding: 10px 15px;
    border: none;
    border-radius: 4px;
    cursor: pointer;
}
input[type="submit"]:hover {
    background-color: #555;
}
h3 {
    margin-top: 20px;
    background-color: #fff;
    padding: 10px;
    border-radius: 4px;
}
```

El código anterior establece un fondo claro para toda la página, y destaca el formulario y el encabezado con un fondo blanco y bordes suavemente redondeados. Los campos de entrada son más espaciosos y visuales, con bordes y rellenos adecuados, y el botón de envío presenta un cambio de color cuando se pasa el cursor sobre él, proporcionando retroalimentación visual. Además, se emplea una tipografía más legible y se separan adecuadamente los elementos con márgenes para evitar que se sientan apretados.

Al introducirlo en el HTML, el código quedaría tal que así:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Iris - Model prediction</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f4f4f4;
        }
        form {
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
        }
        input[type="number"] {
            width: 100%;
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
            border: 1px solid #ccc;
        }
        input[type="submit"] {
            background-color: #333;
            color: #fff;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        input[type="submit"]:hover {
            background-color: #555;
        }
        h3 {
            margin-top: 20px;
            background-color: #fff;
            padding: 10px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <h2>Introduce the values</h2>
    
    <form action="/" method="post">
        Petal width: <input type="number" step="any" name="val1" required><br><br>
        Petal length: <input type="number" step="any" name="val2" required><br><br>
        Sepal width: <input type="number" step="any" name="val3" required><br><br>
        Sepal length: <input type="number" step="any" name="val4" required><br><br>
        <input type="submit" value="Predict">
    </form>
    
    {% if prediction != None %}
        <h3>Prediction: {{ prediction }}</h3>
    {% endif %}
</body>
</html>
```

Tras reejecutar la aplicación y acceder de nuevo a la interfaz web, este es su nuevo aspecto:

![flask-step6](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step6.png?raw=true)

Y nuevamente, al rellenar los valores y lanzar la predicción, así se muestra en el front:

![flask-step7](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step7.png?raw=true)

Tras desarrollar la funcionalidad deseada y contar con un front que satisface nuestras necesidades, integraremos todo esto en Render.

#### Paso 6: Crear servicio en Render y desplegar la aplicación

El último paso es configurar el servicio en Render y conectarlo con nuestro repositorio Git. Debemos ir al Dashboard de Render, seleccionar el apartado de `Web Services` y elegir el repositorio en el que hayamos subido todo el código y las carpetas anteriores.

Una vez lo seleccionemos nos aparecerá un formulario como el siguiente:

![flask-step8](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step8.png?raw=true)

Deberemos rellenarlo con la siguiente información:

- `Name`: El nombre que queramos que tenga nuestro servicio. En este caso introduciremos `4geeks-flask-integration`
- `Branch`: La rama en la que se encuentra nuestro código actualizado, siempre en la última versión. Deberemos dejar el valor por defecto, `master`.
- `Root Directory`: En este caso hemos desarrollado el código dentro de la carpeta `src`, que incluye el script de Python, el template HTML y las librerías del proyecto (archivo `requirements.txt`), por lo que deberemos introducir `src`.
- `Runtime`: El código es Python, así que dejaremos el valor por defecto, `Python 3`.
- `Build Command`: Dejaremos el valor por defecto, `pip install -r requirements.txt`.
- `Start Command`: Ya somos amigables con este comando. Hemos utilizado en el desarrollo gunicorn, así que dejaremos el valor por defecto, `gunicorn app:app`.

Por último, elegiremos la tarifa gratuita. El formulario, una vez relleno, debería tener la siguiente información:

![flask-step9](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step9.png?raw=true)

En el siguiente paso nos aparecerá una consola con los logs del despliegue de la aplicación. El despliegue se hace paso a paso, clonando en primer lugar el repositorio, construyéndolo (*build*), instalando las dependencias, y, en último lugar, ejecutando el comando para lanzar la aplicación web.

![flask-step10](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step10.png?raw=true)

##### Resolver fallo de creación

Debido a que el entorno de Render es diferente al nuestro de desarrollo (especialmente en la versión de Python, ya que se utiliza por defecto la 3.7 y en este caso nosotros usamos de la 3.10 hacia arriba), puede que nos arroje un error la build del proyecto. En este caso su resolución es muy simple:

![flask-step11](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step11.png?raw=true)

Tenemos que acceder, en la misma pantalla donde se abre el log de la ejecución, al apartado `Environment` e introducir una nueva variable de entorno. En este caso nosotros tenemos la versión `3.11.4` de Python pero se podría introducir cualquier otra (siempre y cuando sea a partir de la 3.7).

Volvemos a lanzar el despliegue y ahora debería funcionar.

***

Una vez el despliegue haya sido satisfactorio, este será el log que se mostrará:

![flask-step12](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step12.png?raw=true)

De hecho, hay disponible un apartado en el que podemos visualizar los distintos despliegues de nuestra aplicación web y el status de cada uno de ellos:

![flask-step13](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step13.png?raw=true)

#### Paso 7: Uso del servicio en Render

Una vez que el despliegue ha sido satisfactorio, accedemos a la aplicación desde el enlace situado justo debajo del nombre del servicio, y ya podemos utilizar la aplicación y compartirsela a nuestros amigos/compañeros/clientes. La que hemos creado en esta lección está accesible en el siguiente enlace: `https://fourgeeks-flask-integration.onrender.com/`.

![flask-step14](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step14.png?raw=true)

> NOTA: Al haber utilizado el plan gratuito, puede que Render tire la aplicación si no se utiliza. Depende de cuando leas esto la aplicación estará operativa o no.
