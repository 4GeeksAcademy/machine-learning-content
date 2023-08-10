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

En esta lección integraremos el modelo de clasificación que hemos desarrollado en el [módulo de los árboles de decisión](https://4geeks.com/es/syllabus/spain-ds-pt-1/read/exploring-decision-trees).

El modelo `decision_tree_classifier_default_42.sav` se ha guardado en un objeto `Pickle` de tal forma que pueda ser utilizado, por ejemplo, para desplegarlo en un servicio web como este caso.

#### Paso 1: Crear un repositorio en Git

Para integrar algo en Render primero debemos haber creado un repositorio en Git. El Git que vamos a generar en esta lección se encuentra [aquí](https://github.com/4GeeksAcademy/flask-render-integration), que deriva del Machine Learning Template de 4Geeks.

#### Paso 2: Crear una aplicación básica

Ahora generaremos una aplicación sencilla utilizando la librería `Flask`. En el directorio `src`, creamos un archivo nuevo llamado `hello.py` que modificaremos con el siguiente código:

```py
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"
```

El archivo creado servirá como un ejemplo mínimo de cómo manejar las solicitudes HTTP. En él se importa el objeto `Flask` y se crea una función que devuelve una respuesta HTTP.

Ahora mismo el repositorio luce de la siguiente forma:

![flask-step1](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step1.png?raw=true)

#### Paso 3: Ejecutar la aplicación

Para ejecutar la aplicación en local necesitamos la librería de Python `gunicorn`. Simplemente debemos instalarla, acceder con la consola al directorio donde se encuentra el script y ejecutar `gunicorn app:app`.

![flask-step2](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step2.png?raw=true)

Al terminar se disponibilizará una dirección a través de la cual podemos acceder a la aplicación web:

![flask-step21](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step21.png?raw=true)

En este caso como estamos desarrollándolo en un Codespace, el enlace es distinto al que se generaría en local, que sería `http://127.0.0.1:8000`.

Ahora hemos implementado una aplicación web muy sencilla usando Flask. Además, hemos podido ejecutarla y mostrar información en la interfaz web.

Ahora tienes una pequeña aplicación web Flask. Ha ejecutado su aplicación y ha mostrado información en el navegador web. A continuación, añadiremos archivos HTML para personalizar la aplicación.

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
model = load(open("/workspaces/flask-render-integration/models/decision_tree_classifier_default_42.sav","rb"))
class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        
        val1 = float(request.form['val1'])
        val2 = float(request.form['val2'])
        val3 = float(request.form['val3'])
        val4 = float(request.form['val4'])
        
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

#### Paso 6: Crear servicio en Render

El último paso es configurar el servicio en Render y conectarlo con nuestro repositorio Git. Debemos ir al Dashboard de Render, seleccionar el apartado de `Web Services` y elegir el repositorio en el que hayamos subido todo el código y las carpetas anteriores.

Una vez lo seleccionemos nos aparecerá un formulario como el siguiente:

![flask-step8](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-step8.png?raw=true)

Deberemos rellenarlo con la siguiente información:

- `Name`: El nombre que queramos que tenga nuestro servicio. En este caso introduciremos `4geeks-flask-integration`
- `Branch`: La rama en la que se encuentra nuestro código actualizado, siempre en la última versión. Deberemos dejar el valor por defecto, `master`.
- ``
- ``
- ``
- ``
- ``




## Implementación usando Heroku

Ya deberías tener una cuenta en Heroku, pero si no la tienes, continúa y crea tu cuenta en 'https://www.heroku.com'.

Asegurémonos de tener también lo siguiente antes de implementar en Heroku:

1. Gunicorn maneja las solicitudes y se encarga de las cosas complicadas. Descarga gunicorn a tu entorno virtual. Puedes usar pip para descargarlo.

```bash
pip install gunicorn
```

2. Hemos instalado muchas bibliotecas y otros archivos importantes como flask, gunicorn, sklearn, etc. Necesitamos decirle a Heroku que nuestro proyecto requiere todas estas bibliotecas para ejecutar la aplicación con éxito. Esto se hace creando un archivo requirements.txt.

3. Procfile es un archivo de texto en el directorio raíz de tu aplicación, para declarar explícitamente qué comando debe ejecutarse para iniciar tu aplicación. Este es un requisito esencial para Heroku. Este archivo le dice a Heroku que queremos usar el proceso web con el comando gunicorn y el nombre de la aplicación.

```py
web: gunicorn app:app
```

Tu estructura actual debería ser algo como esto:

![flask-heroku-structure](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/flask-heroku-structure.jpg?raw=true)

4. Finalmente, usa un archivo .gitignore para excluir archivos innecesarios que no queremos implementar en Heroku.

¡Estamos listos! ¡Envía tu proyecto a Heroku! Si deseas hacerlo directamente en el sitio web de Heroku, puedes hacerlo de la siguiente manera:

- Haz clic en 'Crear una nueva aplicación'

- En la pestaña 'implementar': vincula la aplicación Heroku a tu cuenta de Github y selecciona el repositorio para conectarse.

- Desplázate hacia abajo y elije 'despliegue manual'. Después de asegurarte de que estás en la rama que deseas implementar (en este caso: principal), haz clic en 'Implementar rama'. Verás que se han instalado todos los paquetes necesarios como en la siguiente imagen:

![deploying_branch](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/deploying_branch.jpg?raw=true)

- Cuando termines, debería verse como la siguiente captura de pantalla:

![deployed_to_heroku](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/deployed_to_heroku.jpg?raw=true)

- Copia ese enlace y pégalo en tu navegador para probar tu aplicación.

**Si te sientes más cómodo con la línea de comandos, deberás tener instalados git y Heroku CLI, y luego seguir estos pasos:**

> Puedes hacer clic en el siguiente enlace para instalar Heroku CLI: https://devcenter.heroku.com/articles/heroku-cli

```bash
heroku login
```

```bash
heroku create
```

```bash
git init
git add .
git commit -m 'initial commit'
```

```bash
git push heroku master
heroku open
```

¡Anímate y prueba tu aplicación web!

Fuente:

https://www.heroku.com/

https://devcenter.heroku.com/articles/heroku-cli

https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3-es

https://medium.com/towards-data-science/designing-a-machine-learning-model-and-deploying-it-using-flask-on-heroku-9558ce6bde7b

https://medium.com/towards-data-science/create-an-api-to-deploy-machine-learning-models-using-flask-and-heroku-67a011800c50

https://medium.com/towards-data-science/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

https://medium.com/towards-data-science/flask-and-heroku-for-online-machine-learning-deployment-425beb54a274
