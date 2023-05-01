# Implementación de un modelo de Machine Learning usando Flask y Heroku

Los modelos de Machine Learning son herramientas poderosas para hacer predicciones basadas en los datos disponibles. Para que estos modelos sean útiles para la sociedad, deben implementarse para que otros puedan acceder fácilmente a ellos para hacer predicciones. Esto se puede hacer usando Flask y Heroku.

Flask es un marco web de Python pequeño y liviano que proporciona herramientas y características útiles que facilitan la creación de aplicaciones web usando solo un archivo de Python.

Heroku es una plataforma en la nube que le permite crear, entregar, monitorear y escalar aplicaciones. Heroku hace que los procesos de implementación, configuración, escalado, ajuste y administración de aplicaciones sean lo más simples y directos posible para que los desarrolladores puedan concentrarse en crear excelentes aplicaciones. También incluye un rico ecosistema de servicios de datos administrados.

Imaginemos que acabamos de terminar de crear nuestro modelo de predicción de supervivencia del Titanic. ¿Ahora que?

Para predecir con datos desconocidos, tenemos que implementarlos en Internet para que el mundo exterior pueda usarlos.

Para eso, necesitaremos guardar el modelo para que podamos predecir los valores más tarde. Hacemos uso de pickle en python, que es un poderoso algoritmo para serializar y deserializar una estructura de objeto de Python, pero también hay otras herramientas. El siguiente código guarda el modelo usando Pickle:

```py
#serializing our model to a file called model.pkl
import pickle
filename = 'titanic_model.pkl'
pickle.dump(classifier, open(filename,'wb'))
```

## Pasos para crear una aplicación web usando Flask en Python3

Para predecir la supervivencia en el Titanic a partir de varios atributos, primero debemos recopilar los datos (nuevos valores de atributos) y luego usar el modelo que construimos para predecir si un pasajero sobreviviría o no en el Titanic. Por lo tanto, para recopilar los datos, creamos un formulario html que contendría todas las diferentes opciones para seleccionar de cada atributo. Aquí, he creado un formulario simple usando solo html. Si deseas que el formulario sea más interactivo, también puedes hacerlo.

![titanic_prediction_form](https://github.com/Lorenagubaira/machine-learning-content/blob/master/assets/titanic_prediction_form.jpg?raw=true)

### **Paso 1:** Activa el entorno e instala Flask

En la línea de comando ingresa el directorio de tu proyecto. Una vez allí, activa su entorno y usa pip para instalar Flask.

```bash
pip install Flask
```

### **Paso 2:** Crea una aplicación básica

En tu directorio, abre un archivo llamado hello.py para editarlo. Este archivo hello.py servirá como un ejemplo mínimo de cómo manejar las solicitudes HTTP. En el interior, importará el objeto Flask y creará una función que devuelva una respuesta HTTP. Escribe el siguiente código dentro de hello.py:

```py
from flask import Flask

app = Flask(__name__)


@app.route('/')
def hello():
    return 'Hello, World!'
```

Expliquemos lo que acaba de hacer el código anterior. Primero importa el objeto Flask del paquete Flask. Luego lo usará para crear su instancia de la aplicación Flask con el nombre app. Pasa la variable especial __name__ que contiene el nombre del módulo de Python actual. Se utiliza para decirle a la instancia dónde se encuentra. Deberá hacer esto porque Flask configura algunas rutas en segundo plano.

Una vez que crea la instancia de la aplicación, la usa para manejar las solicitudes web entrantes y enviar respuestas al usuario. @app.route es un decorador que convierte una función normal de Python en una función de vista Flask, que convierte el valor de retorno de la función en una respuesta HTTP que mostrará un cliente HTTP, como un navegador web. Pasa el valor '/' a @app.route() para indicar que esta función responderá a las solicitudes web de la URL /, que es la URL principal.

La función de vista hello() devuelve la cadena 'Hello, World!' en respuesta.

Guarda y cierra el archivo.

Para ejecutar su aplicación web, primero le indicarás a Flask dónde encontrar la aplicación (el archivo hello.py en su caso) con la variable de entorno `FLASK_APP`:

```bash
export FLASK_APP=hello
```

Luego, ejecútalo en modo desarrollo con la variable de entorno `FLASK_ENV`:

```bash
export FLASK_ENV=development
```

Finalmente, ejecuta la aplicación usando `flask run`:

```py
flask run
```

Una vez que se está ejecutando, el resultado debería ser similar a este:

```bash
Output
 * Serving Flask app "hello" (lazy loading)
 * Environment: development
 * Debug mode: on
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 813-894-335
 ```

El resultado anterior tiene varias informaciones, tales como:

- El nombre de la aplicación que estás ejecutando.

- El entorno en el que se ejecuta la aplicación.

- Debug mode: on significa que el depurador Flask se está ejecutando. Esto es útil durante el desarrollo porque nos brinda mensajes de error detallados cuando algo sale mal, lo que facilita la resolución de problemas.

- La aplicación se ejecuta localmente en la URL http://127.0.0.1:5000/, 127.0.0.1 es la IP que representa el host local de su computadora y :5000 es el número de puerto.

Ahora abre un navegador y escribe la URL http://127.0.0.1:5000; recibirás la cadena Hello, World! en respuesta. Esto confirma que tu aplicación se está ejecutando correctamente.

Ahora tienes una pequeña aplicación web Flask. Ha ejecutado su aplicación y ha mostrado información en el navegador web. A continuación, utilizará los archivos HTML en su aplicación.

### **Paso 3:** Usar plantillas HTML

Actualmente, su aplicación solo muestra un mensaje simple sin HTML. Las aplicaciones web utilizan principalmente HTML para mostrar información al visitante, por lo que ahora trabajará para incorporar un archivo HTML en tu aplicación, que se puede mostrar en el navegador web.

Flask proporciona una función auxiliar render_template() que permite el uso del motor de plantillas Jinja. Esto hará que la administración de HTML sea mucho más fácil al escribir su código HTML en archivos .html, además de usar la lógica en su código HTML. Utilizarás estos archivos HTML (plantillas) para crear su aplicación web.

En este paso, crearás tu aplicación Flask principal en un archivo nuevo.

Primero, en el directorio de tu proyecto, usa tu editor de texto favorito para crear y editar tu archivo app.py. Anteriormente, has estado usando app.py para escribir el código de tu modelo final. Para evitar confusiones, ahora usarás un 'model.py' o un 'titanic.py' para eso, y el app.py será exclusivamente para construir tu aplicación web. Esto albergará todo el código que utilizarás para crear la aplicación.

En este nuevo archivo, importarás el objeto Flask para crear una instancia de la aplicación Flask, como lo hiciste antes. También importará la función auxiliar render_template() que le permite renderizar archivos de plantilla HTML que existen en la carpeta de plantillas que estás a punto de crear. El archivo tendrá una función de vista única que se encargará de manejar las solicitudes a la ruta principal/. Agrega el siguiente contenido:

```py
import numpy as np
import flask
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
```

La función de vista index() devuelve el resultado de invocar render_template() con index.html como argumento; esto le indica a render_template() que busque un archivo llamado index.html en la carpeta de plantillas. La carpeta y el archivo aún no existen y recibirá un error si ejecutas la aplicación en este momento. Lo ejecutarás de todos modos, para que estés familiarizado con esta excepción que se encuentra comúnmente. Luego lo resolverá creando la carpeta y el archivo necesarios.

Guarda el archivo y ciérralo.

Deten el servidor de desarrollo en tu otro terminal ejecutando la aplicación hello con CTRL+C.

Antes de ejecutar la aplicación, asegúrate de especificar correctamente el valor de la variable de entorno FLASK_APP, ya que ahora no estás utilizando la aplicación hello.

```bash
export FLASK_APP=app
flask run
```
Cuando abras la URL http://127.0.0.1:5000 en tu navegador, se mostrará la página del depurador informándote que no se encontró la plantilla index.html. Se resaltará la línea principal de código en el código responsable de este error. En este caso, es la línea return render_template('index.html').

Si haces clic en esta línea, el depurador revelará más código para que tengas más contexto que te ayude a resolver el problema.

Probablemente verás un error que muestra 'plantilla no encontrada (index.html)'.

Vamos a crear plantillas de carpetas. En tu aplicación, utilizarás plantillas para representar HTML que se mostrará en el navegador del usuario. Esta carpeta contiene nuestro archivo de formulario html index.html. Comienza a editar tu archivo index.html escribiendo el siguiente código:

```py
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>ML app</title>
</head>
<body>
   <h1>Welcome to my Titanic Survival prediction app</h1>
</body>
</html>
```

Guarda el archivo y usa tu navegador para navegar a http://127.0.0.1:500 nuevamente, o actualiza la página. Esta vez, el navegador debería mostrar el texto "Bienvenido a mi aplicación de predicción Titanic Survival" en una etiqueta h1.

Además de la carpeta de plantillas, las aplicaciones web de Flask también suelen tener una carpeta estática para albergar archivos, como los archivos CSS, los archivos JavaScript y las imágenes que utiliza la aplicación.

Puedes crear un archivo de hoja de estilo style.css para agregar CSS a tu aplicación. Primero, crea un directorio llamado estático dentro de tu directorio principal del proyecto. Luego crea otro directorio llamado css dentro de static para alojar los archivos .css. Se puede hacer lo mismo con archivos js e imágenes para aplicaciones más complejas.

Dentro de su directorio css, crea un archivo style.css y agrega la siguiente regla:

```py
h1 {
    border: 2px #eee solid;
    color: brown;
    text-align: center;
    padding: 10px;
}
```
Este código agregará un borde, cambiará el color a marrón, centrará el texto y agregará un pequeño relleno a los tags h1.

Guarda y cierra el archivo.

En tu archivo index.html agregarás un enlace a tu archivo style.css:

```py
<head>
    <meta charset="UTF-8">
    <link rel="stylesheet" href="{{ url_for('static', filename= 'css/style.css') }}">
    <title>Welcome to my Titanic Survival prediction app</title>
</head>
```
Aquí utilizas la función auxiliar url_for() para generar la ubicación de archivo adecuada. El primer argumento especifica que está vinculando a un archivo estático y el segundo argumento es la ruta al archivo dentro del directorio estático.

Guarda y cierra el archivo.

Después de actualizar la página de índice de tu aplicación, notarás que el texto "Bienvenido a mi aplicación de predicción Titanic Survival" ahora es marrón, está centrado y enmarcado dentro de un borde.

Puedes poner el estilo que desees en tu archivo style.css. Sin embargo, el kit de herramientas de Bootstrap puede ayudarte con esto si no eres un experto. Ahora, si tu aplicación tendrá más de una página, puedes evitar la repetición innecesaria de código con la ayuda de un archivo de plantilla base, del cual heredarán todos tus archivos HTML. Si ese es el caso, puedes escribir el siguiente código en tu archivo base.html:

```py
<!doctype html>
<html lang="en">
  <head>
    <!-- Required meta tags -->
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">

    <!-- Bootstrap CSS -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">

    <title>{% block title %} {% endblock %}</title>
  </head>
  <body>
    <nav class="navbar navbar-expand-md navbar-light bg-light">
        <a class="navbar-brand" href="{{ url_for('index')}}">FlaskBlog</a>
        <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav">
            <li class="nav-item active">
                <a class="nav-link" href="#">About</a>
            </li>
            </ul>
        </div>
    </nav>
    <div class="container">
        {% block content %} {% endblock %}
    </div>

    <!-- Optional JavaScript -->
    <!-- jQuery first, then Popper.js, then Bootstrap JS -->
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js" integrity="sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js" integrity="sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM" crossorigin="anonymous"></script>
  </body>
</html>
```

Guarda y cierra el archivo una vez que hayas terminado de editarlo.

La mayor parte del código en el bloque anterior es HTML estándar y requiere código para Bootstrap. Los tags <meta> brindan información para el navegador web, el tag <link> vincula a los archivos CSS de Boostrap y los tags <script> son vínculos al código JavaScript que habilita alguna funcionalidad adicional de Boostrap.

Sin embargo, las siguientes partes resaltadas son específicas del motor de plantillas Jinja:

- {% block title %} {% endblock %}: un bloque que sirve como marcador de posición para un título. Luego lo usarás en otras plantillas para dar un título personalizado a cada página de tu aplicación sin tener que volver a escribir la sección <head> completa cada vez.

- {{ url_for('index')}: una invocación de función que devolverá la URL para la función de vista index(). Esto es diferente de la invocación anterior de url_for() que usó para vincular a un archivo CSS estático, porque solo requiere un argumento, que es el nombre de la función de vista, y vincula a la ruta asociada con la función en lugar de a un archivo estático expediente.

- {% block content %} {% endblock %}: otro bloque que será reemplazado por contenido dependiendo de la plantilla secundaria (plantillas que heredan de base.html) que lo anulará.

Ahora que tienes una plantilla base.html, puedes heredar ese código a index.html agregando solo el siguiente código en tu index.html:

```py
{% extends 'base.html' %}

{% block content %}
    <h1>{% block title %} Welcome to FlaskBlog {% endblock %}</h1>
{% endblock %}
```

Has utilizado plantillas HTML y archivos estáticos en Flask de forma limpia. Sin embargo, para simplificar las cosas para su primera aplicación web, conservaremos solo el archivo index.html.

Veamos cómo debemos codificar un formulario solicitando los atributos de nuestros pasajeros.

> Para poder predecir los datos correctamente, los valores correspondientes de cada etiqueta deben coincidir con el valor de cada entrada seleccionada.

En el formulario Titanic que viste al comienzo de esta lección, solo solicitamos las características numéricas para la predicción, pero en el caso de que incluyamos características categóricas que fueron previamente codificadas con etiquetas, necesitamos poner los mismos valores en el formulario html. El siguiente ejemplo muestra cómo se debe codificar el formulario en caso de que a nuestra función 'Sex' se le haya asignado 0 para Hombre y 1 para Mujer:

```py
<label for="Sex">Gender</label>
    <select id="relation" name="relation">
      <option value="0">Male</option>
      <option value="1">Female</option>
    </select>
```

Puedes encontrar un par de ejemplos de formularios en los siguientes enlaces:

https://github.com/4GeeksAcademy/machine-learning-content/blob/master/07-1d-ml_deploy/form-examples/index_example1.html

https://github.com/4GeeksAcademy/machine-learning-content/blob/master/07-1d-ml_deploy/form-examples/index_example2.html

https://www.geeksforgeeks.org/html-design-form/

### **Paso 4:** Predecir el resultado de supervivencia

Ejecutemos la aplicación.

```bash
export FLASK_APP=app.py
run flask
```

Cuando alguien envía el formulario, la página web debe mostrar el resultado si un pasajero sobreviviría o moriría en el Titanic. Para esto, necesitamos el archivo modelo (model.pkl) que creamos antes, en la misma carpeta del proyecto. Agregamos el siguiente código al archivo app.py:

```py
# función de predicción
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,12)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/result',methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        
        if int(result)==1:
            prediction='Passenger survives'
        else:
            prediction='Passenger dies'
            
        return render_template("result.html",prediction=prediction) 
```

Aquí, después de enviar el formulario, los valores del formulario se almacenan en la variable to_predict_list en forma de diccionario. Lo convertimos en una lista de los valores del diccionario y lo pasamos como argumento a la función ValuePredictor(). En esta función, cargamos el archivo model.pkl y predecimos los nuevos valores y devolvemos el resultado.

Este resultado/predicción (Pasajero sobrevive o no) se pasa como argumento al motor de plantilla con la página html que se mostrará.

Crea el siguiente archivo result.html y agrégalo a la carpeta de plantillas.

```py
<!doctype html>
<html>
   <body>
       <h1> {{ prediction }}</h1>
   </body>
</html>
```

**Un código alternativo para todo el archivo app.py podría ser:**

```py
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('titanic_model.pkl', 'rb'))

@app.route('/') #http://www.google.com/
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Would you survive? {} (1=survived, 0=deceased)'.format(output))

if __name__=="__main__":
    app.run(port=5000, debug=True)
```

Vuelve a ejecutar la aplicación y debería predecir el resultado después de enviar el formulario. Hemos creado con éxito la aplicación Web. Ahora es el momento de usar Heroku para implementarlo.

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

![flask-heroku-structure](https://github.com/Lorenagubaira/machine-learning-content/blob/master/assets/flask-heroku-structure.jpg?raw=true)

4. Finalmente, usa un archivo .gitignore para excluir archivos innecesarios que no queremos implementar en Heroku.

¡Estamos listos! ¡Envía tu proyecto a Heroku! Si deseas hacerlo directamente en el sitio web de Heroku, puedes hacerlo de la siguiente manera:

- Haz clic en 'Crear una nueva aplicación'

- En la pestaña 'implementar': vincula la aplicación Heroku a tu cuenta de Github y selecciona el repositorio para conectarse.

- Desplázate hacia abajo y elije 'despliegue manual'. Después de asegurarte de que estás en la rama que deseas implementar (en este caso: principal), haz clic en 'Implementar rama'. Verás que se han instalado todos los paquetes necesarios como en la siguiente imagen:

![deploying_branch](https://github.com/Lorenagubaira/machine-learning-content/blob/master/assets/deploying_branch.jpg?raw=true)

- Cuando termines, debería verse como la siguiente captura de pantalla:

![deployed_to_heroku](https://github.com/Lorenagubaira/machine-learning-content/blob/master/assets/deployed_to_heroku.jpg?raw=true)

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
