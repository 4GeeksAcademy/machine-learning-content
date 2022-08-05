# Despliegue de un modelo de Machine Learning usando Flask y Heroku

Flask es un marco web de Python pequeño y liviano que proporciona herramientas y características útiles que facilitan la creación de aplicaciones web usando solo un archivo de Python.

Heroku es una plataforma en la nube que le permite crear, entregar, monitorear y escalar aplicaciones. Heroku hace que los procesos de implementación, configuración, escalado, ajuste y administración de aplicaciones sean lo más simples y directos posible para que los desarrolladores puedan concentrarse en crear excelentes aplicaciones. También incluye un rico ecosistema de servicios de datos administrados.

## Pasos para crear una aplicación web usando Flask en Python3

### **Paso 1:** Activa el entorno e instala Flask

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

### **Paso 3:** Usar plantillas HTML

El kit de herramientas Bootstrap te ayudará a darle un poco de estilo a tu aplicación sin tener que escribir tu propio código HTML, CSS y JavaScrip. El conjunto de herramientas te permitirá concentrarse en aprender cómo funciona Flask, en lugar de aprender HMTL.

### **Paso de Heroku: crea una nueva aplicación web en Heroku**

Ya deberías tener una cuenta en Heroku. En caso de que no la tengas, anímate y crea tu cuenta en'https://www.heroku.com'.

- Haz clic en 'Crear una nueva aplicación'

- En la pestaña 'implementar': vincula la aplicación Heroku a su cuenta de Github y selecciona el repositorio al que conectarse.

- Desplázate hacia abajo y elije 'manual deploy'. Después de asegurarse de que estás en la rama que deseas implementar (en este caso: principal), haz clic en 'Deploy branch'. Verás que se han instalado todos los paquetes necesarios como en la siguiente imagen:

*incluir imagen*

- Cuando termine, debería verse como la siguiente captura de pantalla:

*incluir imagen*

- Copia ese enlace y pégalo en tu navegador para probar su aplicación.

## Estructura de tu carpeta

*incluir una imagen de estructura ideal* 

> No olvides que antes de iniciar tu aplicación, debes guardar tu modelo. Puedes usar el código de abajo.

```py
import pickle

filename = 'titanic_model.pkl'

pickle.dump(classifier, open(filename,'wb'))
```

**Uso de Flask para crear una API web para nuestro modelo de Machine Learning**

Lo que normalmente usabas para nombrar app.py deberá cambiarse, por ejemplo, a 'titanic.py' o 'build_features.py', etc. Porque tu archivo app.py ahora estará a cargo de construir la aplicación web.

En tu archivo app.py, tu código debería verse así:

```py
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('titanic_model.pkl','rb'))

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

    output = round(prediction[0],2)

    return render_template('index.html',prediction_text='Would you survive? {} (1=survived, 0=deceased)'.format(output))

if __name__=="__main__":
    app.run(port=5000, debug=True)
```


Fuente:

https://medium.com/towards-data-science/considerations-for-deploying-machine-learning-models-in-production-

https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3-es

https://medium.com/towards-data-science/designing-a-machine-learning-model-and-deploying-it-using-flask-on-heroku-9558ce6bde7b

https://medium.com/towards-data-science/create-an-api-to-deploy-machine-learning-models-using-flask-and-heroku-67a011800c50

https://medium.com/towards-data-science/productionize-a-machine-learning-model-with-flask-and-heroku-8201260503d2

https://medium.com/towards-data-science/flask-and-heroku-for-online-machine-learning-deployment-425beb54a274
