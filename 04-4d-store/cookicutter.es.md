# Creando una Estructura de Proyecto

Te ha pasado que cada vez que quieres crear un nuevo proyecto copias toda la carpeta de un proyecto anterior con el mismo conjunto de código y luego comienzas a reemplazar variables, renombrar carpetas y cambiar manualmente sus entradas de código, con la esperanza de no olvidar nada en el camino. Esta es una tarea bastante tediosa y repetitiva; sin mencionar que es propenso a errores. Es por eso que queremos presentarte una herramienta increíble: ¡Cookiecutter!

## Cookiecutter

¡Cookiecutter es una herramienta poderosa! Esta es una manera increíble de crear una plantilla de proyecto para un tipo de análisis que sabes que necesitarás repetir varias veces, mientras ingresas los datos y/o parámetros necesarios solo una vez.

¿Qué es Cookiecutter?

Los proyectos pueden ser paquetes de Python, aplicaciones web, aplicaciones de Machine Learning con flujos de trabajo complejos o cualquier cosa que se te ocurra. Las plantillas son lo que utiliza el cookiecutter para crear proyectos. Lo que hace cookiecutter es bastante simple: clona un directorio y lo coloca dentro de su nuevo proyecto. Luego reemplaza todos los nombres que están entre {{ y }} (sintaxis de Jinja2) con nombres que encuentra en el archivo cookiecutter.json. Lo mejor es que también cuenta con una plantilla específica para proyectos de data science (ciencia de datos) y Machine Learning. (Veremos un ejemplo de cómo construir una plantilla de cookiecutter)

Cookiecutter debe ser parte de su entorno si deseas utilizarlo.

Puedes instalarlo con pip:

```py
pip install cookiecutter
```

o si estás usando Anaconda:

```py
conda config --add channels conda-forge
conda install cookiecutter
```

¡Ya instalado! ¡Ahora puede usar cookiecutter para crear nuevas plantillas para proyectos y documentos!



## Gitpod

Ahora que tienes un proyecto muy bien estructurado, probablemente ya lo hayas subido a tu cuenta de GitHub.

La mejor manera de configurar Gitpod es usando Gitpod. En un navegador, ve a la página de GitHub, GitLab o Bitbucket de tu proyecto.

En la barra de direcciones del navegador, prefija la URL completa con gitpod.io/# y presiona Enter.

Por ejemplo, gitpod.io/#https://github.com/gitpod-io/website

te recomendamos que instales la extensión del navegador Gitpod para que esto sea una operación de un solo clic.

**Botón Abrir en Gitpod**

Para facilitar que cualquier persona inicie un espacio de trabajo de Gitpod basado en tu proyecto, te recomendamos que agregues un botón "Abrir en Gitpod" a tu README.md.

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#<your-project-url>)

**Agrega tu archivo .gitpod.yml a un repositorio de Github ya existente**

Para cualquier aplicación web, lo más probable es que tenga algún tipo de instalación y ejecución de comandos que luego sirvan en un puerto. Por ejemplo, este es el contenido de un archivo .gitpod.yml para una aplicación Nuxt que usa Yarn:

```py
tasks:
  - init: yarn install
    command: yarn dev
ports:
  - port: 3000
    onOpen: open-preview
```

Cuando el contenedor se active, instalará las dependencias y luego servirá al puerto 3000. También abrirá una vista previa de la aplicación cuando esté lista.

**Agrega tu .gitpod.Dockerfile (opcional)**

De manera predeterminada, Gitpod usa un Dockerfile generalizado, pero puedes especificar el tuyo propio creando este archivo y personalizándolo a tu gusto.

Ve a la URL de tu repositorio de Github y prefija con #gitpod.io.

¡¡Eso es todo!!

**Algunos beneficios adicionales de usar Gitpod:**

- Olvídate de las computadoras portátiles y los sistemas operativos costosos.

Elimina la necesidad de comprar una computadora portátil costosa con un montón de poder de cómputo y a quién le importa qué sistema operativo tiene. Podrías tener un Chromebook de $200 o usar una computadora pública en la biblioteca local y hacer el mismo desarrollo que harías en un MacBook Pro. Siempre que puedas ejecutar un navegador, está bien.

La gente de Machine Learning ya lo sabe. Ejecutan Jupyter notebooks en la nube en GPU, en lugar de gastar miles y todos los dolores de cabeza de hacerlo ellos mismos.

- Elimina los dolores de cabeza de incorporación

*Si quieres saber más sobre cómo potenciar la experiencia con Gitpod para tu proyecto, puedes ir a la siguiente guía: https://www.gitpod.io/docs/getting-started*

## Entendiendo cookiecutter con un proyecto Streamlit

Streamlit es una biblioteca de Python diseñada para crear aplicaciones web. Es muy simple de usar y proporciona muchas funcionalidades que te permiten compartir experimentos y resultados con tu equipo y crear prototipos de aplicaciones de Machine Learning.

Veamos esta estructura de proyecto común de Streamlit de un ingeniero de Machine Learning:

- Carpeta src que contiene:

    - El script principal de la aplicación (app.py).
    
    - Módulo utils que contiene dos scripts: 

        - ui.py: para poner las funciones de diseño.

        - common.py: para contener otras funciones de utilidad para el procesamiento de datos o conexiones de bases de datos remotas (entre otras cosas).

- Archivo .gitignore: para evitar que git controle versiones de archivos innecesarios (como archivos .env o .pyc).

- Procfile y setup.sh: para manejar la implementación en Heroku.

- requisitos.txt: para enumerar las dependencias del proyecto.

- Archivo .env: para almacenar las variables de entorno del proyecto.

- README.md: para compartir detalles sobre el proyecto.

Vamos a crear una plantilla de cookiecutter desde cero que coincida con la estructura de destino anterior. Así entenderemos cómo funciona.

Comencemos por crear una carpeta para esta plantilla.

```py
mkdir streamlit-cookiecutter
cd streamlit-cookiecutter
```

Dentro de esta carpeta, crea un archivo cookiecutter.json que contenga la siguiente información:

```py
{
  "project_name": "my streamlit app",
  "repo_name": "{{ cookiecutter.project_name.lower().replace(' ', '_') }}",
  "author_name": "Ahmed BESBES",
  "description": "A short description of the project.",
  "open_source_license": ["MIT", "BSD-3-Clause", "No license file"],
  "s3_bucket": "",
  "aws_profile": "default",
  "python_interpreter": ["python3", "python"],
  "host": "localhost",
  "port": 8501,
  "api_key": "XXXXXXXXXXXX"
}
```

¿Por qué estamos creando este archivo?

El archivo cookiecutter.json incluye claves y valores. Los valores de los elementos se establecen en orden. Por ejemplo, repo_name se establece después de project_name (ya que su valor depende del valor de project_name). La configuración de los valores de estos elementos ocurre cuando clonamos la plantilla usando cookiecutter, porque cookiecutter los solicitará.

Ahora creemos la carpeta y coloquemos la estructura de destino deseada en ella. Nombraremos esta carpeta según el elemento repo_name.

En la sintaxis de cookiecutter: {{cookiecutter.repo_name}}.

Cada uno de estos archivos puede acceder a los valores de los elementos que pasamos a cookiecutter: todo lo que tenemos que hacer es usar {{ y }}. Esto será útil para generar el README.md, establecer variables de entorno y deduplicar código automáticamente.

**README.md**

Podemos generar automáticamente el README.md insertando los elementos project_name, description, open_source_license en él.

![README.jpg](../assets/README.jpg)

**Establecer variables de entorno**

Podemos completar automáticamente los valores de s3_bucket, aws_profile, port, host y api_key dentro del archivo .env

![environment_variables.jpg](../assets/environment_variables.jpg)

**Código deduplicado**

Si nuestras aplicaciones Streamlit siguen la misma estructura y todas comienzan con el nombre del proyecto como título, no hay necesidad de repetir este código cada vez.

![deduplicate_code.jpg](../assets/deduplicate_code.jpg)

Una vez que haya terminado con el código de la plantilla, podemos enviarlo a Github, y cualquier persona a la que le haya gustado nuestra estructura puede usarla para iniciar un proyecto. 

Digamos que queremos crear una aplicación de análisis de sentimientos en Streamlit. Creamos una carpeta vacía y abrimos la terminal dentro de nuestra carpeta, todo lo que tenemos que hacer es llamar a cookiecutter con la URL de la plantilla.

Ejemplo: cookiecutter git@github.com:maria/streamlit-cookiecutter.git

Una vez que ejecutamos este comando, Cookiecutter le pedirá que establezca los valores de los elementos definidos en el archivo cookiecutter.json (observe que el valor predeterminado de cada elemento se pone entre paréntesis).

![cookiecutter.jpg](../assets/cookiecutter.jpg)

Veamos cómo se han llenado los archivos:

![README_filled.jpg](../assets/README_filled.jpg)

![env_file.jpg](../assets/env_file.jpg)

![app_file.jpg](../assets/app_file.jpg)

**Buenas prácticas para incluir en tus proyectos**

- Manten los secretos y la configuración fuera del control de versiones.

Realmente no deseas filtrar tu clave secreta de Twitter o el nombre de usuario y la contraseña de la base de datos en Github. Esta es una forma de hacerlo, almacenando tus secretos y variables de configuración en un archivo especial (lo aprendiste en la plantilla Cookiecutter)

Crea un archivo .env en la carpeta raíz del proyecto. Gracias a .gitignore, este archivo nunca debería comprometerse en el repositorio de control de versiones. Aquí hay un ejemplo:

```py
#Ejemplo de archivo .env
DATABASE_URL=postgres://username:password@localhost:5432/dbname
AWS_ACCESS_KEY=myaccesskey
AWS_SECRET_ACCESS_KEY=mysecretkey
OTHER_VARIABLE=something
```

- Usando un paquete para cargar estas variables automáticamente

Hay un paquete llamado python-dotenv para cargar todas las entradas en este archivo como variables de entorno para que sean accesibles con os.environ.get. Aquí hay un fragmento de código de ejemplo adaptado de la documentación de python-dotenv aplicada en la plantilla de Data Science (ciencia de datos) de cookiecutter:

```py
# src/data/dotenv_example.py
import os
from dotenv import load_dotenv, find_dotenv

# encuentra .env automáticamente recorriendo directorios hasta que se encuentre
dotenv_path = find_dotenv()

# carga las entradas como variables de entorno
load_dotenv(dotenv_path)

database_url = os.environ.get("DATABASE_URL")
other_variable = os.environ.get("OTHER_VARIABLE")
```

Hemos visto cómo trabajar con cookiecutter en un proyecto streamlit común. Pero Cookiecutter tiene muchas plantillas para diferentes tipos de proyectos, así que como puedes imaginar, Data Science no es la excepción.

Utiliza la plantilla de código abierto Cookiecutter Data Science para iniciar proyectos de ciencia de datos que sigan los mejores estándares de la industria.

Para iniciar un nuevo proyecto de Data Science, puedes abrir la terminal en la carpeta de tu proyecto y escribir:

```py
cookiecutter https://github.com/drivendata/cookiecutter-data-science
```

¡¡Eso es todo!! Ahora puedes ver toda una estructura muy organizada, legible y reutilizable. Tómate un tiempo para explorar cómo está estructurado.

**Construir desde el entorno hacia arriba**

El primer paso para reproducir un análisis siempre es reproducir el entorno computacional en el que se ejecutó. Necesitas las mismas herramientas, las mismas bibliotecas y las mismas versiones para que todo funcione bien en conjunto.

Un enfoque efectivo para esto es usar virtualenv. Al enumerar todos sus requisitos en el repositorio (se incluye un archivo requirements.txt), puedes rastrear fácilmente los paquetes necesarios para recrear el análisis. Aquí hay un buen flujo de trabajo:

1. Ejecuta mkvirtualenv al crear un nuevo proyecto.

2. Pip instala los paquetes que tu análisis necesita.

3. Corre pip freeze > requirements.txt para anclar las versiones exactas del paquete utilizadas para recrear el análisis

4. Si encuentras que necesitas instalar otro paquete, ejecuta pip freeze > requirements.txt nuevamente y confirma los cambios en el control de versiones.
 
Cualquier persona que quiera usar tu proyecto solo necesitará escribir el siguiente comando para tener tus mismas dependencias:

```py
pip install -r requirements.txt
```
Es una buena práctica ejecutar el comando anterior cada vez que estés a punto de comenzar a editar un proyecto.

*Como referencia, dejaremos la documentación de cookiecutter-data-science en el siguiente enlace:*

https://drivendata.github.io/cookiecutter-data-science/

Fuente: 

https://www.gitpod.io/docs/getting-started

https://medium.com/@lukaskf/why-you-should-consider-using-cloud-development-environments-a79c062a2798

https://cookiecutter-data-science-vc.readthedocs.io/en/latest/getting_started/INSTALL.html

https://github.com/drivendata/cookiecutter-data-science/issues/17

https://towardsdatascience.com/automate-the-structure-of-your-data-science-projects-with-cookiecutter-937b244114d8

https://drivendata.github.io/cookiecutter-data-science/

https://github.com/drivendata/cookiecutter-data-science
