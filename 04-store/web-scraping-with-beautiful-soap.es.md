#  Raspado Web

El Raspado Web es uno de los métodos importantes para recuperar datos de un sitio web automáticamente. No todos los sitios web permiten que las personas raspen, sin embargo, puede agregar 'robots.txt' después de la URL del sitio web que deseas raspar, para saber si se le permitirá raspar o no.

¿Cómo obtenemos datos de un sitio web?

Hay tres formas en las que podemos obtener datos de la web:

1. Importando de archivos de Internet.

2. Hacer Raspado Web directamente con código para descargar el contenido HMTL.

3. Consultando datos de la API del sitio web.

Pero, ¿qué es una API de sitio web?

Una API (Interfaz de Programación de Aplicaciones) es una interfaz de software que permite que dos aplicaciones interactúen entre sí sin la intervención del usuario. Se puede acceder a una API web a través de la web utilizando el protocolo HTTP.

Las herramientas de raspado son un software especialmente desarrollado para extraer datos de sitios web. ¿Cuáles son las herramientas más comunes para el Raspado Web?

- Solicitudes: Es un módulo de Python en el que podemos enviar solicitudes HTTP para recuperar contenidos. Nos ayuda a acceder a los contenidos HTML o API del sitio web mediante el envío de solicitudes Get o Post.

- Beautiful Soup: Nos ayuda a analizar documentos HTML o XML en un formato legible. Podemos recuperar información más rápido.

- Selenio: se utiliza principalmente para pruebas de sitios web. Ayuda a automatizar diferentes eventos.



### 1. Importación de archivos planos desde la web

El archivo plano que importaremos será el dataset de iris de http://archive.ics.uci.edu/ml/machine-learning-databases/iris/ obtenido del repositorio de Machine Learning UCI.

Después de importarlo, lo cargaremos en un dataframe (marco de datos) de Pandas.


```python
# Importar paquete
from urllib.request import urlretrieve

# Importar Pandas
import pandas as pd

# Asignar url de archivo: url
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Guardar archivo localmente
urlretrieve(url,'iris.csv')

# Leer el archivo en un dataframe y mirar las primeras filas
df = pd.read_csv('iris.csv', sep=';')
print(df.head())

```

       5.1,3.5,1.4,0.2,Iris-setosa
    0  4.9,3.0,1.4,0.2,Iris-setosa
    1  4.7,3.2,1.3,0.2,Iris-setosa
    2  4.6,3.1,1.5,0.2,Iris-setosa
    3  5.0,3.6,1.4,0.2,Iris-setosa
    4  5.4,3.9,1.7,0.4,Iris-setosa


### 2. Realización de solicitudes HTTP

Pasos para hacer solicitudes HTTP:

1. Inspeccionar el HTML del sitio web que queremos raspar (clic derecho).

2. Acceder a la URL del sitio web usando código y descargar todo el contenido HTML en la página.

3. Dar formato al contenido descargado en un formato legible.

4. Extraer información útil y guardarla en un formato estructurado.

5. Si la información se encuentra en varias páginas del sitio web, es posible que debamos repetir los pasos 2 a 4 para tener la información completa.

**Realización de solicitudes HTTP utilizando urllib**

Extraeremos el HTML en sí, pero primero empaquetaremos, enviaremos la solicitud y luego capturaremos la respuesta.


```python
# Importar paquetes
from urllib.request import urlopen, Request

# Especificar la URL
url = " https://scikit-learn.org/stable/getting_started.html"

# Esto empaqueta la solicitud
request = Request(url)

# Envíar la solicitud y capturar la respuesta
response = urlopen(request)

# Imprimir el tipo de datos de la respuesta
print(type(response))

# ¡Cierrar la respuesta!
response.close()
```

Esta respuesta es un objeto http.client.HTTPResponse. ¿Qué podemos hacer con él?

Como viene de una página HTML, podemos leerlo para extraer el HTML usando un método read() asociado a él. Ahora extraigamos la respuesta e imprimamos el HTML.


```python
request = Request(url)

response = urlopen(request)

# Extraer la respuesta: html
html = response.read()

# imprimir el html
print(html)

# ¡Cerrar la respuesta!
response.close()

```

**Realizando solicitudes HTTP mediante solicitudes**

Ahora vamos a usar la biblioteca de solicitudes. Esta vez no tenemos que cerrar la conexión.


```python
import requests

# Especificar la url: url
url = "https://scikit-learn.org/stable/getting_started.html"

# Empaquetar la solicitud, envíar la solicitud y capturar la respuesta: resp
resp = requests.get(url)

# Extraer la respuesta: texto
text = resp.text

# imprimir el html
print(text)

```

**Analizando HTML usando Beautiful Soup**

Aprenderemos a usar el paquete BeautifulSoup para analizar, embellecer y extraer información de HTML.


```python
# Importar paquetes
import requests
from bs4 import BeautifulSoup

# Especificar la url: url
url = 'https://gvanrossum.github.io//'

# Empaquetar la solicitud, enviar la solicitud y obtener la respuesta: resp
resp = requests.get(url)

# Extraer la respuesta como html: html_doc
html_doc = resp.text

# Luego, ¡todo lo que tenemos que hacer es convertir el documento HTML en un objeto BeautifulSoup!
soup = BeautifulSoup(html_doc)

# Embellecer el objeto BeautifulSoup: pretty_soup
pretty_soup = soup.prettify()

# Imprimir la respuesta
print(pretty_soup)
```

**Los tags se pueden llamar de diferentes maneras**


```python
# Esta línea de código crea un objeto BeautifulSoup desde una página web:
 
soup = BeautifulSoup(webpage.content, "html.parser")
 
# Dentro del objeto `sopa`, los tags se pueden llamar por su nombre:
 
first_div = soup.div
 
# O por selector de CSS:
 
all_elements_of_header_class = soup.select(".header")
 
# O por una llamada a `.find_all`:
 
all_p_elements = soup.find_all("p")
```

### 3. Interactuando con las API

Es un poco más complicado que raspar el documento HTML, especialmente si se requiere autenticación, pero los datos serán más estructurados y estables.

Pasos para consultar datos de la API del sitio web:

1. Inspeccionar la sección de red XHR de la URL que queremos raspar.

2. Averiguar la petición-respuesta que nos da los datos que queremos.

3. Dependiendo del tipo de solicitud (publicar u obtener), simulemos la solicitud en nuestro código y recuperemos los datos de la API. Si se requiere autenticación, primero deberemos solicitar el token antes de enviar nuestra solicitud POST.

4. Extraer información útil que necesitamos.

5. Para API con un límite en el tamaño de la consulta, necesitaremos usar 'for loop' para recuperar repetidamente todos los datos

**Ejemplo: cargando y explorando un Json con solicitud GET**



```python
# Importar paquete
import requests

# Asignar url a la variable: url
url = "https://covid-19-statistics.p.rapidapi.com/regions"

headers = {
	"X-RapidAPI-Host": "covid-19-statistics.p.rapidapi.com",
	"X-RapidAPI-Key": "SIGN-UP-FOR-KEY"
}

response = requests.request("GET", url, headers=headers)

# Decodificar los datos JSON en un diccionario: json_data
json_data = response.json()

# Imprimir cada par clave-valor en json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])

```

Si deseas raspar un sitio web, primero debemos verificar la existencia de API en la sección de red usando inspeccionar. Si podemos encontrar la respuesta a una solicitud que nos proporcione todos los datos que necesitamos, podemos construir una solución estable. Si no podemos encontrar los datos en la red, podemos intentar usar solicitudes o Selenium para descargar contenido HTML y usar Beautiful Soup para formatear los datos.

Otras herramientas principales de Raspado Web en 2022:

1. Newsdata.io

2. Scrapingbee 

3. Bright Data

4. Scraping-bot 

5. Scraper API 

6. Scrapestack 

7. Apify 

8. Agenty 

9. Import.io

10. Outwit 

11. Webz.io 

Referencias: 

https://towardsdatascience.com/web-scraping-basics-82f8b5acd45c

https://rapidapi.com/rapidapi/api

https://newsdata.io/blog/top-21-web-scraping-tools-for-you/

