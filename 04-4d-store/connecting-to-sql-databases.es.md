# Cómo conectarse a bases de datos SQL usando Python

Como ingenieros de Machine Learning, probablemente tengamos que interactuar con bases de datos SQL para acceder a los datos. SQL significa "Structured Query Language" (lenguaje de consulta estructurado). La diferencia clave entre SQL y Python es que los desarrolladores usan SQL para acceder y extraer datos de una base de datos, mientras que los desarrolladores usan Python para analizar y manipular datos ejecutando pruebas de regresión, pruebas de series temporales y otros cálculos de procesamiento de datos.

Algunas bases de datos SQL populares son SQLite, PostgreSQL, MySQL. SQLite es mejor conocido por ser una base de datos integrada. Esto significa que no tenemos que instalar una aplicación adicional o usar un servidor separado para ejecutar la base de datos. Se mueve rápido pero tiene una funcionalidad limitada, por lo que si no necesitamos mucho espacio de almacenamiento de datos, querremos usar una base de datos SQLite. Por otro lado, PostgreSQL y MySQL tienen tipos de bases de datos que son excelentes para soluciones empresariales. Si necesitamos escalar rápido, MySQL y PostgreSQL son la mejor opción. Proporcionarán infraestructura a largo plazo, reforzarán la seguridad y manejarán actividades de alto rendimiento.

En esta lección veremos cómo interactúan Python y algunas bases de datos SQL. ¿Por qué deberíamos preocuparnos por conectar Python y una base de datos SQL?

Tal vez, como ingenieros de Machine Learning, necesitemos construir un "ETL pipeline" (tubería ETL) automatizado. Conectar Python a una base de datos SQL nos permitirá usar Python para sus capacidades de automatización. También podremos comunicarnos entre diferentes fuentes de datos. No tendremos que cambiar entre diferentes lenguajes de programación, podremos usar nuestras habilidades de Python para manipular datos de una base de datos SQL. No necesitaremos un archivo CSV.

Lo importante a recordar es que Python puede integrarse con cada tipo de base de datos. Las bases de datos de Python y SQL se conectan a través de bibliotecas de Python personalizadas. Puede importar estas bibliotecas a su secuencia de comandos de Python.

**El siguiente es un ejemplo de código sobre cómo conectarse a una base de datos SQL:**


```py
from dbmodule import connect 

#Crea un objeto de conexión

CONNECTION = CONNECT('databse name', 'username','password')

#Crea un objeto de cursor

CURSOR = CONNECTION.CURSOR()

#Ejecuta consultas

CURSOR.EXECUTE('select * from mytable')
RESULTS = CURSOR.FETCHALL()

#Recursos gratuitos

CURSOR.CLOSE()
```

**Código de ejemplo para conectarse a una base de datos PostreSQL y almacenar datos en un dataframe (marco de datos) de Pandas:**

En este caso, elegimos AWS Redshift. Importaremos la biblioteca Psycopg. Esta biblioteca traduce el código Python que escribimos para hablar con la base de datos PostgreSQL (AWS Redshift).

De lo contrario, AWS Redshift no entendería nuestro código de Python. Pero debido a la biblioteca Psycopg, ahora hablará un idioma que AWS Redshift puede entender.

```py
#Biblioteca para conectarse a AWS Redshift
import psycopg

#Biblioteca para leer el archivo de configuración, que está en JSON
import json

#Biblioteca de manipulación de datos
import pandas as pd
```

Importamos JSON porque crear un archivo de configuración JSON es una forma segura de almacenar las credenciales de su base de datos. ¡No queremos que nadie más los mire! La función json.load() lee el archivo JSON para que podamos acceder a las credenciales de nuestra base de datos en el siguiente paso.

```py
config_file = open(r"C:\Users\yourname\config.json")
config = json.load(config_file)
```

Ahora queremos crear una conexión de base de datos. Tendremos que leer y usar las credenciales de nuestro archivo de configuración:

```py
con = psycopg2.connect(dbname= "db_name", host=config[hostname], port = config["port"],user=config["user_id"], password=config["password_key"])
cur = con.cursor()
```

### Crear y conectarse a una base de datos SQLite usando Python

Como ya mencionamos, SQLite es un sistema de administración de bases de datos de relaciones que es liviano y fácil de configurar. SQLite no tiene servidor, lo cual es su mayor ventaja. No requiere un servidor para ejecutar una base de datos, a diferencia de otros RDMS como MySQL o PostgreSQL. Así que no necesitamos ninguna configuración de instalación.

Las bases de datos SQLite se almacenan localmente, con archivos almacenados en el disco. Esto hace que acceder y administrar los datos en la base de datos sea notablemente rápido.

**Código de ejemplo para crear una base de datos:**

```py
import sqlite3

connection = sqlite3.connect('shows.db')  #creando una base de datos con nombre:
cursor = connection.cursor()   #crea un objeto de cursor para crear una tabla
cursor.execute('''CREATE TABLE IF NOT EXISTS Shows
              (Title TEXT, Director TEXT, Year INT)''')  #crear una tabla con nombres de columnas y tipos de datos

connection.commit()  #confirmar los cambios en la base de datos
connection.close()  #cerrar la conexión
```

Después de ejecutar el archivo, en el directorio de su proyecto actual, se crea un archivo llamado shows.db. Este es el archivo de base de datos SQLite generado por Python.

**Código de ejemplo para conectarse a la base de datos:**

```py

from sqlalchemy import create_engine
import pandas as pd
 
#Crea motor: motor
engine = create_engine('sqlite:///databse_name.sqlite')
 
#Guarda los nombres de las tablas en una lista: table_names
table_names = engine.table_names()
 
#Imprime los nombres de las tablas en el shell.
print(table_names)
 
#Abre la conexión del motor: "con" (estafa) y seleccione las columnas y el número de filas especificados

with engine.connect() as con:
    ab = con.execute("SELECT Title, Director FROM Shows")
    df = pd.DataFrame(ab.fetchmany(size=5))
    df.columns = ab.keys()

#Conexión cercana
con.close()
 
#Imprime las primeras filas del dataframe
print(df.head())
```

### Conectando a una base de datos DB2

IBM Db2 es una familia de productos de gestión de datos, incluida la base de datos relacional Db2. El plan gratuito proporciona 200 MB de almacenamiento de datos en la nube. Para practicar la creación de una base de datos SQL y la escritura de consultas SQL, este es un buen lugar para comenzar.

Podemos crear nuestras tablas en la nube o directamente desde nuestro notebook (cuaderno) usando Python. Para hacerlo con Python, primero debemos conectarnos a nuestra base de datos en la nube utilizando las credenciales que se nos proporcionaron en el momento en que se creó la instancia de la base de datos.

Para conectarse a un DB2, se requiere la siguiente información:

- Nombre del controlador.
- Nombre de la base de datos.
- Nombre de host DNS o IP.
- Puerto host.
- Protocolo de conexión.
- Identificación de usuario.
- Contraseña.

Ejemplo para crear una conexión de base de datos:

```py
#Crea conexión de base de datos

dsn = (
    "Driver = { {IBM DB2 ODBC DRIVER}};"
    "Database = {0};"
    "HOSTNAME = {1};"
    "PORT = {2};"
    "PROTOCOL = TCPIP;"
    "UID = {3};"
    "PWD = {4};").format(dsn_database, dsn_hostname, dsn_port, dsn_uid, dsn_pwd)

try: 
    conn = ibm_db.connect(dsn, " ", " ")
    print("Connected!")
    
except:
    print("Unable to connect to database")
    
#Cierra la conexión a la base de datos

ibm_db.close(conn)

#Nota: Siempre es importante cerrar las conexiones para evitar que los conectores no utilizados consuman recursos.
```

**Cómo crear una tabla desde Python**

ibm_db.exec_inmediate()  --> función de la API ibm_db

Parámetros para la función:

- Conexión.
- Declaración.
- Opciones.

Ejemplo: Creando una tabla llamada CARS en Python.


#CREAR TABLA
```py
stmt = ibm_db.exec_inmediate(conn, "CREATE TABLE Cars(
    serial_no VARCHAR(20) PRIMARY KEY NOT NULL,
    make varchar(20) NOT NULL,
    model VARCHAR(20) NOT NULL,
    car_class VARCHAR(20) NOT NULL)"
    )
```

#CARGAR DATOS EN LA TABLA
```py
stmt = ibm_db.exec_inmediate(conn, "INSERT INTO Cars(
    serial_no, make, model, car_class)
    VALUES('A2345453','Ford','Mustang','class3');")
```

#OBTENER DATOS DE LA TABLA CARS 
```py
stmt = ibm_db.exec_inmediate(conn, "SELECT * FROM Cars")

ibm_db.fetch_both(stmt)
```

**Using pandas to retrieve data from the tables**

Example:


```py

import pandas
import ibm_db_dbi
pconn = ibm_db_dbi.connection(conn)

df = pandas.read_sql('SELECT * FROM Cars', pconn)
df

#Ejemplo de una gráfica

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns 

#Diagrama de dispersión categórica

plot = sns.swarmplot(x="Category", y="Calcium", data=df)
plt.setp(plot.get_xticklabels(), rotation=70)
plt.title('Calcium content')
plt.show()

#Haciendo un diagrama de caja

#Un diagrama de caja es un gráfico que indica la distribución de 1 o más variables. La caja captura la mediana del 50% de los datos..\

#La línea y los puntos indican posibles valores atípicos y no valores normales.

plot = sns.set_style('Whitegrid')
ax = sns.boxplot(x=df['glucose level'])
plt.show()
```

**Obteniendo las propiedades**

DB2 --->  syscat.tables                                 

SQL Server --->  information=schema.tables   
 
Oracle --->  all_tables or user_tables


```py

#Obteniendo propiedades de tabla de DB2

SELECT * FROM syscat.tables
#(esto mostrará demasiadas tablas)

SELECT tabschema, tabname, create_time
FROM syscat.tables
WHERE tabschema = 'ABC12345' #---> reemplaza con tu propio nombre de usuario de DB2

#Obtener una lista de columnas en la base de datos

SELECT * FROM syscat.columns
WHERE tabname = 'Cats'

#Para obtener propiedades de columna específica:

%sql SELECT DISTINCT(name), coltype, length
    FROM sysibm.syscolumns
    WHERE tbname = 'Cats'
    
%sql SELECT DISTINCT(name), coltype, length
    FROM sysibm.syscolumns 
    WHERE tbname = 'Miami_crime_data'

```

Fuente:

https://www.freecodecamp.org/news/python-sql-how-to-use-sql-databases-with-python/#:~:text=Perhaps%20you%20work%20in%20data%20engineering%20and%20you,be%20able%20to%20communicate%20between%20different%20data%20sources.
