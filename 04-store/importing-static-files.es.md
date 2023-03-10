# Importación de archivos en Python

En esta lectura vamos a ver ejemplos de código sobre cómo cargar diferentes tipos de archivos. No es código ejecutable. Se puede usar como referencia siempre que cargues un nuevo tipo de archivo en tu computadora portátil.

### Importación de archivos planos con Numpy

```py
import numpy as np
 
# Asignar nombre de archivo a la variable: archivo
file = 'digits.csv'
 
# Cargar archivo como array: dígitos
digits = np.loadtxt(file, delimiter=',')
 
# Imprimir tipo de datos de dígitos
print(type(digits))
```

Hay una serie de argumentos que toma np.loadtxt() que encontrarás útiles para cambiar:

- Delimitador cambia el delimitador que espera loadtxt().

- Skiprows te permite especificar cuántas filas (no índices) deseas omitir (por ejemplo, si no deseas incluir el encabezado).

- Usecols toma una lista de los índices de las columnas que deseas conservar.

Si tenemos un archivo que incluye un encabezado que consta de strings (cadenas) y tratamos de importarlo tal como está usando np.load_txt(), Python nos lanzaría un ValueError diciendo que no pudo convertir la string en flotante (float). Tenemos dos formas de solucionar esto:

1. Establece el argumento de tipo de datos dtype igual a str (por string).
2. Omite la primera fila usando el argumento skiprows.

¿Qué sucede si tenemos diferentes tipos de datos en diferentes columnas?

La función np.loadtxt() se asustará con esto, pero hay otra función, np.genfromtxt(), que puede manejar tales estructuras. Al escribir dtype=None en él, averiguará qué tipos debe tener cada columna. El parámetro names=True indica que hay un encabezado.

También hay otra función, np.recfromcsv(), que se comporta de manera similar a np.genfromtxt(), excepto que su dtype predeterminado es Ninguno.

### Importación de archivos planos como marcos de datos con Pandas

**ARCHIVOS CSV**

Podemos importar fácilmente archivos de tipos de datos mixtos como DataFrames usando las funciones de Pandas read_csv() y read_table().

```py
import pandas as pd
 
# Asignar el nombre de archivo a una variable
file = 'titanic.csv'
 
# Leer el archivo en una variable DataFrame
df = pd.read_csv(file)
 
# Ver las primeras filas del DataFrame
print(df.head())
```

También es posible recuperar el array Numpy correspondiente utilizando los valores de los atributos.

```py
# Crear un array Numpy desde DataFrame: data_array
data_array = np.array(data.values)
```

A veces nos encontraremos lidiando con archivos corruptos que pueden incluir comentarios, valores faltantes, etc.

Podemos cargar esos archivos dañados con Pandas de la siguiente manera:

```py
# Importar archivo: datos
data = pd.read_csv(file, sep='\t', comment='#', na_values='Nothing')
```

**ARCHIVOS EXCEL**

En algún momento, también tendremos que lidiar con archivos de Excel. Dado un archivo de Excel importado en una variable, puedes recuperar una lista de los nombres de las hojas usando el atributo sheet_names.

```py
import pandas as pd
 
# Asignar hoja de cálculo a una variable de archivo
file = 'battledeath.xlsx'
 
# Cargar hoja de cálculo: excel
excel_file = pd.ExcelFile(file)
 
# Imprimir nombres de hojas
print(excel_file.sheet_names)


```

Aprenderemos cómo importar cualquier hoja dada de nuestro archivo .xslx cargado como un DataFrame. Podremos hacerlo especificando el nombre de la hoja o su índice.

```py
# Cargar la hoja '2004' en un DataFrame df1
df1 = excel_file.parse('2004')

# Imprimir el encabezado del DataFrame df1
print(df1.head())

# Cargar una hoja en un DataFrame por índice: df2
df2 = excel_file.parse(0)

# Imprimir el encabezado del DataFrame df2
print(df2.head())
```

Hemos utilizado el método parse(). Sin embargo, podemos agregar argumentos adicionales como skiprows, names y parse_cols. Estos argumentos saltan filas, nombran las columnas y designan qué columnas analizar, respectivamente. Todos estos argumentos se pueden asignar a listas que contienen números de fila, cadenas y números de columna específicos, según corresponda.

```py
# Analizar la primera columna de la segunda hoja y cambiar el nombre de la columna: df2

df2 = excel_file.parse(1, parse_cols=[0], skiprows=[0], names=['City'])
```

**ARCHIVOS STATA**

Cómo importar un archivo Stata como DataFrame usando la función pd.read_stata() de Pandas:

```py
import pandas as pd
import matplotlib.pyplot as plt
 
# Cargar el archivo Stata en un DataFrame de Pandas: df
df = pd.read_stata('examplefile.dta')

# Histograma de lotes de una columna del DataFrame
pd.DataFrame.hist(df[['column1']])
```

**ARCHIVOS HDF5**

```py
import numpy as np
import h5py
 
file = 'examplefile.hdf5'
 
# Cargar archivo: datos
data = h5py.File(file, 'r')
```

**ARCHIVOS MATLAB**

En el caso de archivos matlab usaremos scipy.

```py
import scipy.io
 
mat = scipy.io.loadmat('examplefile.mat')
```
