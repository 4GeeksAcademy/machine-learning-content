# Explorando series de tiempo

## Manipulación de Series Temporales usando Pandas


```python
# Crear un rango de fechas con frecuencia por hora

import pandas as pd
from datetime import datetime
import numpy as np

date_rng = pd.date_range(start='1/1/2018', end='1/08/2018', freq='H')
```


```python
date_rng
```




    DatetimeIndex(['2018-01-01 00:00:00', '2018-01-01 01:00:00',
                   '2018-01-01 02:00:00', '2018-01-01 03:00:00',
                   '2018-01-01 04:00:00', '2018-01-01 05:00:00',
                   '2018-01-01 06:00:00', '2018-01-01 07:00:00',
                   '2018-01-01 08:00:00', '2018-01-01 09:00:00',
                   ...
                   '2018-01-07 15:00:00', '2018-01-07 16:00:00',
                   '2018-01-07 17:00:00', '2018-01-07 18:00:00',
                   '2018-01-07 19:00:00', '2018-01-07 20:00:00',
                   '2018-01-07 21:00:00', '2018-01-07 22:00:00',
                   '2018-01-07 23:00:00', '2018-01-08 00:00:00'],
                  dtype='datetime64[ns]', length=169, freq='H')




```python
type(date_rng[0])
```




    pandas._libs.tslibs.timestamps.Timestamp



Ahora vamos a crear un marco de datos de ejemplo con los datos de marca de tiempo que acabamos de crear


```python
df = pd.DataFrame(date_rng, columns=['date'])
df['data'] = np.random.randint(0,100,size=(len(date_rng)))

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>data</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2018-01-01 00:00:00</td>
      <td>23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018-01-01 01:00:00</td>
      <td>73</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2018-01-01 02:00:00</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2018-01-01 03:00:00</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2018-01-01 04:00:00</td>
      <td>44</td>
    </tr>
  </tbody>
</table>
</div>



Si queremos manipular series de tiempo, necesitaremos tener un índice de fecha y hora para que nuestro marco de datos esté indexado en la marca de tiempo.


```python
# Convertir el índice del marco de datos en un índice de fecha y hora

df['datetime'] = pd.to_datetime(df['date'])
df = df.set_index('datetime')
df.drop(['date'], axis=1, inplace=True)
df.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01 00:00:00</th>
      <td>23</td>
    </tr>
    <tr>
      <th>2018-01-01 01:00:00</th>
      <td>73</td>
    </tr>
    <tr>
      <th>2018-01-01 02:00:00</th>
      <td>3</td>
    </tr>
    <tr>
      <th>2018-01-01 03:00:00</th>
      <td>10</td>
    </tr>
    <tr>
      <th>2018-01-01 04:00:00</th>
      <td>44</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Ejemplo de cómo filtrar datos con solo el día 2

df[df.index.day == 2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-02 00:00:00</th>
      <td>27</td>
    </tr>
    <tr>
      <th>2018-01-02 01:00:00</th>
      <td>80</td>
    </tr>
    <tr>
      <th>2018-01-02 02:00:00</th>
      <td>9</td>
    </tr>
    <tr>
      <th>2018-01-02 03:00:00</th>
      <td>67</td>
    </tr>
    <tr>
      <th>2018-01-02 04:00:00</th>
      <td>26</td>
    </tr>
    <tr>
      <th>2018-01-02 05:00:00</th>
      <td>13</td>
    </tr>
    <tr>
      <th>2018-01-02 06:00:00</th>
      <td>22</td>
    </tr>
    <tr>
      <th>2018-01-02 07:00:00</th>
      <td>68</td>
    </tr>
    <tr>
      <th>2018-01-02 08:00:00</th>
      <td>48</td>
    </tr>
    <tr>
      <th>2018-01-02 09:00:00</th>
      <td>75</td>
    </tr>
    <tr>
      <th>2018-01-02 10:00:00</th>
      <td>93</td>
    </tr>
    <tr>
      <th>2018-01-02 11:00:00</th>
      <td>80</td>
    </tr>
    <tr>
      <th>2018-01-02 12:00:00</th>
      <td>36</td>
    </tr>
    <tr>
      <th>2018-01-02 13:00:00</th>
      <td>67</td>
    </tr>
    <tr>
      <th>2018-01-02 14:00:00</th>
      <td>54</td>
    </tr>
    <tr>
      <th>2018-01-02 15:00:00</th>
      <td>66</td>
    </tr>
    <tr>
      <th>2018-01-02 16:00:00</th>
      <td>54</td>
    </tr>
    <tr>
      <th>2018-01-02 17:00:00</th>
      <td>9</td>
    </tr>
    <tr>
      <th>2018-01-02 18:00:00</th>
      <td>38</td>
    </tr>
    <tr>
      <th>2018-01-02 19:00:00</th>
      <td>36</td>
    </tr>
    <tr>
      <th>2018-01-02 20:00:00</th>
      <td>19</td>
    </tr>
    <tr>
      <th>2018-01-02 21:00:00</th>
      <td>19</td>
    </tr>
    <tr>
      <th>2018-01-02 22:00:00</th>
      <td>30</td>
    </tr>
    <tr>
      <th>2018-01-02 23:00:00</th>
      <td>88</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Filtrando datos entre dos fechas
df['2018-01-04':'2018-01-06']
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-04 00:00:00</th>
      <td>61</td>
    </tr>
    <tr>
      <th>2018-01-04 01:00:00</th>
      <td>62</td>
    </tr>
    <tr>
      <th>2018-01-04 02:00:00</th>
      <td>24</td>
    </tr>
    <tr>
      <th>2018-01-04 03:00:00</th>
      <td>76</td>
    </tr>
    <tr>
      <th>2018-01-04 04:00:00</th>
      <td>41</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2018-01-06 19:00:00</th>
      <td>44</td>
    </tr>
    <tr>
      <th>2018-01-06 20:00:00</th>
      <td>14</td>
    </tr>
    <tr>
      <th>2018-01-06 21:00:00</th>
      <td>45</td>
    </tr>
    <tr>
      <th>2018-01-06 22:00:00</th>
      <td>45</td>
    </tr>
    <tr>
      <th>2018-01-06 23:00:00</th>
      <td>63</td>
    </tr>
  </tbody>
</table>
<p>72 rows × 1 columns</p>
</div>



Podríamos tomar el mínimo, el máximo, el promedio, la suma... de los datos con una frecuencia diaria en lugar de una frecuencia horaria según el ejemplo a continuación, donde calculamos el promedio diario de los datos:


```python
df.resample('D').mean()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01</th>
      <td>45.083333</td>
    </tr>
    <tr>
      <th>2018-01-02</th>
      <td>46.833333</td>
    </tr>
    <tr>
      <th>2018-01-03</th>
      <td>48.333333</td>
    </tr>
    <tr>
      <th>2018-01-04</th>
      <td>52.625000</td>
    </tr>
    <tr>
      <th>2018-01-05</th>
      <td>47.791667</td>
    </tr>
    <tr>
      <th>2018-01-06</th>
      <td>37.333333</td>
    </tr>
    <tr>
      <th>2018-01-07</th>
      <td>45.666667</td>
    </tr>
    <tr>
      <th>2018-01-08</th>
      <td>68.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Ejemplo de cómo obtener la suma de los últimos tres valores

df['rolling_sum'] = df.rolling(3).sum()
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
      <th>rolling_sum</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01 00:00:00</th>
      <td>23</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-01-01 01:00:00</th>
      <td>73</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2018-01-01 02:00:00</th>
      <td>3</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>2018-01-01 03:00:00</th>
      <td>10</td>
      <td>86.0</td>
    </tr>
    <tr>
      <th>2018-01-01 04:00:00</th>
      <td>44</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>2018-01-01 05:00:00</th>
      <td>9</td>
      <td>63.0</td>
    </tr>
    <tr>
      <th>2018-01-01 06:00:00</th>
      <td>73</td>
      <td>126.0</td>
    </tr>
    <tr>
      <th>2018-01-01 07:00:00</th>
      <td>83</td>
      <td>165.0</td>
    </tr>
    <tr>
      <th>2018-01-01 08:00:00</th>
      <td>4</td>
      <td>160.0</td>
    </tr>
    <tr>
      <th>2018-01-01 09:00:00</th>
      <td>95</td>
      <td>182.0</td>
    </tr>
  </tbody>
</table>
</div>



Solo comienza a tener valores válidos cuando hay tres períodos sobre los cuales mirar hacia atrás.

La siguiente es una buena oportunidad para ver cómo podemos reenviar o rellenar datos cuando trabajamos con valores de datos faltantes.


```python
df['rolling_sum_backfilled'] = df['rolling_sum'].fillna(method='backfill')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
      <th>rolling_sum</th>
      <th>rolling_sum_backfilled</th>
    </tr>
    <tr>
      <th>datetime</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01 00:00:00</th>
      <td>23</td>
      <td>NaN</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>2018-01-01 01:00:00</th>
      <td>73</td>
      <td>NaN</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>2018-01-01 02:00:00</th>
      <td>3</td>
      <td>99.0</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>2018-01-01 03:00:00</th>
      <td>10</td>
      <td>86.0</td>
      <td>86.0</td>
    </tr>
    <tr>
      <th>2018-01-01 04:00:00</th>
      <td>44</td>
      <td>57.0</td>
      <td>57.0</td>
    </tr>
  </tbody>
</table>
</div>



Suele ser útil poder completar los datos que faltan con valores realistas, como el promedio de un período de tiempo. Pero recuerda siempre que, si estás trabajando con un problema de serie temporal y deseas que tus datos sean realistas, no debes realizar un relleno de tus datos.

Al trabajar con datos de series de tiempo, es posible que encuentres valores de tiempo que están en tiempo de Unix. El tiempo de Unix, también llamado tiempo de época, es el número de segundos que han transcurrido desde las 00:00:00 hora universal coordinada (UTC) del jueves 1 de enero de 1970.

**¿Cómo convertir el tiempo de época a tiempo real?**



```python
epoch_t = 1529272655
real_t = pd.to_datetime(epoch_t, unit='s')
real_t
```




    Timestamp('2018-06-17 21:57:35')




```python
# Ahora, vamos a convertirlo a la hora del Pacífico.

real_t.tz_localize('UTC').tz_convert('US/Pacific')
```




    Timestamp('2018-06-17 14:57:35-0700', tz='US/Pacific')



## Caso de uso:

En el siguiente ejemplo, solo tomaremos datos de una serie temporal univariada. Eso significa que solo estamos considerando la relación entre el valor del eje y y los puntos de tiempo del eje x. No estamos considerando factores externos que puedan estar afectando la serie temporal.

Un error común que cometen los principiantes es que inmediatamente comienzan a aplicar modelos de pronóstico ARIMA a datos que tienen muchos factores externos.


```python
import pandas as pd
data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/machine-learning-content/master/assets/electric_production.csv', index_col=0)
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>IPG2211A2N</th>
    </tr>
    <tr>
      <th>DATE</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1939-01-01</th>
      <td>3.3335</td>
    </tr>
    <tr>
      <th>1939-02-01</th>
      <td>3.3590</td>
    </tr>
    <tr>
      <th>1939-03-01</th>
      <td>3.4353</td>
    </tr>
    <tr>
      <th>1939-04-01</th>
      <td>3.4607</td>
    </tr>
    <tr>
      <th>1939-05-01</th>
      <td>3.4607</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Energy Production</th>
    </tr>
    <tr>
      <th>DATE</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2022-02-01</th>
      <td>114.3064</td>
    </tr>
    <tr>
      <th>2022-03-01</th>
      <td>102.7846</td>
    </tr>
    <tr>
      <th>2022-04-01</th>
      <td>91.4573</td>
    </tr>
    <tr>
      <th>2022-05-01</th>
      <td>95.5598</td>
    </tr>
    <tr>
      <th>2022-06-01</th>
      <td>104.3661</td>
    </tr>
  </tbody>
</table>
</div>



Nuestro índice es en realidad solo una lista de cadenas que parecen una fecha, por lo que debemos ajustarlas para que sean marcas de tiempo, de esa manera nuestro análisis de pronóstico podrá interpretar estos valores.


```python
data.index = pd.to_datetime(data.index)
```

También cambiemos el nombre de nuestra columna IPG2211A2N con un nombre más amigable.


```python
data.columns = ['Energy Production']
```


```python
pip install chart_studio cufflinks statsmodels
```

    Collecting plotly
      Downloading plotly-5.9.0-py2.py3-none-any.whl (15.2 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m15.2/15.2 MB[0m [31m79.4 MB/s[0m eta [36m0:00:00[0mm eta [36m0:00:01[0m0:01[0m
    [?25hCollecting tenacity>=6.2.0
      Downloading tenacity-8.0.1-py3-none-any.whl (24 kB)
    Installing collected packages: tenacity, plotly
    Successfully installed plotly-5.9.0 tenacity-8.0.1
    [33mWARNING: There was an error checking the latest version of pip.[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.



```python
import cufflinks as cf
import plotly.offline as py
import matplotlib.pyplot as plt
```


```python
data.plot(title="Energy Production Jan 1985--Jan 2018", figsize=(15,6))
```




    <AxesSubplot:title={'center':'Energy Production Jan 1985--Jan 2018'}, xlabel='DATE'>




    
![png](exploring-time-series.es_files/exploring-time-series.es_28_1.png)
    


Parece que la tendencia en estos días anteriores está aumentando ligeramente a un ritmo más alto que solo lineal. Experimentar con métodos aditivos versus multiplicativos se hace fácil en solo unas pocas líneas de código con statsmodels:


```python
from pylab import rcParams
rcParams['figure.figsize'] = 11, 9

import statsmodels.api as sm
decomposition = sm.tsa.seasonal_decompose(data, model='multiplicative')
fig = decomposition.plot()
plt.show()

```


    
![png](exploring-time-series.es_files/exploring-time-series.es_30_0.png)
    


Podemos ver claramente el componente estacional de los datos, y también podemos ver la tendencia ascendente separada de los datos. Tiene sentido utilizar un modelo ARIMA estacional. Para hacer esto, necesitaremos elegir valores p,d,q para el ARIMA y valores P,D,Q para el componente estacional.


```python
pip install pmdarima
```

    Collecting pmdarima
      Downloading pmdarima-1.8.5-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.manylinux_2_24_x86_64.whl (1.5 MB)
    [2K     [38;2;114;156;31m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.5/1.5 MB[0m [31m19.0 MB/s[0m eta [36m0:00:00[0mm eta [36m0:00:01[0m0:01[0m
    [?25hRequirement already satisfied: setuptools!=50.0.0,>=38.6.0 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from pmdarima) (62.3.2)
    Requirement already satisfied: numpy>=1.19.3 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from pmdarima) (1.23.1)
    Requirement already satisfied: statsmodels!=0.12.0,>=0.11 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from pmdarima) (0.13.2)
    Requirement already satisfied: scikit-learn>=0.22 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from pmdarima) (1.1.1)
    Requirement already satisfied: pandas>=0.19 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from pmdarima) (1.4.3)
    Requirement already satisfied: joblib>=0.11 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from pmdarima) (1.1.0)
    Requirement already satisfied: scipy>=1.3.2 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from pmdarima) (1.8.1)
    Requirement already satisfied: urllib3 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from pmdarima) (1.26.9)
    Requirement already satisfied: Cython!=0.29.18,>=0.29 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from pmdarima) (0.29.30)
    Requirement already satisfied: pytz>=2020.1 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from pandas>=0.19->pmdarima) (2022.1)
    Requirement already satisfied: python-dateutil>=2.8.1 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from pandas>=0.19->pmdarima) (2.8.2)
    Requirement already satisfied: threadpoolctl>=2.0.0 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from scikit-learn>=0.22->pmdarima) (3.1.0)
    Requirement already satisfied: patsy>=0.5.2 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from statsmodels!=0.12.0,>=0.11->pmdarima) (0.5.2)
    Requirement already satisfied: packaging>=21.3 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from statsmodels!=0.12.0,>=0.11->pmdarima) (21.3)
    Requirement already satisfied: pyparsing!=3.0.5,>=2.0.2 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from packaging>=21.3->statsmodels!=0.12.0,>=0.11->pmdarima) (3.0.9)
    Requirement already satisfied: six in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from patsy>=0.5.2->statsmodels!=0.12.0,>=0.11->pmdarima) (1.16.0)
    Installing collected packages: pmdarima
    Successfully installed pmdarima-1.8.5
    [33mWARNING: There was an error checking the latest version of pip.[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.


La biblioteca pyramid-arima para Python nos permite realizar rápidamente una búsqueda en cuadrícula e incluso crea un objeto modelo que puede ajustarse a los datos de entrenamiento.

Esta biblioteca contiene una función auto_arima que nos permite establecer un rango de valores p,d,q,P,D y Q y luego ajustar modelos para todas las combinaciones posibles. Entonces el modelo mantendrá la combinación que reportó el mejor valor de AIC.


```python
from pmdarima.arima import auto_arima
stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
print(stepwise_model.aic())
```

    Performing stepwise search to minimize aic
     ARIMA(1,1,1)(0,1,1)[12]             : AIC=4023.136, Time=1.80 sec
     ARIMA(0,1,0)(0,1,0)[12]             : AIC=4583.420, Time=0.11 sec
     ARIMA(1,1,0)(1,1,0)[12]             : AIC=4382.760, Time=0.52 sec
     ARIMA(0,1,1)(0,1,1)[12]             : AIC=4129.116, Time=1.20 sec
     ARIMA(1,1,1)(0,1,0)[12]             : AIC=4340.994, Time=0.34 sec
     ARIMA(1,1,1)(1,1,1)[12]             : AIC=4020.582, Time=2.72 sec
     ARIMA(1,1,1)(1,1,0)[12]             : AIC=4228.529, Time=1.62 sec
     ARIMA(1,1,1)(2,1,1)[12]             : AIC=3982.381, Time=5.49 sec
     ARIMA(1,1,1)(2,1,0)[12]             : AIC=4086.153, Time=4.49 sec
     ARIMA(1,1,1)(2,1,2)[12]             : AIC=3969.042, Time=17.31 sec
     ARIMA(1,1,1)(1,1,2)[12]             : AIC=4009.967, Time=9.90 sec
     ARIMA(0,1,1)(2,1,2)[12]             : AIC=4056.170, Time=13.98 sec
     ARIMA(1,1,0)(2,1,2)[12]             : AIC=4113.543, Time=11.19 sec
     ARIMA(2,1,1)(2,1,2)[12]             : AIC=3970.985, Time=22.43 sec
     ARIMA(1,1,2)(2,1,2)[12]             : AIC=3970.979, Time=15.87 sec
     ARIMA(0,1,0)(2,1,2)[12]             : AIC=4183.409, Time=10.02 sec
     ARIMA(0,1,2)(2,1,2)[12]             : AIC=3992.791, Time=19.49 sec
     ARIMA(2,1,0)(2,1,2)[12]             : AIC=4070.929, Time=11.41 sec
     ARIMA(2,1,2)(2,1,2)[12]             : AIC=3970.372, Time=29.53 sec
     ARIMA(1,1,1)(2,1,2)[12] intercept   : AIC=3971.031, Time=41.57 sec
    
    Best model:  ARIMA(1,1,1)(2,1,2)[12]          
    Total fit time: 221.072 seconds
    3969.042177805799



```python
# Train test split

train = data.loc['1985-01-01':'2017-12-01']
test = data.loc['2018-01-01':]
```


```python
# Train the model

stepwise_model.fit(train)
```




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre> ARIMA(1,1,1)(2,1,2)[12]          </pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">ARIMA</label><div class="sk-toggleable__content"><pre> ARIMA(1,1,1)(2,1,2)[12]          </pre></div></div></div></div></div>



Al ajustar modelos ARIMA estacionales (y cualquier otro modelo para el caso), es importante ejecutar diagnósticos del modelo para asegurarse de que no se haya violado ninguna de las suposiciones hechas por el modelo. El objeto plot_diagnostics nos permite generar rápidamente diagnósticos de modelos e investigar cualquier comportamiento inusual.


```python
stepwise_model.fit(train).plot_diagnostics(figsize=(15, 12))
plt.show()
```


    
![png](exploring-time-series.es_files/exploring-time-series.es_38_0.png)
    


Esto es para asegurar que los residuos de nuestro modelo no estén correlacionados y se distribuyan normalmente con media cero. Si el modelo ARIMA estacional no satisface estas propiedades, es una buena indicación de que se puede mejorar aún más.

En la gráfica superior derecha, vemos que la línea KDE naranja sigue de cerca a la línea N(0,1) (donde N(0,1) es la notación estándar para una distribución normal con media 0 y desviación estándar de 1) . Esta es una buena indicación de que los residuos se distribuyen normalmente.

El gráfico qq en la parte inferior izquierda muestra que la distribución ordenada de residuos (puntos azules) sigue la tendencia lineal de las muestras tomadas de una distribución normal estándar con N(0, 1). Nuevamente, esta es una fuerte indicación de que los residuos se distribuyen normalmente.

Ahora que el modelo se ha ajustado a los datos de entrenamiento, podemos pronosticar el futuro. Recuerda que nuestro conjunto de datos de prueba es desde el 01-01-2018 hasta el 01-06-2022, por lo que tenemos 54 períodos. Ese es el valor que usaremos para nuestra llamada al método .predict():


```python
future_forecast = stepwise_model.predict(n_periods=54)
```

Creemos un marco de datos que contenga nuestro pronóstico futuro y luego concatenémoslo con los datos originales.


```python
future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])
pd.concat([test,future_forecast],axis=1).plot()

```




    <AxesSubplot:xlabel='DATE'>




    
![png](exploring-time-series.es_files/exploring-time-series.es_43_1.png)
    


Ahora, obtengamos una imagen más amplia del contexto de nuestra predicción en el conjunto de prueba.


```python
pd.concat([data,future_forecast],axis=1).plot()
```




    <AxesSubplot:xlabel='DATE'>




    
![png](exploring-time-series.es_files/exploring-time-series.es_45_1.png)
    


¡Ahora es tu turno de adaptar nuestro modelo a todo nuestro conjunto de datos y luego pronosticar el futuro real!

Fuente: 

https://towardsdatascience.com/how-to-forecast-time-series-with-multiple-seasonalities-23c77152347e

https://medium.com/@josemarcialportilla/using-python-and-auto-arima-to-forecast-seasonal-time-series-90877adff03c

https://towardsdatascience.com/time-series-analysis-in-python-an-introduction-70d5a5b1d52a

https://github.com/WillKoehrsen/Data-Analysis/tree/master/additive_models

https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b

https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea

https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3
