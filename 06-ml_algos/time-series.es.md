## Series temporales

Una **serie temporal** (*time serie*) es una secuencia de datos ordenados en el tiempo, donde cada punto de datos está asociado a un instante específico. En otras palabras, es una colección de observaciones que se registran en intervalos regulares o irregulares a lo largo del tiempo. Estas observaciones pueden ser recopiladas en horas, días, meses o incluso años, dependiendo del contexto y la naturaleza del fenómeno que se está analizando.

En una serie temporal, el tiempo es la variable independiente, y las observaciones registradas a lo largo del tiempo son las variables dependientes. El objetivo principal de analizar una serie temporal es comprender y modelar el patrón o la estructura subyacente en los datos a lo largo del tiempo, con el fin de hacer predicciones futuras o extraer información relevante.

Las series temporales se encuentran comúnmente en una amplia variedad de campos: economía, finanzas, meteorología, ciencia e ingeniería, entre otras. Algunos ejemplos de series temporales incluyen datos de ventas diarias, precios de acciones, temperaturas diarias, tasas de crecimiento de población, niveles de producción, etcétera.

![temporal-serie-example](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/temporal-serie-example.png?raw=true)

Al analizar una serie temporal, es importante tener en cuenta que los datos están correlacionados en el tiempo. Esto significa que las observaciones en un momento dado pueden depender de las observaciones pasadas y, en algunos casos, también pueden verse afectadas por las observaciones futuras. Este patrón de correlación en el tiempo es lo que hace que el análisis de series temporales sea único y requiera técnicas específicas para su modelado y predicción.

El análisis de series temporales puede involucrar diversas técnicas, como métodos de suavizado, descomposición, modelos autoregresivos, modelos de media móvil, entre otros. Además, el uso de herramientas de visualización, como gráficos de líneas o gráficos de autocorrelación, es común para comprender mejor los patrones y tendencias en los datos a lo largo del tiempo.

### Análisis de una serie temporal

Cuando analizamos visualmente una serie temporal, hay varias cosas importantes que debemos buscar para comprender el comportamiento y los patrones de los datos a lo largo del tiempo. Estas son algunas de las principales cosas que se deben observar:

- **Tendencia**: Identificar si existe una tendencia general en la serie temporal, es decir, si los datos tienden a aumentar o disminuir a lo largo del tiempo. Una tendencia puede ser lineal (crecimiento o decrecimiento constante) o no lineal (crecimiento o decrecimiento acelerado o desacelerado).
- **Estacionalidad**: Observar si hay patrones estacionales o cíclicos en los datos, es decir, si hay comportamientos que se repiten en intervalos regulares, como diariamente, mensualmente o estacionalmente. Por ejemplo, puede haber un aumento en las ventas de juguetes durante las vacaciones de Navidad cada año.
- **Variabilidad**: Verificar si la variabilidad de los datos cambia con el tiempo. Pueden haber períodos de alta variabilidad seguidos de períodos de baja variabilidad. La variabilidad puede indicar momentos de inestabilidad o cambios en el comportamiento del fenómeno estudiado.
- **Puntos atípicos** (*Outliers*): Identificar si hay valores extremos o inusuales que difieran significativamente del patrón general de la serie. Los puntos atípicos pueden afectar la interpretación y los análisis posteriores.
- **Autocorrelación**: Comprobar si existe una correlación entre las observaciones pasadas y las actuales. La autocorrelación puede indicar dependencias en el tiempo y es fundamental para el modelado de series temporales.
- **Puntos de inflexión**: Buscar cambios bruscos o puntos de inflexión en la serie, donde la tendencia o el comportamiento del fenómeno cambian significativamente.

La visualización de una serie temporal puede realizarse mediante gráficos de líneas, gráficos de dispersión, histogramas, gráficos de autocorrelación y otras técnicas de visualización. Al identificar estas características en la serie temporal, podemos obtener información valiosa sobre el comportamiento y las relaciones temporales de los datos, lo que nos permitirá tomar decisiones informadas y realizar análisis más profundos o modelos predictivos.

### Predicción de una serie temporal

Para predecir y analizar series temporales, existen varios tipos de modelos que se pueden utilizar. Algunos de los modelos más comunes son:



# Series de tiempo

Los datos de series de tiempo son cualquier tipo de información que se presenta como una secuencia ordenada. Los sensores, el monitoreo, las previsiones meteorológicas, los precios de las acciones, los tipos de cambio, las métricas de rendimiento de las aplicaciones son solo algunos ejemplos del tipo de datos que incluyen series temporales.

El tiempo es el atributo central que distingue a las series temporales de otros tipos de datos. Los intervalos de tiempo aplicados para ensamblar los datos recopilados en un orden cronológico se denominan frecuencia de serie de tiempo. Entonces, tener el tiempo como uno de los ejes principales sería el principal indicador de que un conjunto de datos dado es una serie temporal. Sin embargo, el hecho de que una serie de eventos tenga un elemento de tiempo no la convierte automáticamente en una serie de tiempo.

Cuando se trabaja con datos de series de tiempo, hay algunas cosas a tener en cuenta:

- Estacionalidad: ¿muestran los datos un patrón periódico claro?

- Tendencia: ¿los datos siguen una pendiente constante hacia arriba o hacia abajo?

- Ruido: ¿hay puntos atípicos o valores faltantes que no son consistentes con el resto de los datos?

## El método ARIMA

Pronosticar es el proceso de hacer predicciones del futuro, basadas en datos pasados ​​y presentes. Uno de los métodos más comunes para esto es el modelo ARIMA, que significa Media Móvil Integrada AutoRegresiva.

Los modelos ARIMA se denotan con la notación ARIMA(p, d, q). Estos tres parámetros explican la estacionalidad, la tendencia y el ruido en los datos.

- **p** es el parámetro asociado al aspecto autorregresivo del modelo, que incorpora valores pasados. Por ejemplo, pronosticar que si llovió mucho en los últimos días, afirmas que es probable que también llueva mañana.

- **d** es el parámetro asociado con la parte integrada del modelo, que afecta la cantidad de diferenciación para aplicar a una serie de tiempo, por ejemplo, pronosticar que la cantidad de lluvia mañana será similar a la cantidad de lluvia hoy, si las cantidades diarias de lluvia han sido similares en los últimos días.

- **q** es el parámetro asociado con la parte del promedio móvil del modelo.

Una de las características más importantes de una serie de tiempo es la variación. Hay 4 categorías de variación: Fluctuaciones estacionales, cíclicas, de tendencia e irregulares. Las variaciones son patrones en los datos de la serie temporal. Se dice que una serie temporal que tiene patrones que se repiten durante períodos de tiempo fijos y conocidos tiene **estacionalidad**.

Si nuestro modelo tiene un componente estacional, utilizamos un modelo ARIMA estacional (SARIMA). En ese caso, tenemos otro conjunto de parámetros: P, D y Q que describen las mismas asociaciones que p, d y q, pero se corresponden con los componentes estacionales del modelo.

Pero, ¿cómo elegimos los valores p,d,q para el ARIMA y los valores P,D,Q para el componente estacional?

Podemos elegir estos valores estadísticamente, como mirar gráficos de correlación o usar su experiencia de dominio, pero también podemos realizar una búsqueda de cuadrícula sobre múltiples valores de p, d, q, P, D y Q usando algún tipo de criterio de rendimiento. 

La biblioteca pyramid-arima para Python nos permite realizar rápidamente esta búsqueda en cuadrícula. Esta biblioteca contiene una función auto_arima que nos permite establecer un rango de valores p,d,q,P,D y Q y luego ajustar modelos para todas las combinaciones posibles.

Al evaluar y comparar modelos estadísticos ajustados con diferentes parámetros, cada uno se puede clasificar entre sí en función de qué tan bien se ajuste a los datos o su capacidad para predecir con precisión puntos de datos futuros. Usaremos el valor **AIC** (Akaike Information Criterion), que se devuelve convenientemente con los modelos ARIMA ajustados mediante statsmodels. Después de realizar la búsqueda de cuadrícula, el modelo mantendrá la combinación que reportó el mejor valor de AIC. Cuando tengamos un modelo que podamos ajustar, necesitaremos entrenamiento y datos de prueba, por lo que "cortaremos" una parte de nuestros datos más recientes y los usaremos como conjunto de prueba.

**Decomposition**

En los datos de series temporales, se puede revelar mucho al visualizarlos. Aquí un ejemplo:

```py
y.plot(figsize=(15, 6))
plt.show()
```

![time-series](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/time-series.jpg?raw=true)

*imagen de www.digitalocean.com*

Algunos patrones distinguibles aparecen cuando graficamos los datos. La serie temporal anterior tiene un patrón de estacionalidad evidente, así como una tendencia general creciente. También podemos visualizar nuestros datos usando un método llamado descomposición de series de tiempo. La descomposición de series temporales es una tarea estadística que deconstruye una serie temporal en varios componentes, lo que nos permite descomponer nuestra serie temporal en tres componentes distintos: tendencia, estacionalidad y ruido.

Con *statsmodels* podremos ver los componentes de tendencia, estacionales y residuales de nuestros datos. Statsmodels proporciona la práctica función season_decompose para realizar la descomposición estacional.

Seasonal_decompose devuelve una cifra de tamaño relativamente pequeño, por lo que las dos primeras líneas de este fragmento de código garantizan que la cifra de salida sea lo suficientemente grande para que la visualicemos.

```py
from pylab import rcParams
rcParams['figure.figsize'] = 11, 9

decomposition = sm.tsa.seasonal_decompose(y, model='additive')
fig = decomposition.plot()
plt.show()
```

**Modelos aditivos y multiplicativos**

Podemos usar un modelo aditivo cuando parece que la tendencia es más lineal y los componentes de estacionalidad y tendencia parecen ser constantes en el tiempo, por ejemplo cuando cada año agregamos 100 unidades de algo. Un modelo multiplicativo es más apropiado cuando estamos aumentando (o disminuyendo) a una tasa no lineal, por ejemplo, cuando cada año duplicamos la cantidad.

Las tendencias pueden ser ascendentes o descendentes, y pueden ser lineales o no lineales. Es importante comprender su conjunto de datos para saber si ha pasado o no un período de tiempo significativo para identificar una tendencia real.

**Código de ejemplo:**

Usaremos una "búsqueda en cuadrícula" para explorar iterativamente diferentes combinaciones de parámetros. Para cada combinación de parámetros, ajustamos un nuevo modelo ARIMA estacional con la función SARIMAX() del módulo statsmodels y evaluamos su calidad general. Una vez que hayamos explorado todo el panorama de parámetros, nuestro conjunto óptimo de parámetros será el que produzca el mejor rendimiento para nuestros criterios de interés. Comencemos generando las diversas combinaciones de parámetros que deseamos evaluar:

```py
# Definir los parámetros p, d y q para que tomen cualquier valor entre 0 y 2
p = d = q = range(0, 2)

# Generar todas las combinaciones diferentes de tripletes p, q y q
pdq = list(itertools.product(p, d, q))

# Generar todas las combinaciones diferentes de trillizos p, q y q estacionales
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
```

El fragmento de código siguiente itera a través de combinaciones de parámetros y utiliza la función SARIMAX de statsmodels para ajustarse al modelo ARIMA estacional correspondiente. Aquí, el argumento de orden especifica los parámetros (p, d, q), mientras que el argumento de seasonal_order especifica el componente estacional (P, D, Q, S) del modelo ARIMA estacional. Después de ajustar cada modelo SARIMAX(), el código imprime su respectivo puntaje AIC.

```py
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
```

Ajusta de un modelo ARIMA: mediante la búsqueda en cuadrícula, hemos identificado el conjunto de parámetros que produce el modelo que mejor se ajusta a nuestros datos de series temporales. Podemos proceder a analizar este modelo en particular con más profundidad.

We’ll start by plugging the optimal parameter values into a new SARIMAX model:

```py
mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
```


## Prophet method

The Core Data Science team at Facebook published a new method called Prophet, which enables data analysts and developers alike to perform forecasting at scale in Python 3.

In order to compute its forecasts, the fbprophet library relies on the STAN programming language. Before installing fbprophet, we need to make sure that the pystan Python wrapper to STAN is installed. This can be done using:

```py
pip install pystan
```

Once installed, we are ready to install fbprophet too and import it as follows:

```py
from fbprophet import Prophet
```

**How to use the Prophet library to predict future values of our time series?**

To begin, we must instantiate a new Prophet object. Prophet enables us to specify a number of arguments. For example, we can specify the desired range of our uncertainty interval by setting the interval_width parameter.

```py
# set the uncertainty interval to 95% (the Prophet default is 80%)
my_model = Prophet(interval_width=0.95)
```

Now that our Prophet model has been initialized, we can call its fit method with our DataFrame as input. The model fitting should take no longer than a few seconds.

```py
my_model.fit(df)
```

Para obtener pronósticos de nuestra serie temporal, debemos proporcionar a Prophet un nuevo marco de datos que contenga una columna de fecha que contenga las fechas para las que queremos predicciones. Convenientemente, no tenemos que preocuparnos por crear manualmente este marco de datos, ya que Prophet proporciona la función auxiliar make_future_dataframe:

```py
# Especificamos claramente la frecuencia deseada de las marcas de tiempo (en este caso, MS es el comienzo del mes).
future_dates = my_model.make_future_dataframe(periods=36, freq='MS')
future_dates.tail()
```

El marco de datos de fechas futuras se usa luego como entrada para el método de predicción de nuestro modelo ajustado.

```py
forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
```

- ds: la marca de fecha del valor pronosticado.

- yhat: el valor pronosticado de nuestra métrica (en Estadística, yhat es una notación utilizada tradicionalmente para representar los valores pronosticados de un valor y).

- yhat_lower: el límite inferior de nuestros pronósticos.

- yhat_upper: el límite superior de nuestras pronósticos.

Es de esperar una variación en los valores de la salida presentada anteriormente, ya que Prophet se basa en los métodos de Monte Carlo de la cadena de Markov (MCMC) para generar sus pronósticos. MCMC es un proceso estocástico, por lo que los valores serán ligeramente diferentes cada vez.

Prophet también proporciona una función conveniente para trazar rápidamente los resultados de nuestros pronósticos:

```py
my_model.plot(forecast, uncertainty=True)
```

![time-series2](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/time-series2.jpg?raw=true)

*imagen de www.digitalocean.com*

## Algunas recomendaciones a la hora de trabajar con datos de series temporales:

- Comprueba si hay discrepancias en tus datos que puedan deberse a cambios de hora específicos de la región, como el horario de verano.

- Permite que otros que revisen tu código sepan en qué zona horaria se encuentran tus datos y piensa en convertirlos a UTC o a un valor estandarizado para mantener sus datos estandarizados.

- Los datos faltantes pueden ocurrir con frecuencia, así que asegúrate de documentar sus reglas de limpieza y piensa en no rellenar la información que no hubiera podido tener en el momento de una muestra.

- Recuerde que a medida que vuelve a muestrear tus datos o completa los valores que faltan, estás perdiendo cierta cantidad de información sobre tu conjunto de datos original. Realiza un seguimiento de todas sus transformaciones de datos y rastrea la causa raíz de tus problemas de datos.

- Cuando vuelvas a muestrear tus datos, el mejor método (media, mín., máx., suma, etc.) dependerá del tipo de datos que tengas y de cómo se muestrearon.


Fuente: 

https://towardsdatascience.com/how-to-forecast-time-series-with-multiple-seasonalities-23c77152347e

https://medium.com/@josemarcialportilla/using-python-and-auto-arima-to-forecast-seasonal-time-series-90877adff03c

https://towardsdatascience.com/basic-time-series-manipulation-with-pandas-4432afee64ea

https://towardsdatascience.com/an-end-to-end-project-on-time-series-analysis-and-forecasting-with-python-4835e6bf050b

https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-arima-in-python-3

https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-visualization-with-python-3

https://www.digitalocean.com/community/tutorials/a-guide-to-time-series-forecasting-with-prophet-in-python-3

https://www.clarify.io/learn/time-series-data
