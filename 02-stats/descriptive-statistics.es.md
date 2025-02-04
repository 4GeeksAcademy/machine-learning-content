---
description: >-
  Master the essentials of descriptive statistics in Python! Learn about
  measures of central tendency, dispersion, and data visualization. Discover
  more now!
---
## Estadística descriptiva

La **estadística descriptiva** (*descriptive statistics*) es una rama de la estadística que se encarga de recolectar, analizar, interpretar y presentar datos de manera organizada y efectiva. Su principal objetivo es proporcionar resúmenes simples y comprensibles acerca de las características principales de un conjunto de datos, sin llegar a hacer inferencias o predicciones sobre una población más amplia.

### Guía de la estadística descriptiva en Python

#### Medidas de tendencia central

Las **medidas de tendencia central** son valores numéricos que describen cómo se centralizan o agrupan los datos en un conjunto. Son esenciales en estadística y análisis de datos porque nos proporcionan un resumen de la información, permitiéndonos comprender rápidamente las características generales de una distribución de datos sin tener que observar cada valor individualmente.

Valor central de un conjunto de datos numéricos. 

```py
import statistics as stats

data = [10, 20, -15, 0, 50, 10, 5, 100]
mean = stats.mean(data)
print(f"Media: {mean}")
```

**Mediana** (*median*)

Valor medio cuando los datos están ordenados.

```py
median = stats.median(data)
print(f"Mediana: {median}")
```

**Moda** (*mode*)

Valor que ocurre con mayor frecuencia.

```py
mode = stats.mode(data)
print(f"Moda: {mode}")
```

Estas medidas son fundamentales para describir y analizar distribuciones de datos.

#### Medidas de dispersión

Las **medidas de dispersión** son valores numéricos que describen cómo de variados son los datos en un conjunto. Mientras las medidas de tendencia central nos indican dónde están "centrados" los datos, las medidas de dispersión nos muestran cuánto se "extienden" o "varían" esos datos alrededor de ese centro.

**Rango** (*range*)

Es la diferencia entre el valor máximo y el valor mínimo de un conjunto de datos. 

```py
range_ = max(data) - min(data)
print(f"Rango: {range_}")
```

**Varianza y desviación estándar** (*variance and standard deviation*)

Ambas métricas miden lo mismo. Indican cómo de lejos están, en promedio, los valores con respecto a la media. No obstante, la desviación típica es una medida que se utiliza para poder trabajar con unidades de medida iniciales, mientras que la varianza, aunque a priori nos pueda parecer un cálculo innecesario, se calcula para poder obtener otros parámetros.

```py
variance = stats.variance(data)
std = stats.stdev(data)
print(f"Varianza: {variance}")
print(f"Desviación estándar: {std}")
```

#### Medidas de posición

Las **medidas de posición** son estadísticas que nos indican la ubicación o posición de un valor específico dentro de un conjunto de datos.

**Percentiles y cuantiles** (*percentiles* and *quantiles*)

Son medidas que tratan sobre cómo se puede dividir un conjunto de datos en partes específicas. Estas medidas se utilizan para entender y describir la distribución de datos.

- **Percentil**: Divide un conjunto de datos en 100 partes iguales. El k-ésimo percentil indica el valor bajo el cual cae el k% de las observaciones.
- **Cuantil**: Divide un conjunto de datos en partes iguales, dependiendo del tipo. Los cuartiles dividen los datos en cuatro partes, los quintiles en cinco, etcétera.

#### Medidas de forma

Las **medidas de forma** describen cómo se distribuyen los valores en un conjunto de datos en relación con las medidas de tendencia central. Específicamente, nos indican la naturaleza de la distribución, ya sea si es simétrica, sesgada o tiene colas pesadas, entre otros.

**Asimetría** (*skewness*)

Mide la falta de simetría en la distribución de datos. Una asimetría positiva indica que la mayoría de los datos están a la izquierda y hay unos pocos valores muy altos a la derecha. Una asimetría negativa indica que hay más valores bajos inusuales. Si es cercana a cero sugiere que los datos son bastante simétricos.

![skewness](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/skewness.png?raw=true)

```py
from scipy.stats import skew

skewness = skew(data)
```

**Curtosis** (*kurtosis*)

Mide la concentración de los datos en torno a la media. Se emplea para describir una distribución y forma parte de algunos contrastes de normalidad. Una curtosis positiva indica un pico más agudo en comparación con la distribución normal. Una curtosis negativa indica un pico más aplanado y unas colas más ligeras. Una curtosis cercana a cero es lo ideal, ya que sugiere una forma similar a la de la distribución normal.

![kurtosis](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/kurtosis.png?raw=true)

```PY
from scipy.stats import kurtosis

kurt = kurtosis(data)
```

#### Visualización de datos

En este apartado, visualizar los datos de los que disponemos es fundamental. Se suelen utilizar histogramas, gráficos de barras y diagramas de dispersión, dependiendo de la tipología de los datos.