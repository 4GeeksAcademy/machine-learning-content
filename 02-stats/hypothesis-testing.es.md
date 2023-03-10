# HYPOTHESIS TESTING

### Curvas de densidad

Tener histogramas de frecuencia es una buena manera de ver cómo se distribuyen nuestros datos. En en algún momento, podríamos querer ver qué porcentaje de nuestros datos cae en cada una de las categorías de la variable. En ese caso podemos construir un histograma de frecuencias relativas, que nos mostrará los porcentajes en lugar de las cantidades. Mientras haya mayor cantidad datos, más conveniente serán los contenedores pequeños en nuestro histograma para poder ver la distribución mucho más clara.

A medida que tengamos más y más datos, tal vez queramos ver contenedores aún más delgados que nos lleven a un punto en el que nos acerquemos a un número infinito de categorías y la mejor manera de verlo sería conectar la parte superior del barras que realmente obtendrá una curva. Esto se llama la curva de densidad.

![density_curve.jpg](../assets/density_curve.jpg)

**Probabilidad de la curva de densidad:**

Veamos un ejemplo de cómo calcular probabilidades a partir de curvas de densidad:

Tenemos un conjunto de datos con la altura de varias mujeres que normalmente van distribuídas con la media de 155cm y una desviación estándar de 15cm. La atura de una mujer aleatoriamente seleccionada de este conjunto será denotada como $W$.

Encuentra e interpreta $P(W > 170)$

![density_probability_problem.jpg](../assets/density_probability_problem.jpg)

### Teorema del límite central

Cuando dibujamos muestras de variables independientes aleatorias (extraído de cualquier distribución individual con una varianza finita), su muestra media tiende hacia la media de la población y su distribución se aproxima a una distribución normal a medida que aumenta el tamaño de la muestra, independientemente de la distribución de la que se extrajeron las variables aleatorias. Su varianza se acercará a la varianza de la población dividida por el tamaño de la muestra.


El teorema del límite central describe la forma de la distribución de medias muestrales como una distribución Gaussian o una distibución normal.

El teorma establece que a medida que el tamaño de la muestra aumenta, la distribución de la media entre múltiples muestras se aproximará a la distribución de Gaussian

El límite central del teorema no establece nada sobre una media de muestra única, en cambio, establece algo sobre la forma o la distribución de las medias muestrales.

Por ejemplo, digamos que tenemos un dado de 6 caras justo y balanceado. El resultado de tirar el dado tiene una distribución uniforme en $[1,2,3,4,5,6]$. El resultado promedio de una tirada de dado es $(1+2+3+4+5+6)/6 = 3.5$

Si tiramos el dado $10$ veces y promediamos los valores, entonces el parámetro resultante va a tener una distribución que comienza a verse similar a una distribución normal, nuevamente centrado en $3.5$.

Si tiramos el dado $100$ veces y promediamos los valores, entonces el parámetro resultante va a tener una distribución que se comporta aún mas similar a una distribución normal, nuevamente centrado en $3.5$, pero ahora con una varianza reducida.

![central_limit_theorem.jpg](../assets/central_limit_theorem.jpg)

### Métodos de muestreo

Vamos a imaginar que tenemos una pobación de estudiantes. Para esa población podemos calcular los parámetros. Por ejemplo, la edad media o la desviación estándar de las calificaciones. Estos son los valores de toda la población. Algunas veces podremos no saber los parámetros de la población, así que la manera de estimar ese parámetro es tomando una muestra.

El método de muestreo es el proceso de estudiar la población recopilando información y analizando esos datos. Se refiere a cómo los miembros de esa población son seleccionados para el estudio.

**Muestreo no representativo:**

El método de muestreo no representativo es una técnica en la cual el investigador selecciona la muestra basándose en un juicio subjetivo en lugar de la selección aleatoria. No todos los miembros de la población tienen un chance de participar en el estudio. 

-Muestreo por conveniencia = Elige muestras que sean más convenientes, como personas a las que se pueda acercar fácilmente.

-Muestreo Consecutivo = Elige una sola persona o grupo para para muestreo. Luego el investigador investiga por un período de tiempo para analizar el resultado y avazar a otro grupo si se es necesario.

-Muestreo intencional = Elige muestras para un propósito específico. Un ejemplo es centrarse en casos extremos. Esto puede ser útil pero es limitado porque no te permite hacer afirmaciones sobre toda la población.

-Muestreo de bola de nieve = En este método, las muestras tienen características que son difíciles de encontrar. Entonces, a cada miembro identificado de la población se le pide encontrar las demás unidades de muestreo. Esas unidades de muestreo también pertenecen a la misma población objetivo.

**Muestreo representativo:**

-Muestreo aleatorio simple = Elige muestras (psuedo) aleatorias. Cada artículo en la población tiene una probabilidad igual y probable de ser seleccionado en la muestra.

-Muestreo Sistemático = Elige muestras con un intervalo fijo. Por ejemplo, cada décima muestra  $(0, 10, 20, etc.)$. Es calculado dividiendo el total del tamaño de la población entre el tamaño de la población deseada.

-Muestreo estratificado = El total de la población es dividida en grupos más pequeños formados en base de algunas características en la población. Elige la misma cantidad de muestras de cada uno de los diferenbtes grupos (estratos) en la población.

-Muestreo por conglomerados = Se divide la población en grupos y se elige las muestras de esos grupos. Los grupos de personas están formados del conjunto de la población. El grupo tiene características significativas similares. Además, tienen la misma oportunidad de ser parte de la muestra. Este método utiliza un muestreo aleatorio simple para el conglomerado de la población.

Veamos algunos ejemplos de cómo obtener muestras.


```python
# Generar Distribución Normal
normal_dist = np.random.randn(10000)

# Ten en cuenta que tomamos muestras muy pequeñas solo para ilustrar los diferentes métodos de muestreo

# Muestras de conveniencia
convenience_samples = normal_dist[0:5]

# Muestras intencionales (seleccionar muestras para un propósito específico)
# En este ejemplo, elegimos los 5 valores más altos en nuestra distribución
purposive_samples = normal_dist.nlargest(n=5)

# Muestra aleatoria simple (pseudo)
random_samples = normal_dist.sample(5)

# Muestra sistemática (Cada valor 2000)
systematic_samples = normal_dist[normal_dist.index % 2000 == 0]

# Muestreo estratificado
# Obtendremos 1 estudiante de cada aula en el conjunto de datos
# Tenemos 8 aulas por lo que hace un total de 8 muestras
# Este es un código de ejemplo. No hay un conjunto de datos real.

df = pd.read_csv('https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/toy_dataset.csv')

strat_samples = []

for city in df['City'].unique():
    samp = df[df['City'] == city].sample(1)
    strat_samples.append(samp['Income'].item())
    
print('Stratified samples:\n\n{}\n'.format(strat_samples))

# Muestreo por conglomerados
# Haz grupos aleatorios de diez personas (aquí con reemplazo)
c1 = normal_dist.sample(10)
c2 = normal_dist.sample(10)
c3 = normal_dist.sample(10)
c4 = normal_dist.sample(10)
c5 = normal_dist.sample(10)


# Tomar muestra de cada grupo (con reemplazo)
clusters = [c1,c2,c3,c4,c5]
cluster_samples = []
for c in clusters:
    clus_samp = c.sample(1)
    cluster_samples.extend(clus_samp)
print('Cluster samples:\n\n{}'.format(cluster_samples))    
```

**Conjuntos de datos desequilibrados**

Un dataset (conjunto de datos) desequilibrado es un dataset donde las clases se distribuyen de manera desigual. Un dato desequilibrado puede crear problemas en la tarea de clasificación. Si usamos la precisión como una métrica de rendimiento, puede crear un gran problema. Digamos que nuestro modelo predice si una transacción bancaria fue una transacción fraudulenta o una transacción legal. Si el total de transacciones legales representó el 99,83 %, el uso de una métrica de precisión en el dataset de la tarjeta de crédito dará una precisión del 99,83 %, lo cual es excelente. ¿Sería un buen resultado? No.

Para un conjunto de datos desequilibrado, se deben usar otras métricas de rendimiento, como la puntuación AUC de recuperación de precisión, la puntuación F1, etc. Además, el modelo estará sesgado hacia la clase mayoritaria. Ya que la mayoría de las técnicas de Machine Learning están diseñadas para trabajar bien con el con un dataset balanceado, debemos crear datos balanceados de un dataset desequilibrado, pero primero debemos dividir el dataset en el entrenamiento y las pruebas porque el cambio es solo para fines de entrenamiento. 

Una manera común de lidiar con datasets desequilibrados es con **remuestreo.** Aquí hay dos posibles técnicas de remuestreo:

-Usa todas las muestras de nuestro evento que ocurre con mayor frecuencia y luego muestrea aleatoriamente nuestro evento que ocurre con menos frecuencia (con reemplazo) hasta que tengamos un dataset equilibrado.

-Usa todas las muestras de nuetroevento que ocurre con menos frecuencia y luego muestrea aleatoriamente nuestro evento que ocurre con más frecuencia (con o sin reemplazo) hasta que tengamos un dataset equilibrado.



### BIAS, MSE, SE

**Bias** es la diferencia entre el valor calculado de un parámetro y el verdadero valor del parámetro de población que se está estimando.
  
Podemos decir que es la medida de cual lejos la media muestral se desvía de la media poblacional. La media de la muestra también se denomina valor esperado.

Por ejemplo, si decidimos encuestar a los propietarios sobre el valor de sus casas y solo los propietarios más ricos responden, entonces el valor estimado de nuestra casa será **sesgado** ya que será mayor que el valor real para nuestra población.

Veamos cómo calcularíamos la media de la población, el valor esperado y el bias usando Python:


```python
# Tomar muestra
df_sample = df.sample(100)

# Calcula el valor esperado (EV), la media de la población y el el bias
ev = df_sample.mean()[0]
pop_mean = df.mean()[0]
bias = ev - pop_mean

print('Sample mean (Expected Value): ', ev)
print('Population mean: ', pop_mean)
print('Bias: ', bias)
```

**Error Cuadrático Medio (MSE)** es la fórmula para medir cuánto los estimadores se desvían de la verdadera distribución. Esto puede ser muy útil para evaluar modelos de regresión.


**Error estándar (SE)** es una fórmula para medir qué tan separada está la distribución de la media de la muestra.


```python
from math import sqrt
from scipy.stats import sem

Y = 100 # Verdadero valor
YH = 85 # Valor previsto

# MSE  
def MSE(Y, YH):
    return np.square(YH - Y).mean()

# RMSE 
def RMSE(Y, YH):
    return sqrt(np.square(YH - Y).mean())


print('MSE: ', MSE(Y, YH))
print('RMSE: ', RMSE(Y, YH))

#SE

norm_sample = normal_dist.sample(100)

print('Standard Error of normal sample: ', sem(norm_sample))
```

### Introducción a los niveles de confianza y los intervalos de confianza


Para hacer frente a la incertidumbre, podemos usar un intervalo estimádo. Provee un rango de valores que mejor describen a la población. Para desarrollar una estimación de intervalo, primero debemos conocer los niveles de confianza.

**Niveles de confianza**

Un nivel de confianza es la probabilidad de que la estimación de intervalo incluya el parámetro de la población (como a la media).

Un parámetro es una descripción numérica de una característica de la población.

![hypothesis_testing_standard_normal_distribution.jpg](../assets/hypothesis_testing_standard_normal_distribution.jpg)


Las medias muestrales seguirán la distribución de probabilidad normal para tamaños de muestra grandes $(n>=30)$

Para construir estimación de intervalo con un 90% de nivel de confianza.

El nivel de confianza corresponde 

El nivel de confianza corresponde a un puntaje z de la tabla normal estándar igual a $1.645$

![hypothesis_testing_confidence_interval.png](../assets/hypothesis_testing_confidence_interval.jpg)

**Intervalo de confianza**


Un intervalo de confianza es un rango de valores utilizado para estimar un parámetro de la población y es asociado con un nivel de confianza específico

El intervalo de confianza necesita ser descrito en el contexto de múltiples muestras.

Vamos a construir un intervalo de confianza alrededor de una media de muestra usando estas ecuaciones:
    
$x̄ ± z c$
    
Donde:
    
$x̄$ = la media muestral

$z$ = el puntaje z, que es el número de desviaciones estándar basadas en el nivel de confianza

$c$ = el error estándar de la media


Selecciona 10 muestras y construye intervalos de confianza del 90 % alrededor de cada una de las medias muestrales.

Teóricamente, 9 de los 10 intervalos contendrán la verdadera media poblacional que permanece desconocida.

![hypothesis_testing_confidence_interval_example.jpg](../assets/hypothesis_testing_confidence_interval_example.jpg)

No malinterpretes la definición de un intervalo de confianza:

Falso: 'Hay un 90% de probabilidad de que la verdadera media de la población esté dentro del intervalo'.

Verdadero: 'Hay un 90% de probabilidad de que cualquier intervalo de confianza dado de una muestra aleatoria, contenga la verdadera media de la población.'

Para resumir, cuando creamos un intervalo de confianza, es importante poder interpretar el significado del nivel de confianza que utilizamos y el intervalo que fue obtenido.

El nivel de confianza se refiere a la tasa de éxito a largo plazo del método, lo que significa con qué frecuencia este tipo de intervalo capturará el parámetro de interés.

Un intervalo de confianza específico da un rango de valores plausibles para el parámetro de interés.

### Pasos para formular una hipótesis


La prueba de hipótesis es un método estadístico que se utiliza para tomar decisiones estadísticas utilizando datos experimentales.

La prueba de hipótesis es básicamente una suposición que hacemos sobre el parámetro de la población.

Por ejemplo, cuando decimos que los chicos son más altos que las chicas. Esta suposición necesita alguna manera estadística para comprobarlo, necesitamos una conclusión matemática sea lo que sea que supongamos que es cierto.

Las hipótesis son afirmaciones y podemos usar estadísticas para probarlas o refutarlas. La prueba de hipótesis estructura los problemas para que podamos usar evidencia estadística para probar estas afirmaciones y verificar si la afirmación es válida o no.

**Pasos:**

1. Definición de hipótesis.

2. Comprobación de supuestos.

3. Establecer los niveles de significancia.

4. Selección de la prueba adecuada.

5. Llevar a cabo la prueba de hipótesis y calcular las estadísticas de prueba y el valor P correspondiente.

6. Compara el valor P con los niveles de significancia y luego decide aceptar o rechazar la hipótesis nula.

**1. Definición de nuestra hipótesis nula y alternativa**

Primero que nada, tenemos que entender a que pregunta científica le estamos buscando una repuesta, y debería de ser formulada en la forma de Hipótesis Nula ($H₀$) y la Hipótesis Alternativa ($H₁$ o $Hₐ$).

¿Qué es la hipótesis nula? ($H₀$)

La hipótesis nula es una declaración sobre un parámetro de población, la declaración asumida como cierta antes de que recopilemos los datos. Probamos la probabilidad de que esta afirmación sea cierta para decidir si aceptamos o rechazamos nuestra hipótesis alternativa. Puede considerarse como el 'control' del experimento y normalmente tiene algún signo igual ($>=, <=, =$)

¿Qué es la hipótesis alternativa? ($Hₐ$)

Es una afirmación que contradice directamente la hipótesis nula. Esto es lo que queremos demostrar que es cierto con nuestros datos recopilados. Se puede considerar como el 'experimento'. Suele tener el signo opuesto de la hipótesis nula.

**Dato:** Las estadísticas de muestra no son lo que debería estar involucrado en nuestra hipótesis. Nuestras hipótesis son afirmaciones sobre la población que queremos estudiar. Recuerda que $H₀$ y $Hₐ$ deben ser mutuamente excluyentes, y $Hₐ$ no debe contener igualdad.

**2. Comprobación de supuestos**

Para decidir si utilizar la versión paramétrica o no paramétrica de la prueba, debemos verificar si:

-Las observaciones en cada muestra son independientes e idénticamente distribuidas (IID).

-Las observaciones en cada muestra se distribuyen normalmente.

-Las observaciones en cada muestra tienen la misma varianza.

Lo siguiente que hacemos es establecer un umbral conocido como nivel de significancia.

**3. Nivel de significancia**



Se refiere al grado de significación en el que aceptamos o rechazamos la hipótesis nula. No es posible una precisión del 100 % para aceptar o rechazar una hipótesis, por lo que seleccionamos un nivel de significación que suele ser del 5 %.

El grado de significancia es definido como la probabilidad fija de eliminación incorrecta de la hipótesis nula cuando, de hecho, es verdadera. El nivel de significación $α$ (alfa) es la probabilidad de cometer un error tipo 1 (falso positivo). La probabilidad del intervalo de confianza es un complemento del nivel de significancia.

Un intervalo de confianza $(1-α)$ tiene un nivel de significación igual a $α$.

Teniendo un 95% de probabilidad de que cualquier intervalo de confianza dado contenga la verdadera media de la población, hay un 5% de probabilidad de que no lo haga.

Este 5% se conoce como el nivel de significancia.

**4. Selección de la prueba adecuada**

Existe una variedad de procedimientos estadísticos. La adecuada depende de la(s) pregunta(s) de investigación que estemos haciendo y del tipo de datos que recopilamos. Necesitamos analizar cuántos grupos se están comparando y si los datos están emparejados o no.

Para determinar si los datos coinciden, es necesario considerar si los datos se recopilaron de las mismas personas.

**Prueba T** : para 1 variable independiente con 2 categorías, y la variable objetivo.

Cuando deseamos saber si las medias de dos grupos en un aula de estudiantes (hombres y mujeres en una variable de género) difieren, es apropiada una prueba t. En orden de calcular una prueba t, necesitamos conocer la media, desviación estándar y número de individuos en cada uno de los dos grupos. Un ejemplo de una pregunta de investigación de prueba t es "¿Existe una diferencia significativa entre los puntajes de escritura de niños y niñas en el aula?" Una respuesta de muestra podría ser: “Los niños $(M=5.67, SD=.45)$ y las niñas $(M=5.76, SD=.50)$ puntúan de manera similar en escritura, $t(35)=.54$, $ p>.05$.” [Nota: $(35)$ son los grados de libertad para una prueba t. Es el número de individuos menos el número de grupos (siempre 2 grupos con una prueba t). En este ejemplo, había 37 personas y 2 grupos, por lo que los grados de libertad son $37-2=35$.] Recuerda, una prueba t solo puede comparar las medias de dos grupos de una variable independiente (por ejemplo, el género) en una única variable dependiente (por ejemplo, puntuación de escritura).

Una distribución t es más plana que una distribución normal. A medida que aumentan los grados de libertad, la forma de la distribución t se vuelve similar a una distribución normal. Con más de 30 grados de libertad (tamaño de muestra de 30 o más) las dos distribuciones son prácticamente idénticas.

Tipos de prueba T:

-Prueba t de dos muestras: si dos grupos independientes tienen medias diferentes.

-Prueba T pareada: si un grupo tiene diferentes medias en diferentes momentos.

-Prueba T de una muestra: media de un solo grupo frente a una media conocida.

Suposiciones acerca de los datos

1. Indepe­ndiente.

2. Normalmente distribuido.

3. Tienen una cantidad similar de varianza dentro de cada grupo que se compara.

Prueba t de una muestra: la prueba t de una muestra determina si la media de la muestra es estadísticamente diferente de una media poblacional conocida o hipotética. La prueba t de una muestra es una prueba paramétrica.

Imaginemos que tenemos un conjunto de datos pequeño con 10 edades y estamos comprobando si la edad promedio es 30 o no. Podríamos resolverlo de esta manera:


```py
    from scipy.stats import ttest_1samp
    import numpy as np
    ages = np.genfromtxt(“ages.csv”)
    print(ages)
    ages_mean = np.mean(ages)
    print(ages_mean)
    tset, pval = ttest_1samp(ages, 30)
    print(“p-values”,pval)
    if pval < 0.05:    # el valor alfa es 0.05 o 5%
      print("we are rejecting null hypothesis")
    else:
      print("we are accepting null hypothesis")
```

Prueba t de dos muestras: la prueba t de muestras independientes o prueba t de 2 muestras compara las medias de dos grupos independientes para determinar si existe evidencia estadística de que las medias de población asociadas son significativamente diferentes. La prueba t para muestras independientes es una prueba paramétrica. Esta prueba también se conoce como: Prueba t independiente.

Por ejemplo, si quisiéramos ver si existe alguna asociación entre la semana 1 y la semana 2, podríamos hacerlo con el siguiente código:

```py
    from scipy.stats import ttest_ind
    import numpy as np

    week1 = np.genfromtxt("week1.csv",  delimiter=",")
    week2 = np.genfromtxt("week2.csv",  delimiter=",")
    print(week1)
    print("week2 data :-\n")
    print(week2)
    week1_mean = np.mean(week1)
    week2_mean = np.mean(week2)
    print("week1 mean value:",week1_mean)
    print("week2 mean value:",week2_mean)
    week1_std = np.std(week1)
    week2_std = np.std(week2)
    print("week1 std value:",week1_std)
    print("week2 std value:",week2_std)
    ttest,pval = ttest_ind(week1,week2)
    print("p-value",pval)
    if pval <0.05:
      print("we reject null hypothesis")
    else:
      print("we accept null hypothesis")
  ```

Prueba t de muestra pareada: la prueba t de muestra pareada también se denomina prueba t de muestra dependiente. Es una prueba univariada que examina una diferencia significativa entre 2 variables relacionadas. Un ejemplo de esto es si fueses recopilar la presión arterial de una persona antes y después de algún tratamiento, condición o momento.

$H0$ : la diferencia de medias entre dos muestras es 0.

$H1$: diferencia media entre dos muestras no es 0.

Intentemos ejecutar el siguiente código de ejemplo:


```python
import pandas as pd
from scipy import stats

df = pd.read_csv('https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/blood_pressure.csv')

df[['bp_before','bp_after']].describe()
ttest,pval = stats.ttest_rel(df['bp_before'], df['bp_after'])
print(pval)

if pval<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")
```

    0.0011297914644840823
    reject null hypothesis



**PRUEBA F (Análisis ANOVA)**

La prueba t funciona bien cuando se trata de dos grupos, pero a veces queremos comparar más de dos al mismo tiempo. Por ejemplo, si quisiéramos probar si la edad de los votantes difiere en función de alguna variable categórica como la raza, tenemos que comparar las medias de cada nivel o agrupar la variable. Podríamos realizar una prueba t separada para cada par de grupos, pero cuando realizas muchas pruebas, aumenta las posibilidades de falsos positivos.

El análisis de varianza o ANOVA es una prueba de inferencia estadística que te permite comparar varios grupos al mismo tiempo. A diferencia de las distribuciones t, la distribución F no tiene valores negativos porque la variabilidad entre y dentro del grupo siempre es positiva debido al cuadrado de cada desviación.

Anova unidireccional: una variable independiente con más de 2 categorías y la variable objetivo.

Si tenemos una variable independiente (con tres o más categorías) y una variable dependiente, hacemos un ANOVA de una vía.Una pregunta de investigación de muestra es: "¿Los médicos, maestros e ingenieros difieren en su opinión sobre un aumento de impuestos?" Una respuesta de muestra es: “Los maestros $(M=3.56, SD=.56)$ tienen menos probabilidades de favorecer un aumento de impuestos que los médicos $(M=5.67, SD=.60)$ o los ingenieros $(M=5.34, SD =.45)$, $F(2,120)=5.67$, $p<.05$.” [Nota: Los $(2,207)$ son los grados de libertad para un ANOVA.El primer número es el número de grupos menos 1. Como teníamos tres profesiones es 2, porque 3-1=2. El segundo número es el número total de individuos menos el número de grupos. Porque teníamos 210 sujetos y 3 grupos, son 207 (210 - 3)].

Ejemplo de código: hay 3 categorías diferentes de plantas y su peso y es necesario verificar si los 3 grupos son similares o no.


```python
df_anova = pd.read_csv('https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/PlantGrowth.csv')
df_anova = df_anova[['weight','group']]
grps = pd.unique(df_anova.group.values)
d_data = {grp:df_anova['weight'][df_anova.group == grp] for grp in grps}
 
F, p = stats.f_oneway(d_data['ctrl'], d_data['trt1'], d_data['trt2'])
print("p-value for significance is: ", p)
if p<0.05:
    print("reject null hypothesis")
else:
    print("accept null hypothesis")
```

    p-value for significance is:  0.0159099583256229
    reject null hypothesis


Anova bidireccional: más de una variable independiente con dos o más categorías cada una y la variable objetivo.

Un ANOVA de dos vías tiene tres preguntas de investigación, una para cada una de las dos variables independientes y otra para la interacción de las dos variables independientes. La prueba F de 2 vías no indica qué variable es dominante. si necesitamos verificar la importancia individual, entonces se deben realizar pruebas post-hoc.

Ejemplos de preguntas de investigación para un ANOVA de dos vías:
¿Difieren los maestros, médicos e ingenieros en su opinión sobre un aumento de impuestos?
¿Los hombres y las mujeres difieren en su opinión sobre un aumento de impuestos?
¿Existe una interacción entre género y profesión en cuanto a las opiniones sobre un aumento de impuestos?

Un ANOVA de dos vías tiene tres hipótesis nulas, tres hipótesis alternativas y tres respuestas a la pregunta de investigación.

Suposiciones sobre los datos:

1. Cada dato $y$ se distribuye normalmente.

2. La varianza de cada grupo de tratamiento es la misma.

3. Todas las observaciones son independientes.

Ejemplo de código: echemos un vistazo al rendimiento de cultivo medio general (el rendimiento de cultivo medio no por ningún subgrupo), así como el rendimiento de cultivo medio por cada factor, así como por los factores agrupados.


```python
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

df_anova2 = pd.read_csv("https://raw.githubusercontent.com/Opensourcefordatascience/Data-sets/master/crop_yield.csv")
model = ols('Yield ~ C(Fert)*C(Water)', df_anova2).fit()
print(f"Overall model F({model.df_model: .0f},{model.df_resid: .0f}) = {model.fvalue: .3f}, p = {model.f_pvalue: .4f}")

res = sm.stats.anova_lm(model, typ= 2)
res
```

    Overall model F( 3, 16) =  4.112, p =  0.0243





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
      <th>sum_sq</th>
      <th>df</th>
      <th>F</th>
      <th>PR(&gt;F)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>C(Fert)</th>
      <td>69.192</td>
      <td>1.0</td>
      <td>5.766000</td>
      <td>0.028847</td>
    </tr>
    <tr>
      <th>C(Water)</th>
      <td>63.368</td>
      <td>1.0</td>
      <td>5.280667</td>
      <td>0.035386</td>
    </tr>
    <tr>
      <th>C(Fert):C(Water)</th>
      <td>15.488</td>
      <td>1.0</td>
      <td>1.290667</td>
      <td>0.272656</td>
    </tr>
    <tr>
      <th>Residual</th>
      <td>192.000</td>
      <td>16.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



**Chi-square**

Podríamos contar los incidentes de algo y comparar lo que mostraron nuestros datos reales con lo que esperaríamos. Supongamos que encuestamos a 45 personas sobre si prefieren el rock, el reggae o el jazz como su tipo de música favorita.

Si no hubiera preferencia, esperaríamos que 15 seleccionaran rock, 15 seleccionarían reggae y 15 seleccionarían jazz. Entonces nuestra muestra indicó que a 30 les gusta el rock, a 5 les gusta el reggae ya 10 les gusta el jazz. Usamos un chi-cuadrado para comparar lo que observamos (real) con lo que esperamos. Una pregunta de investigación de muestra sería: "¿Hay una preferencia por la música rock, reggae o jazz?" Una respuesta de muestra es “No había igual preferencia por el rock, el reggae y el jazz.

Así como las pruebas t nos dicen qué tan seguros podemos estar de decir que hay diferencias entre las medias de dos grupos, el chi-cuadrado nos dice qué tan seguros podemos estar de decir que nuestros resultados observados difieren de los resultados esperados.

Tipos de prueba Chi-Cuadrado:

-Test de independencia: test de independencia de dos variables categóricas.

-Homogeneidad de varianza: prueba si más de dos subgrupos de una población comparten la misma distribución multivariante.

-Bondad de ajuste: si un modelo multinomial para la distribución de la población (P1,....Pm) se ajusta a nuestros datos.

La prueba de independencia y la homogeneidad de la varianza comparten las mismas estadísticas de prueba y grados de libertad por diferente diseño de experimento.

Suposiciones:

1. Una o dos variables categóricas.

2. Observaciones independientes.

3. Resultados mutuamente excluyentes.

4. Grande $n$ y no más del 20% de los conteos esperados $< 5$.

Ejemplo de código: en una encuesta electoral, los votantes pueden clasificarse por género (masculino o femenino) y preferencia de voto (demócrata, republicano o independiente). Podríamos usar una prueba de chi-cuadrado para la independencia para determinar si el género está relacionado con la preferencia de voto.


```python
df_chi = pd.read_csv('https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/chi-test.csv')
contingency_table=pd.crosstab(df_chi["Gender"],df_chi["Like Shopping?"])
print('contingency_table :-\n',contingency_table)

#Valores observados
Observed_Values = contingency_table.values 
print("Observed Values :-\n",Observed_Values)

b=stats.chi2_contingency(contingency_table)
Expected_Values = b[3]
print("Expected Values :-\n",Expected_Values)

no_of_rows=len(contingency_table.iloc[0:2,0])
no_of_columns=len(contingency_table.iloc[0,0:2])
ddof=(no_of_rows-1)*(no_of_columns-1)
print("Degree of Freedom:-",ddof)

alpha = 0.05

from scipy.stats import chi2

chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
chi_square_statistic=chi_square[0]+chi_square[1]
print("chi-square statistic:-",chi_square_statistic)

critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value:',critical_value)

#valor-p
p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)

print('p-value:',p_value)
print('Significance level: ',alpha)
print('Degree of Freedom: ',ddof)
print('chi-square statistic:',chi_square_statistic)
print('critical_value:',critical_value)
print('p-value:',p_value)

if chi_square_statistic>=critical_value:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
    
if p_value<=alpha:
    print("Reject H0,There is a relationship between 2 categorical variables")
else:
    print("Retain H0,There is no relationship between 2 categorical variables")
```

    contingency_table :-
     Like Shopping?  No  Yes
    Gender                 
    Female           2    3
    Male             2    2
    Observed Values :-
     [[2 3]
     [2 2]]
    Expected Values :-
     [[2.22222222 2.77777778]
     [1.77777778 2.22222222]]
    Degree of Freedom:- 1
    chi-square statistic:- 0.09000000000000008
    critical_value: 3.841458820694124
    p-value: 0.7641771556220945
    Significance level:  0.05
    Degree of Freedom:  1
    chi-square statistic: 0.09000000000000008
    critical_value: 3.841458820694124
    p-value: 0.7641771556220945
    Retain H0,There is no relationship between 2 categorical variables
    Retain H0,There is no relationship between 2 categorical variables


**5. ¿Cuál es el valor p?**

Al probar una hipótesis, el valor p es la probabilidad de que observemos resultados al menos tan extremos como nuestro resultado debido puramente al azar si la hipótesis nula fuera cierta. Usamos valores p para sacar conclusiones en las pruebas de significación. Más específicamente, comparamos el valor p con un nivel de significancia para sacar conclusiones sobre nuestras hipótesis.

Si el valor p es menor que el nivel de significancia que elegimos, entonces rechazamos la hipótesis nula, a favor de la hipótesis alternativa.

Si el valor p es mayor o igual que el nivel de significancia, entonces no podemos rechazar la hipótesis nula, pero esto no significa que la aceptemos.

¿Qué valor se usa con más frecuencia para determinar la significación estadística?

Un valor de alfa = 0,05 se utiliza con mayor frecuencia como umbral de significación estadística.

En otras palabras, un valor bajo de p significa que tenemos evidencia convincente para rechazar la hipótesis nula. Si el valor p es inferior al 5 %, a menudo rechazamos $Ho$ y aceptamos que $Ha$ es verdadero. Decimos que $p < 0.05$ es estadísticamente significativo, porque hay menos del 5% de probabilidad de que nos equivoquemos al rechazar la hipótesis nula.

Una forma de calcular el valor p es a través de una prueba T. Podemos usar la función ttest_ind de Scipy para calcular la prueba t para las medias de dos muestras independientes de puntajes.

Vamos a ver un ejemplo:

María diseñó un experimento en el que algunas personas probaron limonadas de cuatro vasos diferentes e intentaron identificar qué vaso contenía limonada casera. Cada sujeto recibió tres vasos que contenían limonada picada y un vaso que contenía limonada casera (el orden fue aleatorio). Quería probar si los sujetos podían hacer algo mejor que simplemente adivinar al identificar la limonada casera.

Su hipótesis nula $Ho$ : $p = 0.30$

Su hipótesis alternativa $Ha$ : $p > 0.30$

*p es la probabilidad de que las personas identifiquen la limonada casera*

El experimento mostró que 18 de los 50 individuos identificaron correctamente la limonada casera. María calculó que la estadística $p = 18/50 = 0,3$ tenía un valor p asociado de aproximadamente $0,061$.

Usando un nivel de significación de $α = 0.05$, ¿qué conclusiones podemos obtener?

* Dado que el valor p es mayor que el nivel de significación, debemos rechazar la hipótesis nula Ho.

* No tenemos suficiente evidencia para decir que estas personas pueden hacer algo mejor que adivinar al identificar la limonada casera.

La hipótesis nula $Ho$: $p=0.25$ dice que su probabilidad no es mejor que adivinar, y fallamos en rechazar la hipótesis nula.m

**6. Decision and Conclusion**

Después de realizar la prueba de hipótesis, obtenemos un valor p relacionado que muestra la importancia de la prueba.

Si el valor p es más pequeño que el nivel de significancia, hay suficiente evidencia para probar que H₀ no es válido; puede rechazar H₀. De lo contrario, no puede rechazar H₀.

### Posibles errores en las pruebas

¿Qué es el error tipo I?

El error de tipo I es el rechazo de una hipótesis nula verdadera o una clasificación de "falso positivo".

¿Qué es el error tipo II?

El error de tipo II es el no rechazo de una hipótesis nula falsa o una clasificación negativa falsa.

Podemos tomar cuatro decisiones diferentes con la prueba de hipótesis:

Rechaza $Ho$ y $Ho$ no es verdadero (sin error)

No rechaces $ Ho $ y $ Ho $ es verdadero (sin error)

Rechaza $Ho$ y $Ho$ es verdadero (Error tipo 1)

No rechaces $Ho$ y $Ho$ no es verdadero (Error tipo 2)

El error de tipo 1 también se denomina error alfa. El error de tipo 2 también se denomina error Beta.

## Implicaciones en in Machine Learning

El teorema del límite central tiene implicaciones importantes en el aprendizaje automático aplicado. El teorema informa la solución de algoritmos lineales como la regresión lineal, pero no métodos exóticos como redes neuronales artificiales que se resuelven mediante métodos de optimización numérica.

**Covarianza**

La covarianza es una medida de cuánto varían juntas dos variables aleatorias. Si dos variables son independientes, su covarianza es 0.

Covariance is a measure of how much two random variables vary together. If two variables are independent, their covariance is 0. Sin embargo, una covarianza de 0 no implica que las variables sean independientes.

```py
# Código para obtener la covarianza entre dos variables.

df[['Feature1', 'Feature2']].cov()

# Correlación entre dos distribuciones normales utilizando la correlación de Pearson.

df[['Feature1', 'Feature2']].corr(method='pearson')
```

**Pruebas de significación**


-Para hacer inferencias sobre la habilidad de un modelo en comparación con la habilidad de otro modelo, debemos usar herramientas como las pruebas de significación estadística.

-Estas herramientas estiman la probabilidad de que las dos muestras de puntajes de habilidades del modelo se extraigan de la misma o diferente distribución subyacente desconocida de puntajes de habilidades del modelo.

-Si parece que las muestras se extrajeron de la misma población, entonces no se asume ninguna diferencia entre la habilidad de los modelos y las diferencias reales se deben al ruido estadístico.

-La capacidad de hacer afirmaciones de inferencia como esta se debe al teorema del límite central, nuestro conocimiento de la distribución de Gaussian y la probabilidad de que las dos medias de muestra sean parte de la misma distribución de Gaussian de medias de muestra.

**Intervalos de confianza**

-Una vez que hemos entrenado un modelo final, es posible que deseemos hacer una inferencia sobre qué tan hábil se espera que sea el modelo en la práctica.

-La presentación de esta incertidumbre se denomina intervalo de confianza.

-Podemos desarrollar múltiples evaluaciones independientes (o casi independientes) de la precisión de un modelo para generar una población de estimaciones de habilidades candidatas.

-La media de las estimaciones de esta habilidad será una estimación (con error) de la verdadera estimación subyacente de la habilidad del modelo en el problema.

-Con el conocimiento de que la media de la muestra será parte de una distribución Gaussian del teorema del límite central. Podemos usar el conocimiento de la distribución de Gaussian para estimar la probabilidad de la media de la muestra en función del tamaño de la muestra y calcular un intervalo de confianza deseado alrededor de la habilidad del modelo.

Fuente:

https://www.kaggle.com/code/carlolepelaars/statistics-tutorial/notebook
    
https://byjus.com/maths/level-of-significance/

https://byjus.com/maths/sampling-methods/

https://researchbasics.education.uconn.edu/anova_regression_and_chi-square/#

https://cheatography.com/mmmmy/cheat-sheets/hypothesis-testing-cheatsheet/

https://towardsdatascience.com/hypothesis-testing-in-machine-learning-using-python-a0dc89e169ce

https://github.com/yug95/MachineLearning

