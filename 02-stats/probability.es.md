## Probabilidad

La **probabilidad** (*probability*) es una medida que describe la posibilidad de que ocurra un evento en particular dentro de un conjunto de posibles eventos. Se expresa en una escala que va desde 0 (indicando que un evento es imposible) hasta 1 (indicando que un evento es cierto). También puede expresarse como un porcentaje entre 0% y 100%.

La probabilidad proporciona un marco para cuantificar la incertidumbre asociada con predicciones, inferencias, decisiones y eventos aleatorios. Para entenderla y calcularla, es esencial comprender los siguientes conceptos:

- **Experimento** (*experiment*): Es cualquier acción o procedimiento que puede producir resultados bien definidos. Ejemplo: lanzar un dado.
- **Espacio muestral** (*sample space*): Es el conjunto de todos los posibles resultados de un experimento. Para el ejemplo del dado, el espacio muestral es $S = {1, 2, 3, 4, 5, 6}$ (ya que son todas las posibles caras que pueden salirnos al lanzar un dado).
- **Evento** (*event*): También llamado suceso, es el subconjunto específico de resultados dentro de un espacio muestral. En nuestro ejemplo, refiriéndonos al evento "obtener un número par" se representaría como $E = {2, 4, 6}$ (ya que todos esos números son pares).

La probabilidad de un evento se calcula como la relación entre el número de resultados favorables para ese evento y el número total de resultados en el espacio muestral. Por ejemplo, la probabilidad de obtener un número par al lanzar un dado es:

$P(E) = {3} \ {6} = 0.5$

### Operaciones entre sucesos

Como hemos visto, un **suceso** (*event*) es el resultado de un experimento aleatorio y siempre lleva asociada una probabilidad. A menudo puede que nos interese relacionar las probabilidades entre dos sucesos, y esto se lleva a cabo a través de las operaciones entre sucesos.

#### Unión de sucesos

Es el suceso que tiene lugar si ocurre al menos uno de los dos sucesos. Se denota por $A \cup B$. Por ejemplo, si $A$ es el suceso de obtener un 2 en un lanzamiento de dado y $B$ es el suceso de obtener un 3, entonces $A \cup B$ es el suceso de obtener un 2 o un 3.

![union](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/union.png?raw=true)

#### Intersección de sucesos

Es el suceso que tiene lugar si ocurren ambos sucesos a la vez. Se denota por $A \cap B$. Por ejemplo, si $A$ es el suceso de obtener un número menor que 4 y $B$ es el suceso de obtener un número par, entonces $A \cap B$ es el suceso de obtener un 2 (porque es par y el único número menor que 4).

![intersection](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/intersection.png?raw=true)

#### Complemento de un suceso

Es el suceso que tiene lugar si no ocurre el suceso dado. Se denota por $A'$. Por ejemplo, si $A$ es el suceso de obtener un número par al lanzar un dado, entonces $A'$ es el suceso de obtener un número impar.

![complement](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/complement.png?raw=true)

#### Diferencia entre sucesos

Es el suceso que tiene lugar si ocurre el primer suceso pero no el segundo. Se denota por $A - B$. Por ejemplo, si $A$ es el suceso de obtener un número menor que 4 y $B$ es el suceso de obtener un número par, entonces $A - B$ es el suceso de obtener un 1 o un 3.

![difference](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/difference.png?raw=true)





En muchos casos, es más práctico usar una regla simple pero incierta en lugar de una regla compleja pero cierta, incluso si la regla verdadera es determinista y nuestro sistema de modelado tiene la fidelidad para adaptarse a una regla compleja.

Por ejemplo, la regla simple "La mayoría de las aves vuelan" es **económica** de desarrollar y es ampliamente útil, mientras que una regla de la forma "Las aves vuelan, excepto las aves muy jóvenes que aún no han aprendido a volar, enfermas o lesionadas, aves que han perdido la capacidad de volar, especies de aves que no vuelan, incluidos el casuario, el avestruz y el kiwi..." es **cara** de desarrollar, mantener, comunicar y después de todo este esfuerzo, sigue siendo **frágil* * y propenso al **fracaso**.

La probabilidad puede verse como la extensión de la lógica para hacer frente a la incertidumbre.

La teoría de la probabilidad proporciona un set de reglas formales para determinar la probabilidad de que una proposición sea verdadera dada la probabilidad de otras proposiciones.

### Tipos de probabilidad

**Probabilidad Frecuentista:**

•	Frecuencia de eventos. Ejemplo: La posibilidad de sacar una determinada mano en el póquer.

•	Modelo fijo, datos diferentes. (Ejecutamos los mismos experimentos cada vez con datos diferentes)

**Probabilidad Bayesiana:**

•	Un grado de creencia. Ejemplo: cuando un médico que dice que un paciente tiene un 40% de posibilidades de tener gripe.

•	Datos fijos y diferentes modelos. (Usamos la misma creencia para comprobar la incertidumbre de diferentes modelos y actualizar nuestras creencias) -Basado en la Regla de Bayes.

**Diferencias entre estadística bayesiana y frecuentista.**

Ambos intentan estimar un parámetro de población basado en una muestra de datos.

Los frecuentistas tratan los datos como aleatorios y la estadística como fija. Las inferencias se basan en un muestreo infinito a largo plazo, y las estimaciones del parámetro vienen en forma de estimaciones puntuales o intervalos de confianza.

Los Bayesianos tratan el parámetro de población como aleatorio y los datos como fijos. Las estadísticas Bayesianas nos permiten hacer conjeturas informadas sobre el valor del parámetro en forma de distribuciones previas. Las estimaciones del parámetro vienen en forma de distribuciones posteriores.

![bayessian.png](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/bayessian.png?raw=true)

### Eventos

Un evento es un set de resultados de un experimento en probabilidad.

En la probabilidad Bayesiana, un evento se define como la descripción del siguiente espacio de estado posible utilizando el conocimiento del estado actual.

Los eventos pueden ser :

•	Independiente - cada evento no se ve afectado ni por los eventos anteriores o futuros.

•	Dependiente - un evento se ve afectado por otros eventos.

•	Mutuamente excluyentes - los eventos no pueden ocurrir al mismo tiempo.

El complemento de un evento son todos los demás resultados de un evento. Por ejemplo, si el evento es Cruz, el complemento es Cara. Si conoces la probabilidad de $p(x)$, puedes encontrar el complemento haciendo $1 - p(x)$. A veces es más fácil calcular primero el complemento antes que la probabilidad real.

### La Probabilidad Condicional

La probabilidad condicional de un evento $B$ es la probabilidad de que ocurra el evento dado que el evento $A$ ya ha ocurrido.

Si $A$ y $B$ son dos eventos, entonces la probabilidad condicional se puede designar como $P(A dado B)$ o $P(A|B)$. La probabilidad condicional se puede calcular a partir de la probabilidad conjunta $(A | B) = P(A, B) / P(B)$. (La probabilidad condicional no es simétrica).

Por ejemplo: $P(A | B) != P(B | A)$

Otras formas de calcular la probabilidad condicional incluyen el uso de la otra probabilidad condicional, es decir:

$P(A|B) = P(B|A) * P(A) / P(B)$ ----->Bayes Theorem

También se usa el reverso:

$P(B|A) = P(A|B) * P(B) / P(A)$

Esta forma de cálculo es útil cuando es difícil calcular la probabilidad conjunta. De lo contrario, cuando la probabilidad condicional inversa está disponible, el cálculo a través de esto se vuelve fácil.

Este cálculo alternativo de probabilidad condicional se denomina Regla de Bayes o Teorema de Bayes.

**Teorema de Bayes**

El teorema de Bayes es una fórmula matemática simple utilizada para calcular probabilidades condicionales. Este teorema establece que:

$P(A|B) = P(B|A) * P(A) / P(B)$

El teorema de Bayes se usa para el cálculo de una probabilidad condicional donde la intuición a menudo falla. Aunque se usa ampliamente en probabilidad, el teorema también se está aplicando en el campo de Machine Learning.

Su uso en Machine Learning incluye el ajuste de un modelo a un DataSet (conjunto de datos) de entrenamiento y el desarrollo de modelos de clasificación.

Siempre es posible responder preguntas de probabilidad condicional mediante el Teorema de Bayes. Nos dice la probabilidad del evento A dada alguna nueva evidencia B, pero si lo olvidaste, siempre puedes cultivar historias con árboles podados.

Ejemplo:

Los incendios peligrosos son raros (1 %), pero el humo es bastante común (10 %) debido a las barbacoas, y el 90 % de los incendios peligrosos producen humo.

¿Cuál es la probabilidad de un Incendio peligroso cuando hay Humo?

Cálculo

$P(Fire|Smoke) =P(Fire) P(Smoke|Fire)/P(Smoke)$

= 1% x 90%/10%

= 9%

**Teorema de Bayes en Machine Learning**

El teorema de Bayes se puede utilizar tanto en regresión como en clasificación.

Generalmente, en Machine Learning supervisado, cuando queremos entrenar un modelo, los componentes principales son:

- Un set de puntos de datos que contienen características (los atributos que definen dichos puntos de datos)

- Las etiquetas de dicho punto de datos (El tag numérico o categórico que luego queremos predecir en nuevos puntos de datos)

- Una función de hipótesis que vincula dichas características con sus etiquetas correspondientes.

- También tenemos una función de pérdida, que es la diferencia entre las predicciones del modelo y las etiquetas reales que queremos reducir para lograr los mejores resultados posibles.

Referencias:

https://medium.com/brandons-computer-science-notes/an-introdcution-to-probability-45a64aee7606

https://discovery.cs.illinois.edu/learn/Prediction-and-Probability/Conditional-Probability/

https://www.upgrad.com/blog/bayes-theorem-explained-with-example-complete-guide/

https://www.edx.org/es/micromasters/mitx-statistics-and-data-science

https://towardsdatascience.com/bayes-theorem-clearly-explained-with-visualization-5083ea5e9b14

