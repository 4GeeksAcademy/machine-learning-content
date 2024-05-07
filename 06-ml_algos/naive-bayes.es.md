## Naive Bayes

**Naive Bayes** es un algoritmo de clasificación basado en el teorema de Bayes, que es una técnica estadística que utiliza la probabilidad para hacer predicciones. Este algoritmo es muy simple, pero efectivo y se utiliza ampliamente en diversas áreas del Machine Learning.

El nombre *Naive* (ingenuo en español) proviene del supuesto que defiende que todas las características (variables predictoras) del conjunto de datos son independientes entre sí (no existe ninguna correlación entre ellas) dado el valor de la variable objetivo. En otras palabras, este supuesto asume que cada característica contribuye de forma independiente a la probabilidad de pertenecer a una clase particular.

### Teorema de Bayes

El **teorema de Bayes** es un concepto fundamental en probabilidad que nos permite actualizar nuestras creencias o probabilidades sobre un evento dadas nuevas evidencias. La fórmula en la que se fundamenta este teorema es la siguiente:

$P(A|B) = {P(B|A) · P(A)} / {P(B)}$,

Y donde:

- $P(A|B)$ es la probabilidad de que el evento $A$ ocurra dado que ya sabemos que el evento $B$ ha ocurrido.
- $P(B|A)$ es la probabilidad de que el evento $B$ ocurra dado que ya sabemos que el evento $A$ ha ocurrido.
- $P(A)$ es la probabilidad inicial de que el evento $A$ ocurra antes de considerar la evidencia $B$.
- $P(B)$ es la probabilidad de que ocurra $B$.

Además, $A$ es la variable a predecir y $B$ es la predictora. El teorema de Bayes nos permite ajustar nuestras creencias originales ($P(A)$) sobre un evento, usando la nueva información ($P(B|A)$ y $P(B)$). Esencialmente, nos ayuda a calcular la probabilidad actualizada de que un evento ocurra tomando en cuenta una nueva evidencia. Es una herramienta muy útil en muchos campos, desde la ciencia y la medicina hasta el Machine Learning y la inteligencia artificial, para tomar decisiones y hacer predicciones basadas en datos observados.

### Implementaciones

En `scikit-learn` hay tres implementaciones de este modelo: `GaussianNB`, `MultinomialNB` y `BernoulliNB`. Estas implementaciones se diferencian principalmente en el tipo de datos que pueden manejar y las suposiciones que hacen sobre la distribución de los datos:

| | GaussianNB | MultinomialNB | BernoulliNB |
|-|------------|---------------|-------------|
| Tipo de datos | Datos continuos. | Datos discretos. | Datos binarios. | 
| Distribuciones | Supone que los datos siguen una distribución normal. | Supone que los datos siguen una distribución multinomial. | Supone que los datos siguen una distribución Bernoulli. |
| Uso común | Clasificación con características numéricas. | Clasificación con características que representan conteos o frecuencias discretas. | Clasificación con características binarias. |

Si tenemos características numéricas y categóricas en tus datos, hay diferentes estrategias, pero la mejor para preservar la utilidad e idoneidad de este modelo es transformar las categóricas en numéricas utilizando técnicas de codificación como hemos visto atrás: `pd.factorize` de `Pandas`.

### Hiperparametrización del modelo

Podemos construir un modelo de Naive Bayes fácilmente en Python utilizando la librería `scikit-learn` y las funciones `GaussianNB`, `MultinomialNB` y `BernoulliNB`. Algunos de sus hiperparámetros más importantes y los primeros en los que debemos centrarnos son:

- `alpha`: Se utiliza para evitar probabilidades cero en características que no aparecen en el conjunto de entrenamiento. Un valor mayor agrega más suavizado (solo para `MultinomialNB` y `BernoulliNB`).
- `fit_prior`: Indica si se deben aprender las probabilidades a priori de las clases a partir de los datos o si se deben usar probabilidades uniformes (solo para `MultinomialNB`).
- `binarize`: Umbral para normalizar características. Si se proporciona un valor, las características se binarizan según ese umbral; de lo contrario, se asume que las características ya están binarizadas. Si no lo estuvieran y no se usa este hiperparámetro, el modelo puede no funcionar bien (solo para `BernoulliNB`).

Como puedes ver la hiperparametrización en este tipo de modelos es muy reducida, así que una forma de optimizar este tipo de modelos es, por ejemplo, eliminar las variables que estén muy correlacionadas (si la variable $A$ y $B$ tienen una alta correlación, se elimina una de ellas), ya que en este tipo de modelos tienen una importancia doble.
