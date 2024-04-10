## Modelos Lineales Regularizados

Un **modelo lineal regularizado** (*regularized linear model*) es una versión de un modelo lineal que incluye un elemento en su función para evitar el sobreajuste (*overfitting*) y mejorar la capacidad de aprendizaje del modelo.

En términos generales, un modelo lineal (como el que vimos en el módulo anterior) intenta encontrar la relación entre las variables de entrada y la variable de salida. Sin embargo, si un modelo lineal tiene demasiados parámetros o si los datos son muy ruidosos, puede suceder que el modelo se ajuste demasiado a los datos de entrenamiento, produciéndose un claro sobreajuste y que dificulta generalizar bien a nuevos datos.

Para evitar este problema, los modelos lineales regularizados añaden un término extra para penalizar los valores de los coeficientes que son demasiado grandes. Estos modelos son regresiones lineales como las vistas en el módulo anterior pero con la adición de un término de regularización. Los dos tipos de modelos son:

- **Modelo lineal regularizado de Lasso** (*L1*): Añade una penalización igual al valor absoluto de la magnitud de los coeficientes. Puede resultar en coeficientes iguales a cero, lo que indica que la característica correspondiente no se utiliza en el modelo.
- **Modelo lineal regularizado de Ridge** (*L2*): Añade una penalización igual al cuadrado de la magnitud de los coeficientes. Esto tiende a reducir los coeficientes, pero no los hace exactamente cero, por lo que todas las características se mantienen en el modelo.

Ambas técnicas intentan limitar o "penalizar" el tamaño de los coeficientes del modelo. Imaginemos que estamos ajustando una línea a puntos en un gráfico:

- **Regresión lineal**: Solo nos preocupamos por encontrar la línea que se ajusta mejor a los puntos.
- **Regresión lineal de Ridge**: Intentamos encontrar la línea que se ajusta mejor, pero también queremos que la pendiente dé la línea lo más pequeña posible.
- **Regresión lineal de Lasso**: Al igual que con Ridge, intentamos ajustar la línea y mantener la pendiente pequeña, pero Lasso puede llevar la pendiente a cero si eso ayuda a ajustar los datos. Esto es como si "seleccionase" qué variables son importantes y cuáles no, porque puede reducir la importancia de algunas variables a cero.

### Hiperparametrización del modelo

Podemos construir un modelo lineal regularizado fácilmente en Python utilizando la librería `scikit-learn` y las funciones `Lasso` y `Ridge`. Algunos de sus hiperparámetros más importantes y los primeros en los que debemos centrarnos son:

- `alpha`: Este es el hiperparámetro de regularización. Controla cuánto queremos penalizar los coeficientes altos. Un valor más alto aumenta la regularización y, por lo tanto, los coeficientes del modelo tienden a ser más pequeños. Por el contrario, un valor más bajo la reduce y permite coeficientes más altos. El valor por defecto es `1.0` y su rango de valores va desde `0.0` hasta `infinito`.
- `max_iter`: Es el número máximo de iteraciones del modelo. 

Otro hiperparámetro muy importante es el `random_state`, que controla la semilla de generación aleatoria. Este atributo es crucial para asegurar la replicabilidad.

### Uso del modelo en Python

Puedes fácilmente utilizar `scikit-learn` para programar estos métodos posterior al EDA:

#### Lasso

```py
from sklearn.linear_model import Lasso

# Carga de los datos de train y test
# Estos datos deben haber sido normalizados y correctamente tratados en un EDA completo

lasso_model = Lasso(alpha = 0.1, max_iter = 300)

lasso_model.fit(X_train, y_train)

y_pred = lasso_model.predict(y_test)
```

#### Ridge

```py
from sklearn.linear_model import Ridge

# Carga de los datos de train y test
# Estos datos deben haber sido normalizados y correctamente tratados en un EDA completo

ridge_model = Ridge(alpha = 0.1, max_iter = 300)

ridge_model.fit(X_train, y_train)

y_pred = ridge_model.predict(y_test)
```
