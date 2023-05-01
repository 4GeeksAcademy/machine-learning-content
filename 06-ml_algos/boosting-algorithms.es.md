# Boosting de Algoritmos

Ya mencionamos el significado del aprendizaje conjunto, que en general es un modelo que hace predicciones basadas en varios modelos diferentes. En la última lección hablamos sobre el embolsado, una técnica que entrena un grupo de modelos individuales de manera paralela con subconjuntos aleatorios de datos. En esta lección aprenderemos la técnica de Boosting.

**¿Qué significa Boosting?**

Entrenamiento de un grupo de modelos individuales de forma **secuencial**. Cada modelo individual aprende de los errores cometidos por el modelo anterior.

## Boosting de Gradiente

Gradient Boosting (Boosting de Gradiente) aprende del error o error residual directamente, en lugar de actualizar los pesos de los puntos de datos.

Pasos de un algoritmo de boosting de gradiente:

- Paso 1: entrenar un árbol de decisión.

- Paso 2: Aplicar el árbol de decisión recién entrenado para predecir.

- Paso 3: Calcular el residuo de este árbol de decisión, guarde los errores residuales como la nueva y.

- Paso 4: Repitir el paso 1 (hasta alcanzar la cantidad de árboles que configuramos para entrenar).

- Paso 5: Haz la predicción final.

**¿En qué se diferencian las máquinas de boosting de gradiente de los algoritmos de árboles de decisión tradicionales?**

El boosting de gradiente implica el uso de múltiples predictores débiles (árboles de decisión) para crear un predictor fuerte. Específicamente, incluye una función de pérdida que calcula el gradiente del error con respecto a cada característica y luego crea iterativamente nuevos árboles de decisión que minimizan el error actual. Se agregan más y más árboles al modelo actual para continuar corrigiendo errores hasta que las mejoras caen por debajo de un umbral mínimo o se ha creado un número predeterminado de árboles.

**¿Qué hiperparámetros se pueden ajustar en el boosting de gradiente además de los hiperparámetros de cada árbol individual?**

Los principales hiperparámetros que se pueden ajustar con los modelos GBM son:

- Función de pérdida: función de pérdida para calcular el gradiente de error

- Tasa de aprendizaje: la tasa a la que los árboles nuevos corrigen/modifican el predictor existente

- Estimadores numéricos - el número total de árboles a producir para el predictor final

Hiperparámetros adicionales específicos de la función de pérdida

Algunas implementaciones específicas, por ejemplo, el boosting de gradiente estocástico, pueden tener hiperparámetros adicionales, como el tamaño de la submuestra (el tamaño de la submuestra afecta la aleatorización en las variaciones estocásticas).

Documentación de Scikit learn sobre Boosting de Gradiente para clasificación: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

Documentación de Scikit learn sobre Boosting de Gradiente para regresión: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html

**¿Cómo podemos reducir el sobreajuste al aumentar el gradiente?**

Reducir la tasa de aprendizaje o reducir el número máximo de estimadores son las dos formas más fáciles de lidiar con modelos de boosting de gradiente que sobreajustan los datos.

Con el boosting de gradiente estocástico, la reducción del tamaño de la submuestra es una forma adicional de combatir el sobreajuste.

Los algoritmos de boosting tienden a ser vulnerables al sobreajuste, por lo que es importante saber cómo reducir el sobreajuste.

## Implementación en Scikit Learn

Imaginemos que ya tenemos nuestros datos divididos en conjuntos de datos de entrenamiento y prueba:

```py

# Cargar bibliotecas

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

# Ajustar un modelo de árbol de decisión como comparación

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

OUTPUT: 0.756

# Ajustar un modelo de Random Forest

clf = RandomForestClassifier(n_estimators=100, max_features="auto",random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

OUTPUT: 0.797

# Paso 6: ajustar un modelo de boosting de gradiente

clf = GradientBoostingClassifier(n_estimators=100)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

OUTPUT:0.834
```

> n_estimators representa cuántos árboles queremos que crezcan.

## ¿Qué es XGBoost?

XGBoost es un algoritmo de Machine Learning basado en un árbol de decisión que utiliza un marco de refuerzo de gradiente.

Piensa en XGBoost como un boosting de gradiente con "esteroides", también se le llama "boosting de gradiente extremo". Una combinación perfecta de técnicas de optimización de software y hardware para obtener resultados superiores utilizando menos recursos informáticos en el menor tiempo posible.

**¿Por qué funciona tan bien?**

Miremos la siguiente imagen para entender las razones por las que funciona tan bien.

![xgboost](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/xgboost.jpg?raw=true)

(Imagen de haciadatascience.com)

Ahora, veamos la siguiente comparación. El modelo XGBoost tiene la mejor combinación de rendimiento de predicción y tiempo de procesamiento en comparación con otros algoritmos.

![xgboost_performance](https://github.com/4GeeksAcademy/machine-learning-content/blob/master/assets/xgboost_performance.jpg?raw=true)

(Imagen de haciadatascience.com)

Establecer los hiperparámetros óptimos de cualquier modelo de Machine Learning puede ser un desafío. Entonces, ¿por qué no dejar que Scikit Learn lo haga por ti?

Para que XGBoost pueda manejar nuestros datos, necesitaremos transformarlos a un formato específico llamado DMatrix. Veamos un ejemplo de cómo definir un modelo XGBoost:

```py
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

D_train = xgb.DMatrix(X_train, label=Y_train)
D_test = xgb.DMatrix(X_test, label=Y_test)

param = {
    'eta': 0.3, 
    'max_depth': 3,  
    'objective': 'multi:softprob',  
    'num_class': 3} 

steps = 20  # El número de iteraciones de entrenamiento.

model = xgb.train(param, D_train, steps)

```

¿Cómo combinaríamos la búsqueda en cuadrícula de Scikit Learn con un clasificador XGBoost?

> Solo haz eso en un gran conjunto de datos si tienes tiempo para matar: ¡hacer una búsqueda en cuadrícula es esencialmente entrenar un conjunto de árboles de decisión muchas veces!

```py

from sklearn.model_selection import GridSearchCV

clf = xgb.XGBClassifier()
parameters = {
     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
     }

grid = GridSearchCV(clf,
                    parameters, n_jobs=4,
                    scoring="neg_log_loss",
                    cv=3)

grid.fit(X_train, Y_train)

```

La lista completa de posibles parámetros está disponible en el sitio web oficial de XGBoost: https://xgboost.readthedocs.io/en/latest/parameter.html 


El aprendizaje de conjunto es muy poderoso y puede usarse no solo para la clasificación sino también para la regresión. Aunque generalmente funcionan principalmente en métodos de árbol, también se pueden aplicar en modelos lineales y svm dentro de los conjuntos de embolsado o refuerzo, para lograr un mejor rendimiento. Pero recuerda, elegir el algoritmo correcto no es suficiente. También debemos elegir la configuración correcta del algoritmo para un conjunto de datos ajustando los hiperparámetros.


Fuente:

https://towardsdatascience.com/basic-ensemble-learning-random-forest-adaboost-gradient-boosting-step-by-step-explained-95d49d1e2725

https://medium.com/@aravanshad/gradient-boosting-versus-random-forest-cfa3fa8f0d80

https://towardsdatascience.com/https-medium-com-vishalmorde-xgboost-algorithm-long-she-may-rein-edd9f99be63d

https://xgboost.readthedocs.io/en/latest/parameter.html

https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7

https://towardsdatascience.com/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e
