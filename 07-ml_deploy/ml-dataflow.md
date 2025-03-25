---
description: >-
  Learn how to manage data workflows for machine learning models. Discover
  strategies for re-training and improving model performance over time!
---
## Data workflow

En el mundo real, lo normal es que los modelos vayan evolucionando a lo largo del tiempo conforme interactúan con los datos y su rendimiento disminuye. Es fácil entender por qué, y es que los datos de los que disponemos en la fase de entrenamiento inicial del modelo pueden no corresponderse plenamente a los que luego vaya a recibir, por diversos motivos:

1. **Abordar el problema utilizando una muestra y no la población total**. Por ejemplo, puede que hayamos diseñado un modelo para predecir el precio de una vivienda en España en función de sus características (metros cuadrados, número de habitaciones, número de baños, etcétera) utilizando nada más que las viviendas de Andalucía, y cuando probamos el modelo sobre una vivienda de otra comunidad autónoma vemos que los importes no son predecibles.
2. **Datos insuficientes**. A veces puede suceder simplemente que el problema es tan nuevo o inexplorado que no contemos con datos suficientes para entrenar un modelo en su totalidad, y en el mundo real no se comporte como pensábamos.
3. **Problema variable**. Puede suceder también que el problema que aborda el modelo cambie con el tiempo, y sea necesario un constante reentrenamiento cada cierto tiempo. 


 Debido a esto, es vital contar con una estrategia que controle y prepare estos datos de cara al reentrenamiento. 