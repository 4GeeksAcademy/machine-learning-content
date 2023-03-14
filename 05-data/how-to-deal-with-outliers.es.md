# Cómo lidiar con los valores atípicos

Como vimos en el análisis exploratorio de datos, en estadística, un valor atípico es un punto de datos que difiere significativamente de otras observaciones. Un valor atípico puede deberse a la variabilidad en la medición o puede indicar un error experimental.

Los valores atípicos son problemáticos para muchos análisis estadísticos porque pueden confundir el proceso de entrenamiento y dar malos resultados. Desafortunadamente, no existen reglas estadísticas estrictas para identificar definitivamente los valores atípicos. Encontrar valores atípicos depende del conocimiento del área temática y la comprensión del proceso de recopilación de datos. Hay muchas razones por las que podemos encontrar valores atípicos en nuestros datos.

## Cómo identificar posibles valores atípicos

- Graficando sus datos para identificar valores atípicos

Los diagramas de caja, los histogramas y los diagramas de dispersión pueden resaltar los valores atípicos. Los diagramas de caja utilizan el método intercuartílico con vallas para encontrar valores atípicos. También podemos usarlo para encontrar valores atípicos en diferentes grupos de una columna. Los histogramas buscan barras aisladas. Diagramas de dispersión para detectar valores atípicos en un entorno multivariado. Un diagrama de dispersión con línea de regresión muestra cómo la mayoría de los puntos siguen la línea ajustada para el modelo, excepto algunos valores atípicos.

- Usando puntuaciones Z para detectar valores atípicos

Los puntajes Z pueden cuantificar valores atípicos cuando nuestros datos tienen una distribución normal. Las puntuaciones Z son el número de desviaciones estándar por encima y por debajo de la media que cae cada valor. Por ejemplo, una puntuación Z de 2 indica que una observación está dos desviaciones estándar por encima del promedio, mientras que una puntuación Z de -2 significa que está dos desviaciones estándar por debajo de la media. Una puntuación Z de cero representa un valor que es igual a la media. Los puntajes Z pueden ser engañosos con conjuntos de datos pequeños. Los tamaños de muestra de 10 o menos observaciones no pueden tener puntajes Z que excedan un valor de corte de +/-3.

- Using the IQR to create boundaries

El IQR es el 50% medio del conjunto de datos. Es el rango de valores entre el tercer cuartil y el primer cuartil (Q3 – Q1).

Hay muchas maneras de identificar valores atípicos. Debemos utilizar nuestro profundo conocimiento sobre todas las variables al analizar los datos. Parte de este conocimiento es saber qué valores son típicos, inusuales e imposibles. Con esa comprensión de los datos, la mayoría de las veces es mejor usar métodos visuales.

No todos los valores atípicos son malos y algunos no deben eliminarse. De hecho, los valores atípicos pueden ser muy informativos sobre el área temática y el proceso de recopilación de datos. Es importante comprender cómo se producen los valores atípicos y si podrían volver a ocurrir como parte normal del proceso o área de estudio.

## Cómo manejar los valores atípicos 

¿Deberíamos eliminar los valores atípicos?

Depende de lo que causa los valores atípicos.

Causas de los valores atípicos:

- Errores de medición y entrada de datos

Imaginemos que tenemos un array de edades: {30,56,21,50,35,83,62,22,45,233}. 233 es claramente un valor atípico porque es una edad imposible ya que nadie vive 233 años. Examinando los números más de cerca, podemos concluir que la persona a cargo de la entrada de datos pudo haber ingresado accidentalmente un 3 doble, por lo que el número real habría sido 23. Si determinas que un valor atípico es un error, corrije el valor cuando sea posible. Eso puede implicar corregir el error tipográfico o posiblemente volver a medir el artículo o la persona.

- Problemas de muestreo

Las estadísticas inferenciales usan muestras para sacar conclusiones sobre una población específica. Los estudios deben definir cuidadosamente una población y luego extraer una muestra aleatoria de ella específicamente. Ese es el proceso por el cual un estudio puede aprender acerca de una población.

Desafortunadamente, tu estudio podría obtener accidentalmente un artículo o una persona que no sea de la población objetivo. Si puedes establecer que un elemento o una persona no representa a tu población objetivo, puedes eliminar ese punto de datos. Sin embargo, debes poder atribuir una causa o razón específica por la cual ese artículo de muestra no se ajusta a su población objetivo.

- Variación natural

El proceso o la población que está estudiando puede producir valores extraños de forma natural. No hay nada malo con estos puntos de datos. Son inusuales, pero son una parte normal de la distribución de datos. Si nuestro tamaño de muestra es lo suficientemente grande, podemos encontrar valores inusuales. En una distribución normal, aproximadamente 1 de cada 340 observaciones estará al menos a tres desviaciones estándar de la media. Si el valor extremo es una observación legítima que es una parte natural de la población que estamos estudiando, debe dejarlo en el conjunto de datos.

### ¿Cómo manejarlos?

A veces es mejor mantener valores atípicos en sus datos. Pueden capturar información valiosa que es parte de su área de estudio. Retener estos puntos puede ser difícil, ¡particularmente cuando reduce la significancia estadística! Sin embargo, la exclusión de valores extremos únicamente debido a su carácter extremo puede distorsionar los resultados al eliminar información sobre la variabilidad inherente al área de estudio. Estás obligando al área temática a parecer menos variable de lo que es en realidad.

Si el valor atípico en cuestión es:

Un error de medición o error de entrada de datos, corrije el error si es posible. Si no puedes arreglarlo, elimina esa observación porque sabes que es incorrecta.

Si los valores atípicos no son parte de la población que está estudiando (es decir, propiedades o condiciones inusuales), puede eliminar legítimamente el valor atípico. Si la característica también tiene valores nulos, es una opción reemplazar los valores atípicos con valores nulos para que puedan tratarse junto con los valores nulos. Si los valores atípicos son una parte natural de la población que estás estudiando, no debes eliminarlos.

Cuando decidas eliminar valores atípicos, documenta los puntos de datos excluidos y explica tu razonamiento. Debes poder atribuir una causa específica para eliminar los valores atípicos. Otro enfoque es realizar el análisis con y sin estas observaciones y discutir las diferencias. Comparar los resultados de esta manera es particularmente útil cuando no estás seguro de eliminar un valor atípico y cuando hay un desacuerdo sustancial dentro de un grupo sobre esta pregunta.

¿Qué hacer cuando queremos incluir valores atípicos pero no queremos que engañen nuestros resultados?

Podemos usar técnicas de bootstrapping, modelos que son robustos y no sensibles a los valores atípicos.

En el caso del conjunto de datos del Titanic, solo vamos a tratar la columna 'Fare' que tenía valores atípicos.

Fuente:

https://www.kdnuggets.com/2017/01/3-methods-deal-outliers.html?msclkid=69e30c98cf9111ecb308f37782fbdee6

https://statisticsbyjim.com/basics/outliers/

https://statisticsbyjim.com/basics/remove-outliers/#:~:text=If%20the%20outlier%20in%20question%20is%3A%201%20A,you%20are%20studying%2C%20you%20should%20not%20remove%20it.?msclkid=69e228d8cf9111ec8475cfedf095ca97

https://medium.com/@mxcsyounes/hands-on-with-feature-engineering-techniques-dealing-with-outliers-fcc9f57cb63b
