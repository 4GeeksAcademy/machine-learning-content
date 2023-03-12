# Procesamiento del lenguaje natural (NLP)

**En primer lugar, ¿qué es la clasificación de texto?**

La clasificación de texto es un proceso automatizado de clasificación de texto en categorías predefinidas. Podemos clasificar los correos electrónicos en spam o no spam, los artículos de noticias en diferentes categorías, como Política, Bolsa, Deportes, etc. 

Esto se puede hacer con la ayuda del procesamiento del lenguaje natural y diferentes algoritmos de clasificación como Naive Bayes, Máquinas de Vectores de Soporte e incluso redes neuronales en Python.

**¿Qué es el procesamiento del lenguaje natural?**

El procesamiento del lenguaje natural (NLP) es un campo de inteligencia artificial (AI) que permite que los programas de computadora reconozcan, interpreten y manipulen los lenguajes humanos.

Si bien una computadora puede ser bastante buena para encontrar patrones y resumir documentos, debe transformar palabras en números antes de darles sentido. Esta transformación es necesaria porque las máquinas “aprenden” gracias a las matemáticas, y las matemáticas no funcionan muy bien con las palabras. Antes de transformar las palabras en números, a menudo se limpian de elementos como caracteres especiales y puntuación, y se modifican en formas que las hacen más uniformes e interpretables.

## Pasos para construir un modelo de NLP

- **Paso 1:** Agregar las bibliotecas requeridas.

- **Paso 2:** Establecer una semilla aleatoria. Esto se usa para reproducir el mismo resultado cada vez si el script se mantiene consistente; de ​​lo contrario, cada ejecución producirá resultados diferentes. La semilla se puede establecer en cualquier número.

- **Paso 3:** Cargar el conjunto de datos.

- **Paso 4:** Preprocesar el contenido de cada texto. Este es un paso muy importante.

    Los datos del mundo real a menudo son incompletos, inconsistentes y/o carecen de ciertos comportamientos o tendencias, y es probable que contengan muchos errores. El preprocesamiento de datos es un método comprobado para resolver tales problemas. Esto ayudará a obtener mejores resultados a través de los algoritmos de clasificación. Puede ver el detalle del proceso de limpieza a continuación en "Detalle de los pasos de preprocesamiento de datos".

- **Paso 5:** Separar los conjuntos de datos de entrenamiento y prueba.

- **Paso 6:** Codificación del objetivo.

    La etiqueta codifica la variable de destino para transformar su tipo de cadena, si es así, en valores numéricos que el modelo pueda entender. Esto también se puede hacer como uno de sus primeros pasos, pero asegúrese de convertirlo antes de dividir los datos en entrenamiento y prueba.

- **Paso 7:** Bolsa de palabras (vectorización)

    Es el proceso de convertir palabras de oraciones en vectores de características numéricas. Es útil ya que los modelos requieren que los datos estén en formato numérico. Entonces, si la palabra está presente en esa oración en particular, pondremos 1; de lo contrario, 0. El método más popular se llama TF-IDF. Significa "Frecuencia de término - Frecuencia de documento inverso". TF-IDF son puntajes de frecuencia de palabras que intentan resaltar palabras que son más interesantes, por ejemplo, frecuentes en un documento pero no entre documentos.
    
    Esto ayudará a TF-IDF a construir un vocabulario de palabras que ha aprendido de los datos del corpus y asignará un número entero único a cada una de estas palabras. De acuerdo con el siguiente código, habrá un máximo de 5000 palabras únicas. Finalmente, transformaremos X_train y X_test en X_train_Tfidf y X_test_Tfidf vectorizados. Estos ahora contendrán para cada fila una lista de números enteros únicos y su importancia asociada según lo calculado por TF-IDF.

```py
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(news_df['text_final'])

X_train_Tfidf = Tfidf_vect.transform(X_train)
X_test_Tfidf = Tfidf_vect.transform(X_test)
```

- **Paso 8:** Usar el algoritmo ML para predecir el resultado.


### **Detalle de los pasos de preprocesamiento de datos**

Las siguientes partes del proceso de limpieza tendrán diferentes ejemplos de implementación de código. Siempre puede agregar o eliminar los pasos que mejor se adapten al conjunto de datos con el que está tratando:

**1. Quita las filas en blanco o duplicadas** de los datos. Podemos hacer esto usando dropna y drop_duplicates respectivamente.

**2. Cambia todo el texto a minúsculas** porque Python interpreta mayúsculas y minúsculas de manera diferente. Aquí un ejemplo de cómo convertir entradas a minúsculas. Recuerda que puedes incluir este paso como parte de una función de limpieza.

```py
news_df['text'] = [entry.lower() for entry in news_df['text']]
```

**3. Elimina texto no alfabético, tags y caracteres de puntuación**. Esto se puede hacer con la ayuda de expresiones regulares.

**4. Elimina palabras vacías.** Elimina todas las palabras de uso frecuente, como "yo, o ella, tengo, hiciste, tú, para".

Los pasos 3 y 4 del proceso de limpieza (eliminación de caracteres especiales y palabras vacías) se pueden lograr fácilmente usando los módulos nltk y string. Veamos un ejemplo de la puntuación y las palabras vacías que ya han definido estos módulos. 

```py
import string
import nltk

nltk.download('stopwords')
nltk.download('punkt')

stopwords = nltk.corpus.stopwords.words('english')
punctuation = string.punctuation

print(stopwords[:5])
print(punctuation)
>>>   ['i', 'me', 'my', 'myself', 'we']
>>>   !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

# Creación de funciones para eliminar la puntuación y las palabras vacías

def remove_punctuation(text):
    no_punct=[words for words in text if words not in string.punctation]
    words_wo_punct=''.join(no_punct)
    return words_wo_punct
news_df['title_wo_punct'] = news_df['title'].apply(lambda x: remove_punctuation(x))

def remove_stopwords(text):
    text=[word for word in text if word not in stopword]
    return text
news_df['title_wo_punct_split_wo_stopwords'] = news_df['title_wo_punct_split'].apply(lambda x: remove_stopwords(x))

```

También podemos decidir eliminar las palabras vacías agregando un parámetro llamado "stop_words" en el "TFidfVectorizer" del paso de vectorización.

**5. Tokenización de palabras.** Es el proceso de dividir un flujo de texto en palabras, frases, símbolos u otros elementos significativos llamados tokens. La lista de tokens se convierte en entrada para su posterior procesamiento. La biblioteca NLTK tiene word_tokenize y sent_tokenize para dividir fácilmente un flujo de texto en una lista de palabras u oraciones, respectivamente. Aquí un par de ejemplos. El segundo ejemplo es una función que tokeniza y convierte a minúsculas al mismo tiempo.

```py
# Ejemplo 1
news_df['text']= [word_tokenize(entry) for entry in news_df['text']]

# Ejemplo 2 : 
def tokenize(text):
    split=re.split("\W+",text) #--->\W+” se divide en uno o más caracteres que no son palabras
    return split
news_df['title_wo_punct_split']=news_df['title_wo_punct'].apply(lambda x: tokenize(x.lower())) #---> La nueva columna ha creado una lista, al dividir todos los caracteres que no son palabras.
```

**6. Lematización/Derivación.** Es el proceso de reducir las formas flexivas de cada palabra a una base o raíz común. El objetivo principal es reducir las variaciones de una misma palabra, reduciendo así el corpus de palabras que incluimos en el modelo. La diferencia entre la derivación y la lematización es que la derivación corta el final de la palabra sin tener en cuenta el contexto de la palabra. Mientras que la Lematización considera el contexto de la palabra y la acorta a su forma raíz según la definición del diccionario. La Derivación es un proceso más rápido en comparación con la Lematización. Por lo tanto, es un compromiso entre velocidad y precisión. Por ejemplo, si el mensaje contiene alguna palabra de error como "frei" que podría estar mal escrita para "gratis". La Derivación reducirá esa palabra de error a su palabra raíz, es decir, "fre". Como resultado, “fre” es la raíz de la palabra tanto para “free” como para “frei”.

```py
print(ps.stem('believe'))
print(ps.stem('believing'))
print(ps.stem('believed'))
print(ps.stem('believes'))
```

*Los resultados del tallo para todo lo anterior es believ*

```py
print(wn.lemmatize(“believe”))
print(wn.lemmatize(“believing”))
print(wn.lemmatize(“believed”))
print(wn.lemmatize(“believes”))
```

*Los resultados de la lematización en el orden de las declaraciones impresas son: creer, creyendo, creyó y creencia. La lematización produce el mismo resultado si la palabra no está en el corpus. Creer está lematizado a creencia (la raíz de la palabra)*    
    
## Proceso de limpieza con un ejemplo de nube de palabras

¿Qué es una nube de palabras?

Las nubes de palabras son una forma útil de visualizar datos de texto porque facilitan la comprensión de las frecuencias de las palabras. Las palabras que aparecen con más frecuencia dentro del texto del correo electrónico aparecen más grandes en la nube. Las nubes de palabras facilitan la identificación de "palabras clave".

```py
# Importación de bibliotecas
import pandas as pd
import sqlite3
import regex as re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Cargando conjunto de datos
df = pd.read_csv('emails.csv')

# Visualizar primeras filas

df.head()
```

![wordcloud](../assets/wordcloud.jpg)


```py
# EDA: eliminar las filas duplicadas y establecer algunos recuentos de referencia.
print("spam count: " +str(len(df.loc[df.spam==1])))
print("not spam count: " +str(len(df.loc[df.spam==0])))
print(df.shape)
df['spam'] = df['spam'].astype(int)

df = df.drop_duplicates()
df = df.reset_index(inplace = False)[['text','spam']]
```

Todo el texto debe estar en minúsculas y sin signos de puntuación ni caracteres especiales, para facilitar su análisis. Usando expresiones regulares, es fácil limpiar el texto usando un bucle. El siguiente código crea una lista vacía clean_desc, luego usa un bucle for para recorrer el texto línea por línea, poniéndolo en minúsculas, eliminando la puntuación y los caracteres especiales, y agregándolo a la lista. Finalmente, reemplaza la columna de texto con los datos en la lista clean_desc.

```py
clean_desc = []

for w in range(len(df.text)):
    desc = df['text'][w].lower()
    
    # Eliminar puntuación
    desc = re.sub('[^a-zA-Z]', ' ', desc)
    
    # Eliminar tags
    desc=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",desc)
    
    # Eliminar dígitos y caracteres especiales
    desc=re.sub("(\\d|\\W)+"," ",desc)
    
    clean_desc.append(desc)

# Asignar las descripciones limpias al marco de datos
df['text'] = clean_desc

df.head(3)
```

![wordcloud2](../assets/wordcloud2.jpg)

Eliminar **palabras vacías** del texto del correo electrónico permite que se destaquen las palabras frecuentes más relevantes. ¡Eliminar palabras vacías es una técnica común! Algunas bibliotecas de Python como NLTK vienen precargadas con una lista de palabras vacías, pero es fácil crear una desde cero. El siguiente código incluye algunas palabras relacionadas con el correo electrónico como "re" y "asunto", pero depende del analista determinar qué palabras deben incluirse o excluirse. 

```py
stop_words = ['is','you','your','and', 'the', 'to', 'from', 'or', 'I', 'for', 'do', 'get', 'not', 'here', 'in', 'im', 'have', 'on', 're', 'new', 'subject']
```

Ahora, ¿cómo podemos construir una nube de palabras?

Hay una biblioteca de Python para crear nubes de palabras. Podemos usar pip para instalarlo. La nube de palabras se puede configurar con varios parámetros como alto y ancho, palabras vacías y palabras máximas, y se puede mostrar usando Matplotlib.

```py
pip install wordcloud

wordcloud = WordCloud(width = 800, height = 800, background_color = 'black', stopwords = stop_words, max_words = 1000
                      , min_font_size = 20).generate(str(df['text']))
                      
# Trazar la nube de palabras
fig = plt.figure(figsize = (8,8), facecolor = None)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
```

![wordcloud3](../assets/wordcloud3.jpg)

Cuando observas una nube de palabras, fíjate que se trata principalmente de palabras sueltas. Cuanto mayor sea la palabra, mayor será su frecuencia. Para evitar que la nube de palabras genere oraciones, el texto pasa por un proceso llamado tokenización. Es el proceso de dividir una oración en palabras individuales. Las palabras individuales se llaman tokens.

Fuente:

https://projectgurukul.org/spam-filtering-machine-learning/

https://towardsdatascience.com/3-super-simple-projects-to-learn-natural-language-processing-using-python-8ef74c757cd9

https://www.youtube.com/watch?v=VDg8fCW8LdM

https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34

https://towardsdatascience.com/nlp-in-python-data-cleaning-6313a404a470










