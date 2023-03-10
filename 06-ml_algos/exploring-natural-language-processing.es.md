# Explorando NLP con un clasificador de filtro de SPAM usando SVM

Vamos a construir un modelo de Machine Learning que pueda identificar si un correo electrónico es spam o no, por lo que este es un problema de clasificación binaria. Voy a utilizar la biblioteca de Python Scikit-Learn para explorar los algoritmos de tokenización, vectorización y clasificación estadística. Comencemos por importar las dependencias necesarias, como CountVectorizer, SVM y un par de métricas.

**Importación de dependencias**


```python
import pandas as pd 
import regex as re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import model_selection, svm
from sklearn.metrics import classification_report, accuracy_score
```

**Cargando los datos**


```python
df = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/machine-learning-content/master/assets/spam.csv")
```


```python
# Mirando las primeras filas

df.head()
```




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
      <th>Category</th>
      <th>Message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ham</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ham</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>spam</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ham</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ham</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>



En este ejemplo, hemos decidido convertir nuestra variable objetivo en números de inmediato. El spam se convertirá en 1 y el ham se convertirá en 0.


```python
df['Category'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
```


```python
df.head()
```




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
      <th>Category</th>
      <th>Message</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>Go until jurong point, crazy.. Available only ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>Ok lar... Joking wif u oni...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>Free entry in 2 a wkly comp to win FA Cup fina...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>U dun say so early hor... U c already then say...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>Nah I don't think he goes to usf, he lives aro...</td>
    </tr>
  </tbody>
</table>
</div>



**Proceso de limpieza**

Ya aprendimos cómo limpiar los datos, por lo que comenzaremos contando los correos electrónicos no deseados (spam), los deseados, y también eliminando los duplicados.


```python
# EDA: Establece algunos recuentos de referencia y elimine las filas duplicadas

print("spam count: " +str(len(df.loc[df.Category==1])))
print("not spam count: " +str(len(df.loc[df.Category==0])))
print(df.shape)
df['Category'] = df['Category'].astype(int)
```

    spam count: 747
    not spam count: 4825
    (5572, 2)



```python
df = df.drop_duplicates()
df = df.reset_index(inplace = False)[['Message','Category']]
df.shape
```




    (5157, 2)



Observaciones: hemos descartado 415 filas duplicadas.

Usando expresiones regulares vamos a convertir todas las palabras a minúsculas, limpiamos los datos de signos de puntuación y caracteres especiales.


```python
clean_desc = []

for w in range(len(df.Message)):
    desc = df['Message'][w].lower()
    
    # Eliminar puntuación
    desc = re.sub('[^a-zA-Z]', ' ', desc)
    
    # Eliminar tags
    desc=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",desc)
    
    # Eliminar dígitos y caracteres especiales
    desc=re.sub("(\\d|\\W)+"," ",desc)
    
    clean_desc.append(desc)

# Asignar las descripciones limpias al marco de datos
df['Message'] = clean_desc
        
df.head(3)
```




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
      <th>Message</th>
      <th>Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>go until jurong point crazy available only in ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ok lar joking wif u oni</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>free entry in a wkly comp to win fa cup final ...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Construyamos una lista de palabras vacías desde cero y luego visualicemos nuestra nube de palabras.


```python
# Creando una lista de stop words (palabras vacías) desde cero

stop_words = ['is','you','your','and', 'the', 'to', 'from', 'or', 'I', 'for', 'do', 'get', 'not', 'here', 'in', 'im', 'have', 'on', 
're', 'new', 'subject']
```


```python
wordcloud = WordCloud(width = 800, height = 800, background_color = 'black', stopwords = stop_words, max_words = 1000
                      , min_font_size = 20).generate(str(df['Message']))
# Trazar la nube de palabras
fig = plt.figure(figsize = (8,8), facecolor = None)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()
```


    
![png](exploring-natural-language-processing.es_files/exploring-natural-language-processing.es_16_0.png)
    


Ahora que se han limpiado los mensajes, para evitar que la nube de palabras genere oraciones, los mensajes pasan por un proceso llamado tokenización. Es el proceso de dividir una oración en palabras individuales. Las palabras individuales se llaman tokens.

**Vectorizando**

Usando CountVectorizer() de SciKit-Learn, es fácil transformar el cuerpo del texto en una matriz dispersa de números que la computadora puede pasar a los algoritmos de Machine Learning. Vamos a crear la matriz dispersa, luego dividiremos los datos usando sk-learn train_test_split().


```python
# Crear matriz dispersa

message_vectorizer = CountVectorizer().fit_transform(df['Message'])

# Dividir los datos

X_train, X_test, y_train, y_test = train_test_split(message_vectorizer, df['Category'], test_size = 0.45, random_state = 42, shuffle = True)
```

**El algoritmo clasificador**

En este ejemplo, practicaremos el uso de **Support Vector Machine** (Máquina de vectores de soporte), pero siéntete libre de probar clasificadores adicionales.


```python
classifier = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
```

Ahora vamos a generar algunas predicciones.


```python
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
print(classification_report(y_test, predictions))
```

                  precision    recall  f1-score   support
    
               0       0.98      1.00      0.99      2024
               1       0.96      0.89      0.92       297
    
        accuracy                           0.98      2321
       macro avg       0.97      0.94      0.96      2321
    weighted avg       0.98      0.98      0.98      2321
    



```python
# Usa la función precision_score para obtener la precisión

print("SVM Accuracy Score -> ",accuracy_score(predictions, y_test)*100)
```

    SVM Accuracy Score ->  98.10426540284361


¡Nuestro modelo logró una precisión del 98%!

Fuente: 

https://towardsdatascience.com/3-super-simple-projects-to-learn-natural-language-processing-using-python-8ef74c757cd9

https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34

https://medium.com/analytics-vidhya/how-to-build-a-simple-sms-spam-filter-with-python-ee777240fc
