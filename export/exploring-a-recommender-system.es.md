# Explorando un sistema de recomendación

Un sistema de recomendación (también conocido comúnmente como motor/plataforma de recomendación/recomendador) busca predecir el interés de un usuario en los elementos disponibles (canciones en Spotify, por ejemplo) y dar recomendaciones en consecuencia.

```python
import pandas as pd
import numpy as np

# read data
df_movies = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/machine-learning-content/master/assets/movies.csv",
    usecols=['movieId', 'title'],
    dtype={'movieId': 'int32', 'title': 'str'})

df_ratings = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/machine-learning-content/master/assets/ratings.csv",
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
```


```python
#Look at first rows of movies dataset
df_movies.head()
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
      <th>movieId</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Toy Story (1995)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Jumanji (1995)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Grumpier Old Men (1995)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Waiting to Exhale (1995)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Father of the Bride Part II (1995)</td>
    </tr>
  </tbody>
</table>
</div>

```python
df_ratings.head()
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
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>3</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>6</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>47</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>50</td>
      <td>5.0</td>
    </tr>
  </tbody>
</table>
</div>

```python

```

Entonces, aproximadamente el 1% de las películas tienen aproximadamente 97,999 o más calificaciones, el 5% tiene 1,855 o más y el 20% tiene 100 o más. Como tenemos tantas películas, lo limitaremos al 25% superior. Este es un umbral arbitrario de popularidad, pero nos da unas 13.500 películas diferentes. Todavía tenemos una buena cantidad de películas para modelar. Hay dos razones por las que queremos filtrar aproximadamente 13 500 películas en nuestro conjunto de datos.

Problema de memoria: no queremos encontrarnos con el "MemoryError" durante el entrenamiento del modelo

Mejora el rendimiento de KNN: las películas menos conocidas tienen calificaciones de menos espectadores, lo que hace que el patrón sea más ruidoso. Descartar películas menos conocidas puede mejorar la calidad de las recomendaciones


```python
# filter data
popularity_thres = 50
popular_movies = list(set(df_movies_cnt.query('count >= @popularity_thres').index))
df_ratings_drop_movies = df_ratings[df_ratings.movieId.isin(popular_movies)]
print('shape of original ratings data: ', df_ratings.shape)
print('shape of ratings data after dropping unpopular movies: ', df_ratings_drop_movies.shape)
```

Después de descartar el 75 % de las películas en nuestro conjunto de datos, todavía tenemos un conjunto de datos muy grande. A continuación, podemos filtrar usuarios para reducir aún más el tamaño de los datos.

```python
# get number of ratings given by every user
df_users_cnt = pd.DataFrame(df_ratings_drop_movies.groupby('userId').size(), columns=['count'])
df_users_cnt.head()
```

Fuente:

https://towardsdatascience.com/prototyping-a-recommender-system-step-by-step-part-1-knn-item-based-collaborative-filtering-637969614ea

https://towardsdatascience.com/beginners-recommendation-systems-with-python-ee1b08d2efb6
