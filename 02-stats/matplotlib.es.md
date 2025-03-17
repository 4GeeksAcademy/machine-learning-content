---
title: Matplotlib para Ciencia de Datos
tags:
  - visualización de datos
  - matplotlib
  - python
  - ciencia de datos
description: >-
  Aprende los fundamentos de Matplotlib y domina la visualización de datos en Python.
  Descubre cómo crear, personalizar y analizar datos visualmente paso a paso.
---



# Matplotlib para Data Science

La visualización de datos es una de las herramientas más poderosas en Data Science. Nos permite explorar, analizar y comunicar información de manera efectiva. En este artículo, aprenderemos a utilizar Matplotlib, una de las bibliotecas más populares de Python para crear gráficos y representar datos de manera visual.

## Fundamentos de Matplotlib

### ¿Qué es Matplotlib?

Matplotlib es una biblioteca de visualización en Python que permite crear una amplia variedad de gráficos, desde simples hasta altamente personalizados. Su módulo principal es pyplot, que proporciona una interfaz similar a la de MATLAB para generar visualizaciones con facilidad.

#### Instalación de Matplotlib

Si aún no tienes instalada la librería, puedes hacerlo con el siguiente comando:

```bash
pip install matplotlib
```

Después de la instalación, estás listo para comenzar a graficar.

### Diferencia entre pyplot y figure/axes

`Matplotlib` ofrece dos enfoques principales para generar gráficos:

- **pyplot:** Es una interfaz fácil de usar que permite crear gráficos rápidamente, ideal para principiantes.

- **Figure/Axes (Enfoque orientado a objetos):** Proporciona mayor control sobre la personalización de los gráficos.


Veamos la utilización de pyplot para crear graficos básicos. Para comenzar, importamos la biblioteca principal:

```python
import matplotlib.pyplot as plt
```

## Crear un gráfico de líneas básico

Veamos cómo podemos generar un gráfico de líneas con Matplotlib:

```python
import numpy as np
import matplotlib.pyplot as plt

# Datos
x = np.linspace(0, 10, 100)  # Generamos 100 puntos entre 0 y 10
y = np.sin(x)  # Calculamos el seno de cada punto

# Crear la figura
plt.plot(x, y, label='Seno de x', color='b', linestyle='--')
plt.xlabel('Valores de X')
plt.ylabel('Valores de Y')
plt.title('Gráfico de una función seno')
plt.legend()
plt.show()
```
Este código genera una gráfica de la función seno, mostrando los ejes etiquetados, un título y una leyenda, como lo muestra la siguiente imagen.


![image1](/assets/plot-function.png)


## Subgráficos (Múltiples Gráficas en una Figura)

En muchas ocasiones, necesitamos mostrar varias gráficas en la misma figura. Para ello, usamos `subplot` o `subplots`.

### Usando `plt.subplot()`

Este método permite crear subgráficos en una cuadrícula definida por filas y columnas:

```python
plt.subplot(2, 1, 1)  # (filas, columnas, índice)
plt.plot(x, np.sin(x), 'r')
plt.title('Seno')

plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x), 'b')
plt.title('Coseno')

plt.tight_layout()  # Ajusta los espacios entre subgráficos
plt.show()
```

### Usando `plt.subplots()`

Este método proporciona mayor flexibilidad y control sobre los subgráficos:

```python
fig, ax = plt.subplots(2, 1, figsize=(6, 6))  # 2 filas, 1 columna
ax[0].plot(x, np.sin(x), 'r')
ax[0].set_title('Seno')
ax[1].plot(x, np.cos(x), 'b')
ax[1].set_title('Coseno')
plt.tight_layout()
plt.show()
```

![image2](/assets/grafico_subgraficos.png)

## Enfoque Orientado a Objetos

El enfoque orientado a objetos de Matplotlib proporciona mayor control sobre las visualizaciones.

### Diferencia con pyplot

En lugar de usar `plt.plot()`, el enfoque basado en `Figure` y `Axes` permite personalizar los gráficos con mayor precisión.

Ejemplo:

```python
fig, ax = plt.subplots()
ax.plot(x, y, color='g', linestyle='-.', linewidth=2)
ax.set_title('Ejemplo con Enfoque Orientado a Objetos')
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.grid(True)
plt.show()
```

![image3](/assets/plot-oo.png)

## Trabajar con Datos Reales

Matplotlib es muy útil para visualizar datos provenientes de archivos como CSV. Podemos combinarlo con pandas para cargar y graficar datos reales.

### Cargar y visualizar datos desde un CSV

```python
import pandas as pd

df = pd.read_csv('data.csv')  # Cargar un archivo CSV

plt.plot(df['Fecha'], df['Ventas'], marker='o', linestyle='-')
plt.xlabel('Fecha')
plt.ylabel('Ventas')
plt.title('Ventas a lo largo del tiempo')
plt.xticks(rotation=45)
plt.show()
```

### Integración con Pandas

Podemos usar `df.plot()` en lugar de `plt.plot()` para visualizaciones rápidas:

```python
df.plot(x='Fecha', y='Ventas', kind='line')
plt.title('Gráfico generado con Pandas')
plt.show()
```
