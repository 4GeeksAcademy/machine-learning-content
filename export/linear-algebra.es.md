# Álgebra Lineal

El álgebra lineal es la rama de la matemática sobre ecuaciones lineales que utiliza espacio vectorial y matrices.

Las dos entidades matemáticas primarias que son de interés en álgebra lineal son el vector y la matriz. Son ejemplos de una entidad más general conocida como tensor. Los tensores poseen un orden (o rango), que determina el número de dimensiones en una matriz requerida para representarlo.

Matriz: Conjunto de números en forma rectangular representados por filas y columnas.

Ejemplo:

$$\begin{matrix} 3 & 4 \\ 5 & 6 \end{matrix}$$

Vectores: Un vector es una fila o una columna de una matriz.

Ejemplo: 
		 
$$\begin{matrix} 3 \\ 4 \\ 5 \end{matrix}$$

Tensores: Los tensores son una matriz de números o funciones que se transmutan con ciertas reglas cuando cambian las coordenadas.

Machine Learning es el punto de contacto para Ciencias Informáticas y Estadística. 

Numpy es una biblioteca de Python que funciona con arreglos multidimensionales.
	

### Vectores

Dado que los escalares existen para representar valores, ¿por qué son necesarios los vectores? Uno de los principales casos de uso de los vectores es representar cantidades físicas que tienen tanto una magnitud como una dirección. Los escalares solo son capaces de representar magnitudes.

Por ejemplo, los escalares y los vectores codifican la diferencia entre la rapidez de un automóvil y su velocidad. La velocidad contiene no solo su rapidez, sino que también su dirección de viaje.

En Machine Learning los vectores a menudo representan vectores de características, con sus componentes individuales especificando qué tan importante es una característica en particular. Tales características podrían incluir la importancia relativa de las palabras en un documento de texto, la intensidad de un conjunto de píxeles en una imagen bidimensional o valores de precios históricos para una muestra representativa de instrumentos financieros.

**Operaciones Escalares**

Las operaciones escalares involucran un vector y un número


![linear_algebra_scalar_operations.png](../assets/linear_algebra_scalar_operations.png)

**Multiplicación de vectores**

Hay dos tipos de multiplicación vectorial: producto punto y producto de Hadamard.

El producto punto de dos vectores es un escalar. El producto punto de vectores y matrices (multiplicación de matrices) es una de las operaciones más importantes en el aprendizaje profundo.

![linear_algebra_dot_product.png](../assets/linear_algebra_dot_product.png)

El producto de Hadamard es una multiplicación por elementos y genera un vector.

![linear_algebra_hadamard_product.png](../assets/linear_algebra_hadamard_product.png)


### Matrices

Una matriz es una cuadrícula rectangular de números o términos (como una hoja de cálculo de Excel) con reglas especiales para sumar, restar y multiplicar.

Dimensiones de la matriz: Describimos las dimensiones de una matriz en términos de filas por columnas.

**Operaciones escalares de matrices**

Las operaciones escalares con matrices funcionan de la misma manera que con los vectores.

Simplemente aplica el escalar a cada elemento de la matriz — suma, resta, divide, multiplica, etc.


![matrix_scalar_operations.png](../assets/matrix_scalar_operations.png)

**Operaciones con elementos matriciales**

Para sumar, restar o dividir dos matrices deben tener las mismas dimensiones. Combinamos los valores correspondientes de forma elemental para producir una nueva matriz.

![matrix_elementwise_operations.png](../assets/matrix_elementwise_operations.png)

**Multiplicación de matrices**

El producto de Hadamard de matrices es una operación elemental.
Los valores que corresponden posicionalmente se multiplican para producir una nueva matriz.

![matrix_hadamard_operations.png](../assets/matrix_hadamard_operations.png)

*Normas*

El número de columnas de la primera matriz debe ser igual al número de filas de la segunda.
El producto de una matriz M x N y una matriz N x K es una matriz M x K.
La nueva matriz toma las filas de la 1ra y las columnas de la 2da.

**Transposición de matriz**

La transposición de matriz proporciona una forma de "rotar" una de las matrices para que la operación cumpla con los requisitos de multiplicación y pueda continuar.

Hay dos pasos para transponer una matriz:

1. Rotar la matriz 90° a la derecha
2. Invier el orden de los elementos en cada fila (por ejemplo, $[a b c]$ se convierte en $[c b a]$)

Numpy usa la función np.dot $(A,B)$ para la multiplicación de vectores y matrices.


### Análisis Numérico Básico

La biblioteca estándar de Python para álgebra lineal es Numpy. El objeto básico en Numpy es ndarray, una generalización n-dimensional de la lista de Python.


```python
import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([[1, 2, 3, 4]])
c = np.array([[1], [2], [3], [4]])
d = np.array([[1, 2], [3, 4]])

print(a)
print('shape of a: {}'.format(a.shape))
print()

print(b)
print('shape of b: {}'.format(b.shape))
print()

print(c)
print('shape of c: {}'.format(c.shape))
print()

print(d)
print('shape of d: {}'.format(d.shape))
```

    [1 2 3 4]
    shape of a: (4,)
    
    [[1 2 3 4]]
    shape of b: (1, 4)
    
    [[1]
     [2]
     [3]
     [4]]
    shape of c: (4, 1)
    
    [[1 2]
     [3 4]]
    shape of d: (2, 2)


La forma de un ndarray nos da las dimensiones. b es una matriz de 1 por 4, o un vector fila. c es un vector columna. d es una matriz de 2 por 2.

Ingresar un vector de columna requiere muchos corchetes, lo que puede ser tedioso. En su lugar, podemos usar **transpose** (transponer).


```python
print(b)
print('shape of b: {}'.format(b.shape))
print()

print(b.transpose())
print('shape of b.transpose(): {}'.format(b.transpose().shape))
```

    [[1 2 3 4]]
    shape of b: (1, 4)
    
    [[1]
     [2]
     [3]
     [4]]
    shape of b.transpose(): (4, 1)


De manera similar, se puede ingresar una matriz escribiendo primero un arreglo unidimensional y luego poniendo el arreglo a través de **reshape** (reformar):


```python
print(b)
print('shape of b: {}'.format(b.shape))
print()

print(b.reshape((2,2)))
print('shape of b.reshape((2,2)): {}'.format(b.transpose().reshape((2,2))))
print()

print(b.reshape((4,1)))
print('shape of b.reshape((4,1)): {}'.format(b.transpose().reshape((4,1))))

```

    [[1 2 3 4]]
    shape of b: (1, 4)
    
    [[1 2]
     [3 4]]
    shape of b.reshape((2,2)): [[1 2]
     [3 4]]
    
    [[1]
     [2]
     [3]
     [4]]
    shape of b.reshape((4,1)): [[1]
     [2]
     [3]
     [4]]


Hay una serie de funciones preconstruidas para tipos especiales de vectores y matrices.


```python
print(np.arange(5))
print()

print(np.arange(2, 8))
print()

print(np.arange(2, 15, 3))
print()

print(np.eye(1))
print()

print(np.eye(2))
print()

print(np.eye(3))
print()

print(np.zeros(1))
print()

print(np.zeros(2))
print()

print(np.zeros(3))
print()
```

    [0 1 2 3 4]
    
    [2 3 4 5 6 7]
    
    [ 2  5  8 11 14]
    
    [[1.]]
    
    [[1. 0.]
     [0. 1.]]
    
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    
    [0.]
    
    [0. 0.]
    
    [0. 0. 0.]
    


Las multiplicaciones de matrices se realizan mediante **dot** (punto):


```python
x = np.array([[2, 3], [5, 7]])
y = np.array([[1, -1], [-1, 1]])

print(np.dot(x,y))
print()
print(x.dot(y))
```

    [[-1  1]
     [-2  2]]
    
    [[-1  1]
     [-2  2]]


Ndarray admite operaciones coordinadas. Observa en particular, que * no da como resultado una multiplicación de matrices.


```python
print(np.array([1, 2, 3]) + np.array([3,4,5]))
print()

print(np.array([[4,3],[2,1]]) - np.array([[1,1],[1,1]]))
print()

print(np.array([[1, 2, 3], [4, 5, 6]]) * np.array([[1, 2, 1], [3, -1, -1]]))
print()

print(np.array([[1], [3]]) / np.array([[2], [2]]))
```

    [4 6 8]
    
    [[3 2]
     [1 0]]
    
    [[ 1  4  3]
     [12 -5 -6]]
    
    [[0.5]
     [1.5]]


Las operaciones con dimensiones que no coinciden a veces dan como resultado una operación legítima. Por ejemplo, la multiplicación escalar funciona bien:


```python
3 * np.array([[2,4,3], [1,2,5], [-1, -1, -1]])
```




    array([[ 6, 12,  9],
           [ 3,  6, 15],
           [-3, -3, -3]])



Las operaciones por filas y las operaciones por columnas también son posibles:


```python
x = np.array([5, -1, 3])
y = np.arange(9).reshape((3, 3))

print(y)
print()

print(x.reshape((3, 1)) + y)
print()

print(x.reshape((1, 3)) + y)
```

    [[0 1 2]
     [3 4 5]
     [6 7 8]]
    
    [[ 5  6  7]
     [ 2  3  4]
     [ 9 10 11]]
    
    [[ 5  0  5]
     [ 8  3  8]
     [11  6 11]]


### Aplicaciones del Álgebra Lineal en el proceso de Machine Learning.

El Álgebra Lineal es aplicable en predicciones, análisis de señales, reconocimiento facial, etc.

Algunas aplicaciones:

-	Los datos en un Dataset (conjunto de datos) están vectorizados. Las filas se insertan en un modelo una a la vez para realizar cálculos más fáciles y auténticos.

-	Todas las imágenes tienen una estructura tabular. Forma una matriz en Álgebra Lineal. Las técnicas de edición de imágenes, como recortar y escalar, utilizan operaciones algebraicas.

-	La regularización es un método que minimiza el tamaño de los coeficientes mientras los inserta en los datos.

-	El aprendizaje profundo funciona con vectores, matrices e incluso tensores, ya que requiere estructuras de datos lineales agregadas y multiplicadas.

-	El método 'One hot encoding' codifica categorías para facilitar las operaciones de álgebra.

-	En la regresión lineal, se utiliza el álgebra lineal para describir la relación entre las variables.

-	Cuando encontramos datos irrelevantes, normalmente eliminamos las columnas redundantes, por lo que PCA actúa con la factorización de matrices.

-	Con la ayuda del álgebra lineal, los sistemas de recomendación pueden tener datos más depurados.

Fuente:

https://www.educba.com/linear-algebra-in-machine-learning/

https://en.wikipedia.org/wiki/Linear_algebra

https://www.quantstart.com/articles/scalars-vectors-matrices-and-tensors-linear-algebra-for-deep-learning-part-1/#:~:text=Vectors%20and%20Matrices%20The%20two%20primary%20mathematical%20entities,a%20more%20general%20entity%20known%20as%20a%20tensor.

https://github.com/markkm/math-for-ml

https://towardsdatascience.com/linear-algebra-cheat-sheet-for-deep-learning-cd67aba4526c

