# Linear Algebra

Linear Algebra is the branch of mathematics concerning linear equations that uses vector space and matrices.
The two primary mathematical entities that are of interest in linear algebra are the vector and the matrix. They are examples of a more general entity known as a tensor. Tensors possess an order (or rank), which determines the number of dimensions in an array required to represent it.
Matrix: Array of numbers in a rectangular form represented by rows and columns.

Example: 

$$\begin{matrix} 3 & 4 \\ 5 & 6 \end{matrix}$$

Vectors: A vector is a row or a column of a matrix.

Example: 
		 
$$\begin{matrix} 3 \\ 4 \\ 5 \end{matrix}$$

Tensors:  Tensors are an array of numbers or functions that transmute with certain rules when coordinate changes.
Machine Learning is the point of contact for Computer Science and Statistics.
Numpy is a Python library that works with multidimensional arrays.
	

### Vectors

Given that scalars exist to represent values why are vectors necessary? One of the primary use cases for vectors is to represent physical quantities that have both a magnitude and a direction. Scalars are only capable of representing magnitudes.
For instance, scalars and vectors encode the difference between the speed of a car and its velocity. The velocity contains not only its speed but also its direction of travel. 
In machine learning vectors often represent feature vectors, with their individual components specifying how important a particular feature is. Such features could include relative importance of words in a text document, the intensity of a set of pixels in a two-dimensional image, or historical price values for a cross-section of financial instruments.

**Scalar operations**

Scalar operations involve a vector and a number



![linear_algebra_scalar_operations.png](../assets/linear_algebra_scalar_operations.png)

**Vector multiplication**

There are two types of vector multiplication: Dot product and Hadamard product.
The dot product of two vectors is a scalar. Dot product of vectors and matrices (matrix multiplication) is one of the most important operations in deep learning.


![linear_algebra_dot_product.png](../assets/linear_algebra_dot_product.png)

Hadamard Product is elementwise multiplication, and it outputs a vector.

![linear_algebra_hadamard_product.png](../assets/linear_algebra_hadamard_product.png)


### Matrices


A matrix is a rectangular grid of numbers or terms (like an Excel spreadsheet) with special rules for addition, subtraction, and multiplication.
Matrix dimensions: We describe the dimensions of a matrix in terms of rows by columns.

**Matrix scalar operations**

Scalar operations with matrices work the same way as they do for vectors. 
Simply apply the scalar to every element in the matrix — add, subtract, divide, multiply, etc.



![matrix_scalar_operations.png](../assets/matrix_scalar_operations.png)

**Matrix elementwise operations**

To add, subtract, or divide two matrices they must have equal dimensions. 
We combine corresponding values in an elementwise fashion to produce a new matrix.


![matrix_elementwise_operations.png](../assets/matrix_elementwise_operations.png)

**Matrix multiplication**

Hadamard product of matrices is an elementwise operation. 
Values that correspond positionally are multiplied to produce a new matrix.


![matrix_hadamard_operations.png](../assets/matrix_hadamard_operations.png)

*Rules*

The number of columns of the 1st matrix must equal the number of rows of the 2nd 
The product of an M x N matrix and an N x K matrix is an M x K matrix. 
The new matrix takes the rows of the 1st and columns of the 2nd

**Matrix transpose**

Matrix transpose provides a way to “rotate” one of the matrices so that the operation complies with multiplication requirements and can continue. 
There are two steps to transpose a matrix: 

1. Rotate the matrix right 90° 
2. Reverse the order of elements in each row (e.g. $[a b c]$ becomes $[c b a]$)

Numpy uses the function np.dot $(A,B)$ for both vector and matrix multiplication.


### Basic Numerical Analysis

The standard Python library for linear algebra is numpy. The basic object in numpy is ndarray, an n-dimensional generalization of Python's list.


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


The shape of an ndarray gives us the dimensions. b is a 1-by-4 matrix, or a row vector. c is a column vector. d is a 2-by-2 matrix.

Entering a column vector requires many brackets, which can be tedious. We can instead use **transpose**.


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


Similarly, a matrix can be entered by first typing out a one-dimensional array and then putting the array through **reshape**:


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


There are a number of pre-built functions for special kinds of vectors and matrices.


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
    


Matrix multiplications are performed via **dot**:


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


ndarray supports coordinatewise operations. Observe, in particular, that * does not result in matrix multiplication.


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


Operations with mismatching dimensions sometimes result in legitimate operation. For example, scalar multiplication works fine:


```python
3 * np.array([[2,4,3], [1,2,5], [-1, -1, -1]])
```




    array([[ 6, 12,  9],
           [ 3,  6, 15],
           [-3, -3, -3]])



Row-wise operations and column-wise operations are also possible:


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


### Applications of Linear Algebra in the Machine Learning process.

Linear Algebra is applicable in predictions, signal analysis, facial recognition, etc.

Some applications:

-	Data in a dataset is vectorized. Rows are inserted into a model one at a time for easier and authentic calculations.

-	All images are tabular in structure. It forms a matrix in Linear Algebra. Image editing techniques such as cropping and scaling use algebraic operations.

-	Regularization is a method that minimizes the size of coefficients while inserting it into data.

-	Deep learning work with vectors, matrices, and even tensors as it requires linear data structures added and multiplied together.

-	The ‘One hot encoding’ method encodes categories for easier algebra operations.

-	In linear regression, linear algebra is used to describe the relationship among variables.

-	When we find irrelevant data, we normally remove redundant columns, so PCA acts with matrix factorization.

-	With the help of linear algebra, recommender systems can have more purified data.


Source:

https://www.educba.com/linear-algebra-in-machine-learning/

https://en.wikipedia.org/wiki/Linear_algebra

https://www.quantstart.com/articles/scalars-vectors-matrices-and-tensors-linear-algebra-for-deep-learning-part-1/#:~:text=Vectors%20and%20Matrices%20The%20two%20primary%20mathematical%20entities,a%20more%20general%20entity%20known%20as%20a%20tensor.

https://github.com/markkm/math-for-ml

https://towardsdatascience.com/linear-algebra-cheat-sheet-for-deep-learning-cd67aba4526c

