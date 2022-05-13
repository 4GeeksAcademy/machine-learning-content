
# Linear Algebra

Linear Algebra is the branch of mathematics concerning linear equations that uses vector space and matrices.
The two primary mathematical entities that are of interest in linear algebra are the vector and the matrix. They are examples of a more general entity known as a tensor. Tensors possess an order (or rank), which determines the number of dimensions in an array required to represent it.
Matrix: Array of numbers in a rectangular form represented by rows and columns.

Example: 

3 4

5 6

Vectors: A vector is a row or a column of a matrix.

Example: 
		 
3

4
                
5

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
2. Reverse the order of elements in each row (e.g. [a b c] becomes [c b a])

Numpy uses the function np.dot(A,B) for both vector and matrix multiplication.


### Applications of Linear Algebra in the Machine Learning process.

Linear Algebra is applicable in predictions, signal analysis, facial recognition, etc.

Some applications:
•	Data in a dataset is vectorized. Rows are inserted into a model one at a time for easier and authentic calculations.
•	All images are tabular in structure. It forms a matrix in Linear Algebra. Image editing techniques such as cropping and scaling use algebraic operations.
•	Regularization is a method that minimizes the size of coefficients while inserting it into data.
•	Deep learning work with vectors, matrices, and even tensors as it requires linear data structures added and multiplied together.
•	The ‘One hot encoding’ method encodes categories for easier algebra operations.
•	In linear regression, linear algebra is used to describe the relationship among variables.
•	When we find irrelevant data, we normally remove redundant columns, so PCA acts with matrix factorization.
•	With the help of linear algebra, recommender systems can have more purified data.



References:

https://www.educba.com/linear-algebra-in-machine-learning/

https://en.wikipedia.org/wiki/Linear_algebra

https://www.quantstart.com/articles/scalars-vectors-matrices-and-tensors-linear-algebra-for-deep-learning-part-1/#:~:text=Vectors%20and%20Matrices%20The%20two%20primary%20mathematical%20entities,a%20more%20general%20entity%20known%20as%20a%20tensor.

https://towardsdatascience.com/linear-algebra-cheat-sheet-for-deep-learning-cd67aba4526c

