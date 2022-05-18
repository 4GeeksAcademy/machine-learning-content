
# Algorithm Optimization problems

### Exercise 1

Efficiently compute sums of diagonals of a matrix.

For example:

```py
Input : 

4
1 2 3 4
4 3 2 1
7 8 9 6
6 5 4 3

Output :

Principal Diagonal: 16
Secondary Diagonal: 20
```

The primary diagonal is formed by the elements 1,3,9,3. 
 

Condition for Principal Diagonal: The row-column condition is row = column. 

The secondary diagonal is formed by the elements 4,2,8,6.

Condition for Secondary Diagonal: The row-column condition is row = numberOfRows – column -1.

Method 1 (O(n ^ 2) :

In this method, we use two loops i.e. a loop for columns and a loop for rows and in the inner loop we check for the condition stated above


```python
# A simple Python program to
# find sum of diagonals
MAX = 100
 
def printDiagonalSums(mat, n):
 
    principal = 0
    secondary = 0;
    for i in range(0, n):
        for j in range(0, n):
 
            # Condition for principal diagonal
            if (i == j):
                principal += mat[i][j]
 
            # Condition for secondary diagonal
            if ((i + j) == (n - 1)):
                secondary += mat[i][j]
                
    print("Principal Diagonal:", principal)
    print("Secondary Diagonal:", secondary)
 
# Driver code
a = [[ 1, 2, 3, 4 ],
     [ 5, 6, 7, 8 ],
     [ 1, 2, 3, 4 ],
      [ 5, 6, 7, 8 ]]
printDiagonalSums(a, 4)
 
# This code is contributed
# by ihritik
```

Method 2 (O(n) :

In this method we use one loop i.e. a loop for calculating sum of both the principal and secondary diagonals: 


```python
# A simple Python3 program to find
# sum of diagonals
MAX = 100
 
def printDiagonalSums(mat, n):
 
    principal = 0
    secondary = 0
    for i in range(0, n):
        principal += mat[i][i]
        secondary += mat[i][n - i - 1]
         
    print("Principal Diagonal:", principal)
    print("Secondary Diagonal:", secondary)
 
# Driver code
a = [[ 1, 2, 3, 4 ],
     [ 5, 6, 7, 8 ],
     [ 1, 2, 3, 4 ],
     [ 5, 6, 7, 8 ]]
printDiagonalSums(a, 4)
 
# This code is contributed
# by ihritik
```

Source:

https://www.geeksforgeeks.org/
