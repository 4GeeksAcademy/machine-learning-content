
# ALGORITHMS AND DATA STRUCTURES

## What is an algorithm?

Algorithms are more simple than we think. Since we are kids we are taught how to complete day to day tasks.
An algorithm is a set of instructions followed to achieve a goal or to produce an output, for example learning to walk, tie your shoes or preparing a cake. All these processes are taught to us using a step to step procedure.
Let's see an example of a very simple algorithm, the steps to prepare some brownies:


```python
function PrepareBrownies(flavor){
"
 1. Heat Oven to 350 F
 2. Mix flour, baking powder, salt in a bowl
 3. Mix melted butter with "+ flavor +" chips in another bowl
 4. Mix eggs with sugar in a different bowl
 5. First combine the butter bowl with the eggs bowl.
 6. Add the flour bowl to the previous mix.
 7  Mix. 
 8. Put in pan
 9. Bake for 30 minutes
"
}

PrepareBrownies('chocolate')
```


      Input In [1]
        function PrepareBrownies(flavor){
                 ^
    SyntaxError: invalid syntax



## Data Structures

Most of the problems in computer science have some sort of data associated with, which we have to use to solve the problem and come up with a conclusion. So, a data structure is  a way to organize and manage that data at memory level such that we can effectively and efficiently do operation on that data.

To have a clear difference between algorithms and data structures we can say that the algorithms would act like the verbs, and data structures would be the nouns.

## What is time complexity?

A simple problem can be solved using many different algorithms. Some solutions just take less time and space than others.
But how do we know which solutions are more efficient?

Time complexity in programming it's most commonly used in the design of algorithms. It means how long an algorithm with a given number of inputs (n) will take to complete its task. It is usually defined using Big-O notation. Analyze time complexity all times you try to solve a problem. It will make you a better developer.

Let's look at some time complexities and its definitions:

**O(1) - Constant time:** Given an input of size n, it only takes a single step for the algorithm to accomplish the task.

**O(log n) - Logarithmic time:** given an input of size n, the number of steps it takes to accomplish the task are decreased by some factor with each step.

**O(n) - Linear time:** Given an input of size n, the number of steps required is directly related (1 to 1).

**O(n^2) - Quadratic time:** Given an input of size n, the number of steps it takes to accomplish a task is square of n.

**O(C^n) - Exponential time:** Given an input of size n, the number of steps it takes to accomplish a task is a constant to the n power (pretty large number).


 

Let's use n=16 to understand time complexity with an example.


```python
let n = 16:
    
    
O(1) = 1 step

O(log n) = 4 steps -- assuming base 2

O(n) = 16 steps

O(n^2) = 256 steps

O(2^n) = 65,356 steps
```

![big-o%20complexity.png](attachment:big-o%20complexity.png)

## Understanding algorithms and data structures

### 1. First algorithm:  Binary Search 

Suppose you’re searching for a word in a dictionary, and it starts with K. You could start at the beginning and keep flipping pages until you get to the Ks. But you’re more likely to start at a page in the middle, because you know the Ks are going to be near the middle of the dictionary. This is a search problem. And all these cases use the same algorithm to solve the problem: binary search.

For any list of n, binary search will take log2n steps to run in the worst case.

Retunring to our initial example we can say that:

To look in a 240,000 words dictionary with binary search, you cut the number of words in half until you are left with one word. The result at the worst case would be 18 steps, while a simple search would have taken 240,000 steps.

**Time complexity:** 

-A normal search runs in linear time O(n).

-A binary search runs in log time (logarithmic time) O(log n).

### 2. Second Algorithm: Selection Sort

#### Arrays vs LinkedList  (Data structures)

Sometimes you need to store a list of elements in memory. Suppose you have a list of tasks to do today. Let's store it in an array first. That means all your tasks will be stored right next to each other in memory.

10 is at index 2

![arrays.png](attachment:arrays.png)  

Now let's suppose after adding the first four tasks, you want to add a fifth one but the next space is occupied. With linked lists your tasks can be anywhere in memory. 

Elements aren't next to each other so you can't instantly calculate the position, you have to go to the first element to get the adress of the second element. Then go to the second element to get the adress of the third element. The blue squares are memory in use by someone else, you can't add data there because it is already occupied.

![LinkedList.png](attachment:LinkedList.png)

Suppose you do your spend list of the month . At the end of the month you check how much you spent in that month. 
That means you are having a lot of Inserts and a few Reads. Should you use an array or a list?

-Arrays  have fast reads and slow inserts.

-Linked Lists have slow reads and fast inserts.

It would make more sense to use a Linked List because you will be inserting more often than reading your spend list. Also, they only have slow reads if you are accesing random elements of the list but you are accesing all. Another important fact is that to add an item in the middle of an ordered list, also linked lists will be better because you just have to change what the previous element points to.

What if I want to delete an element from the list?

Again, lists are better because you just need to change what the previous element points to.


Which are used more? It depends on the case!

-Arrays are more used because they allow random access.

-Linked list can only do sequential access (reading elements one by one), but they are good for inserts/deletes

When you want to store multiple elements, should you use an array or a list?

-In an array the elements would be stored right next to each other. It allows fast reads.
*All elements in the array should be the same type.*

-In a list, elements would be stores all over, one element stores the adress of the next one. It allows fast insert and deletes.



**Time complexity:** 

**Best practice: Keep track of first and last items in a linked list so it only takes O(1)**

![arrays%20and%20linkedlist%20time%20complexity.png](attachment:arrays%20and%20linkedlist%20time%20complexity.png)

#### Selection Sort Algorithm

Remember O(n) time means you touch every element in a list once.

Example: 

You have a list of songs with the count of times you have played them. To sort the list of songs and see which one is your favorite you have to check each item in the list to find the one with the highest play count.
This takes O(n) time  and you have to do it n times.


```python
#Find the smallest element in an array and use it to write selection sort

def findsmallest(arr):
    smallest = arr[0] ----> stores smallest value
    smallest_index = 0 -----> stores index of the smallest value
    for i in range(1, len(arr)):
        if arr[i] < smallest:
            smallest = arr[i]
            smallest_index = i
    return smallest_index

#Now let's write the Selection Sort:

def SelectionSort(arr):   ------> sorts an array
    newArr = []
    for i in range(len(arr)):
        smallest = findSmallest(arr)   ----->finds the smallest and adds it to the new array.
        newArr.append(arr.pop(smallest))
    return newArr
```

### 3. Recursion

Recursion means breaking down a problem into a base case and a recursive case.

Recursion is where a function calls itself. Because a recursive function calls itself, it is easy to write a function incorrectly that ends up in an infinite loop.
If that happens you need to press **CTRL + C** to kill your script.
That is why every recursive function has 2 parts:

1. Base case (when it does not call itself again)
2. Recursive case (when it calls itself)

There is no performance benefit of using recursion, in fact, sometimes loops are better.

"Loops may achieve a performance gain for your program. Recursion may achieve a performance gain for the programmer"

#### The stack (data structure)

The stack is a simple data structure.
Remember when we talked about arrays and lists? We had a task list. We could add task items anywhere in the list or delete random items. Now, let's imagine we have that same task list but now in the form of a stack of sticky notes. This is much simpler because when we insert an item, it gets added to the top of the list. When we read an item, we only read the most recent  top item and it's taken off the list. Now our task list only has two actions: 

1. Push (insert, add a new item to the top)
2. Pop (remove the topmost item and read it)

In our computer, all the function calls go into the call stack.
The call stack can get very large, which takes up a lot of memory.

### 4. Third Algorithm: QuickSort

Divide and conquer, a well known recursive technique for solving problems.



### 5. Hash Tables (Data structure)

A hash function is where you put in a string and you get a number.
They are great when you want to create a mapping from one thing to another thing or when you want to look something up.

### 6. Breadth-first Search

Bread-first search tells you if there is a path from A to B, and if there is a path, bread-first will find the shortest path.

### 7. Dijkstra 's algorithm

### 8. Greedy Algorithm

How to tackle the impossible: problems that have no fast algorithmic solution. 
A greedy algorithm is simple. At each step, pick the optimal move.

### 9. Dynamic Programming

It is useful when you are trying to optimize something given a constraint. You can use dynamic programming when the problem can  be broken into discrete subproblems, and they don't depend on each other.

### 10. K-Nearest Neighbors (KNN)

It's simple but useful. If you are trying to classify something, you might want to try KNN first.

Source:
    
https://www.freecodecamp.org/news/time-is-complex-but-priceless-f0abd015063c/

https://www.bigocheatsheet.com/

Book: Grooking Algorithms
