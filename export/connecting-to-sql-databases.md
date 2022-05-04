
# How to connect to a SQL database using Python in Jupyter


```python
from dbmodule import connect 

#Create a connection object

CONNECTION = CONNECT('databse name', 'username','password')

#Create a cursor object

CURSOR = CONNECTION.CURSOR()

#Run queries

CURSOR.EXECUTE('select * from mytable')
RESULTS = CURSOR.FETCHALL()

#Free resources
CURSOR.CLOSE()
```

# CONNECTING TO A DB2 DATABASE


```python

```
