
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

# Connecting to a SQLite database


```python

from sqlalchemy import create_engine
import pandas as pd
 
# Create engine: engine
engine = create_engine('sqlite:///databse_name.sqlite')
 
# Save the table names to a list: table_names
table_names = engine.table_names()
 
# Print the table names to the shell
print(table_names)
 
# Open engine connection: con,  and select specified columns and number of rows

with engine.connect() as con:
    ab = con.execute("SELECT ID, Date, Name FROM Clients")
    df = pd.DataFrame(ab.fetchmany(size=5))
    df.columns = ab.keys()

# Close connection
con.close()
 
# Print first rows of dataframe
print(df.head())
```

# Connecting to a DB2 Database

To connect to a DB2, it requires the following information:

- controler name
- database name
- DNS host name or IP
- Host port
- Connection protocole
- User id
- Password

Example to create a databse connection:


```python
#Create database connection

dsn = (
    "Driver = { {IBM DB2 ODBC DRIVER}};"
    "Database = {0};"
    "HOSTNAME = {1};"
    "PORT = {2};"
    "PROTOCOL = TCPIP;"
    "UID = {3};"
    "PWD = {4};").format(dsn_database, dsn_hostname, dsn_port, dsn_uid, dsn_pwd)

try: 
    conn = ibm_db.connect(dsn, " ", " ")
    print("Connected!")
    
except:
    print("Unable to connect to database")
    
#Close the database connection

ibm_db.close(conn)

#Note: It is always important to close the connections to avoid non used connectors taking resources.
```

# How to create a table from python

ibm_db.exec_inmediate()  --> function of the ibm_db API

Parameters for the function:

- connection
- statement
- options

Example: Creating a table called CARS in Python


```python
#CREATE TABLE

stmt = ibm_db.exec_inmediate(conn, "CREATE TABLE Cars(
    serial_no VARCHAR(20) PRIMARY KEY NOT NULL,
    make varchar(20) NOT NULL,
    model VARCHAR(20) NOT NULL,
    car_class VARCHAR(20) NOT NULL)"
    )

#LOAD DATA IN TABLE

stmt = ibm_db.exec_inmediate(conn, "INSERT INTO Cars(
    serial_no, make, model, car_class)
    VALUES('A2345453','Ford','Mustang','class3');")

#FETCH DATA FROM CARS TABLE

stmt = ibm_db.exec_inmediate(conn, "SELECT * FROM Cars")

ibm_db.fetch_both(stmt)


```

# Using pandas to retrieve data from the tables



```python
import pandas
import ibm_db_dbi
pconn = ibm_db_dbi.connection(conn)

df = pandas.read_sql('SELECT * FROM Cars', pconn)
df

#Example of a plot

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns 

#categorical scatterplot

plot = sns.swarmplot(x="Category", y="Calcium", data=df)
plt.setp(plot.get_xticklabels(), rotation=70)
plt.title('Calcium content')
plt.show()

#Making a boxplot
#A boxplot is a graph that indicates the distribution of 1 or more variables. The box captures the median 50% of the data.
# The line and dots indicate possible outliers and not normal values.

plot = sns.set_style('Whitegrid')
ax = sns.boxplot(x=df['glucose level'])
plt.show()
```

# Getting the properties

DB2 --->  syscat.tables                                 

 SQL Server --->  information=schema.tables   
 
Oracle --->  all_tables or user_tables



```python
#Getting table properties from DB2

SELECT * FROM syscat.tables
#(this will show too many tables)

SELECT tabschema, tabname, create_time
FROM syscat.tables
WHERE tabschema = 'ABC12345' #---> replace with your own DB2 username

#Getting a list of columns in database

SELECT * FROM syscat.columns
WHERE tabname = 'Cats'

#To obtain specific column properties:

%sql SELECT DISTINCT(name), coltype, length
    FROM sysibm.syscolumns
    WHERE tbname = 'Cats'
    
%sql SELECT DISTINCT(name), coltype, length
    FROM sysibm.syscolumns 
    WHERE tbname = 'Miami_crime_data'
```
