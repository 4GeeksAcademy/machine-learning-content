# Web scraping problem

### Exercise 1

In this project, we are going to scrape the stock values of Twitter,Inc company from yahoo finance, that is present in the tabular form, creating a database in SQLite3, and storing the scraped data in it. 

**Step 1:** Make sure you have sqlite3 and pandas installed. In case they are not installed, you can use the following command in the terminal:

```py
pip install pandas
```

and 

```py
pip install sqlite3
```

**Step 2:** Import pandas  and sqlite3

If you are using a notebook, run the cell. If you are using a text editor make sure to use print to execute the code.


```python
# import packages

import pandas as pd
import sqlite3
```

**Step 3:** Scraping the Yahoo finance website


```python
url = 'https://finance.yahoo.com/quote/TWTR/history?p=TWTR'

data = pd.read_html(url, match = 'Date')

data
```

It is not in a tabular way so we will do a small data cleaning operation:


```python
# small data cleaning
df = data[0].iloc[:100,:]
df
```


```python
# Let's make sure is a dataframe
type(df)
```


```python
# to insert the data into sqlite3 we will convert the dataframe into a list of tuples

records = df.to_records(index=False)
list_of_tuples = list(records)
list_of_tuples
```

Now let's create our SQLite3 database. The following command is to connect to a Sqlite3 database. In case the databse does not exist, it will create it.


```python
# Use the connect() function of sqlite3 to create a database. It will create a connection object.

connection = sqlite3.connect('twitter_stocks.db')
```

Now let's create a table in our database to store our stock values.


```python
c = conn.cursor()

# Create table
c.execute('''CREATE TABLE stocks
             (Date, Open, High, Low, Close, Adjclose, Volume)''')
```


```python
# Insert the values
c.executemany('INSERT INTO teslastocks VALUES (?,?,?,?,?,?,?)', list_of_tuples)
# Save (commit) the changes
conn.commit()
```

Now retrieve the data from the database


```python
for row in c.execute('SELECT * FROM stocks'):
    print(row)
```

Our database name is “twitter_stocks.db”. We saved the connection to the connection object.

Next time we run this file, it just connects to the database, and if the database is not there, it will create one.

Source:

https://github.com/bhavyaramgiri/Web-Scraping-and-sqlite3/blob/master/week%209-%20web%20scraping%20sqlite.ipynb

https://coderspacket.com/scraping-the-web-page-and-storing-it-in-a-sqlite3-database
