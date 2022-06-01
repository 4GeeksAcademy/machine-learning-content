# Web scraping problem

In this project, we are going to scrape the Tesla revenue data and store it in a dataframe, and also in a sqlite database.

To know whether a website allows web scraping or not, you can look at the website’s “robots.txt” file. You can find this file by appending “/robots.txt” to the URL that you want to scrape.

**Step 1:** Make sure you have sqlite3 and pandas installed. In case they are not installed, you can use the following command in the terminal:

```py
pip install pandas
```

and 

```py
pip install sqlite3
```

**Step 2:** Import necessary libraries


```python
# import packages

```

**Step 3:** Use the requests library to download the webpage https://www.macrotrends.net/stocks/charts/TSLA/tesla/revenue. Save the text of the response as a variable named html_data.


```python
#your code here

```

**Step 4:** Parse the html data using beautiful_soup


```python
#your code here
```

**Step 5:** Use beautiful soup or the read_html function to extract the table with Tesla Quarterly Revenue and store it into a dataframe named tesla_revenue. The dataframe should have columns Date and Revenue. Make sure the comma and dollar sign is removed from the Revenue column. Inspect the html code to know what parts of the table should be found.


```python
#find all tables


#find table with Tesla quarterly revenue




#create the dataframe        



#iterate over the table rows to get the values and remove the $ and comma 




```

**Step 6:** Remove the rows in the dataframe that are empty strings or are NaN in the Revenue column. Print the entire tesla_revenue DataFrame to see if you have any.


```python
#your code here

```

**Step 7:** Make sure tesla_revenue is still a dataframe


```python
#your code here
```

**Step 8:** Insert the data into sqlite3 by converting the dataframe into a list of tuples


```python
#your code here


```

**Step 9:** Now let's create a SQLite3 database. Use the connect() function of sqlite3 to create a database. It will create a connection object. In case the databse does not exist, it will create it.


```python
# Use the connect() function of sqlite3 to create a database. It will create a connection object.

```

**Step 10:** Let's create a table in our database to store our revenue values.


```python
#your code here


# Create table

```


```python
# Insert the values


# Save (commit) the changes

```

**Step 11:** Now retrieve the data from the database


```python
#your code here
```

Our database name is “Tesla.db”. We saved the connection to the connection object.

Next time we run this file, it just connects to the database, and if the database is not there, it will create one.

**Step 12:** Finally create a plot to visualize the data


```python
#your code here
```

Source:

https://github.com/bhavyaramgiri/Web-Scraping-and-sqlite3/blob/master/week%209-%20web%20scraping%20sqlite.ipynb

https://coderspacket.com/scraping-the-web-page-and-storing-it-in-a-sqlite3-database

https://gist.github.com/elifum/09dcaecfbc6c6e047222db3fcfe5f3b8
