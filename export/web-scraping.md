# Web Scraping

Web Scraping is one of the important methods to retrieve data from a website automatically. Not all websites allow people to scrape, however you can add the ‘robots.txt’ after the URL of the website you want to scrape, in order to know whether you will be allowed to scrape or not. 

How do we get data from a website?

There are three ways in which we can get data from the web:

1. Importing files from the internet.

2. Doing web scraping directly with code to download the HMTL content.

3. Querying data from the website API.

But what is a website API?

An API (Application Programming Interface) is a software interface that allows two applications to interact with each other without any user intervention. A web API can be accesed over the web using the HTTP protocol.

Scraping tools are specially developed software to extract data from websites. Which are the most common tools for web scraping?

- Requests: It is a Python module in which we can send HTTP requests to retrieve contents. It helps us access website HTML contents or API by sending Get or Post requests.

- Beautiful Soup: It helps us parse HTML or XML documents into a readable format. We can retrieve information faster.

- Selenium : Mostly used for website testing. It helps automate different events.



### 1. Importing flat files from the web

The flat file we will import will be the iris dataset from http://archive.ics.uci.edu/ml/machine-learning-databases/iris/ obtained from the UCI Machine Learning Repository.

After importing it, we will load it into a pandas dataframe.


```python
# Import package
from urllib.request import urlretrieve

# Import pandas
import pandas as pd

# Assign url of file: url
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Save file locally
urlretrieve(url,'iris.csv')

# Read file in a dataframe and look at the first rows
df = pd.read_csv('iris.csv', sep=';')
print(df.head())

```

       5.1,3.5,1.4,0.2,Iris-setosa
    0  4.9,3.0,1.4,0.2,Iris-setosa
    1  4.7,3.2,1.3,0.2,Iris-setosa
    2  4.6,3.1,1.5,0.2,Iris-setosa
    3  5.0,3.6,1.4,0.2,Iris-setosa
    4  5.4,3.9,1.7,0.4,Iris-setosa


### 2. Performing HTTP requests

Steps to do HTTP requests:

1. Inspect the website HTML that we want to scrape (right-click)

2. Access URL of the website using code and download all the HTML contents on the page

3. Format the downloaded content into a readable format

4. Extract out useful information and save it into a structured format

5. If the information is in multiple pages of the website, we may need to repeat steps 2–4 to have the complete information.

**Performing HTTP requests using urllib**


We'll extract the HTML itself, but first we are going to package and send the request and then catch the response.


```python
# Import packages
from urllib.request import urlopen, Request

# Specify the url
url = " https://scikit-learn.org/stable/getting_started.html"

# This packages the request
request = Request(url)

# Sends the request and catch the response
response = urlopen(request)

# Print the datatype of response
print(type(response))

# Close the response!
response.close()

```

This response is a http.client.HTTPResponse object. What can we do with it?

As it came from an HTML page, we can read it to extract the HTML using a read() method associated to it. Now let's extract the response and print the HTML.


```python
request = Request(url)

response = urlopen(request)

# Extract the response: html
html = response.read()

# Print the html
print(html)

# Close the response!
response.close()

```

**Performing HTTP requests using requests**

Now we are going to use the requests library. This time we don't have to close the connection.


```python
import requests

# Specify the url: url
url = "https://scikit-learn.org/stable/getting_started.html"

# Packages the request, send the request and catch the response: resp
resp = requests.get(url)

# Extract the response: text
text = resp.text

# Print the html
print(text)

```

**Parsing HTML Using Beautiful Soup**

We'll learn how to use the BeautifulSoup package to parse, prettify and extractinformation from HTML.


```python
# Import packages
import requests
from bs4 import BeautifulSoup

# Specify url: url
url = 'https://gvanrossum.github.io//'

# Package the request, send the request and catch the response: resp
resp = requests.get(url)

# Extracts the response as html: html_doc
html_doc = resp.text

# Create a BeautifulSoup object from the HTML: soup
soup = BeautifulSoup(html_doc)

# Prettify the BeautifulSoup object: pretty_soup
pretty_soup = soup.prettify()

# Print the response
print(pretty_soup)
```

### 3. Interacting with APIs

it is a bit more complicated than scraping the HTML document, especially if authentication is required, but the data will be more structured and stable.

Steps to querying data from the website API:

1. Inspect the XHR network section of the URL that we want to scrape

2. Find out the request-response that gives us the data that we want

3. Depending on the type of request(post or get), let's simulate the request in our code and retrieve the data from API. If authentication is required, we will need to request for token first before sending our POST request

4. Extract useful information that we need

5. For API with a limit on query size, we will need to use ‘for loop’ to repeatedly retrieve all the data

**Example: Loading and exploring a Json with GET request**



```python
# Import package
import requests

# Assign URL to variable: url
url = "https://covid-19-statistics.p.rapidapi.com/regions"

headers = {
	"X-RapidAPI-Host": "covid-19-statistics.p.rapidapi.com",
	"X-RapidAPI-Key": "SIGN-UP-FOR-KEY"
}

response = requests.request("GET", url, headers=headers)

# Decode the JSON data into a dictionary: json_data
json_data = response.json()

# Print each key-value pair in json_data
for k in json_data.keys():
    print(k + ': ', json_data[k])

```

If you want to scrape a website, we should check the existence of API first in the network section using inspect. If we can find the response to a request that gives us all the data we need, we can build a stable solution. If we cannot find the data in-network, we can try using requests or Selenium to download HTML content and use Beautiful Soup to format the data.

Other top web scraping tools in 2022:

1. Newsdata.io

2. Scrapingbee 

3. Bright Data

4. Scraping-bot 

5. Scraper API 

6. Scrapestack 

7. Apify 

8. Agenty 

9. Import.io

10. Outwit 

11. Webz.io 

References: 

https://towardsdatascience.com/web-scraping-basics-82f8b5acd45c

https://rapidapi.com/rapidapi/api

https://newsdata.io/blog/top-21-web-scraping-tools-for-you/

