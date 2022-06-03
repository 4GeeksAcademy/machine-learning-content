# Interacting with the Twitter API

Twitter can be used as a data source for various data science projects.

In this exercise we will learn how to interact with the Twitter API. We will practice storing tweets in a dataframe and saving them into a csv file.

Tweepy is a Python library to access the Twitter API. You’ll need to set up a twitter application at dev.twitter.com to obtain a set of authentication keys to use with the API. 

**Step 1:** 

Create an App in the developer account: https://developer.twitter.com/ . 

Make sure to get the bearer_token, consumer_key, consumer_secret, access_token, access_token_secret and have them in a safe place.
These can be generated in your developer portal, under the “Keys and tokens” tab for your developer App.

Guidance on how to create a Twitter app (step 1 and 2): https://developer.twitter.com/en/docs/tutorials/step-by-step-guide-to-making-your-first-request-to-the-twitter-api-v2

**Step 2:** 

Create a new project folder with a new .py file. 
Next, we need to install tweepy.


```python
#install tweepy
```

**Step 3:** 

You need to provide the Twitter keys and tokens in order to use the API v2.

To do it in a safe way, you should store the secrets in a seperate .env file.
A dotenv file contains only text, where it has one environment variable assignment per line.
Create a .env file in your project and add your secret keys or passwords: 

```py
CONSUMER_KEY="insert your API key"
CONSUMER_SECRET="insert your API secret"
ACCEESS_TOKEN="insert your access token"
ACCESS_TOKEN_SECRET="insert your access token secret"
BEARER_TOKEN="insert your bearer token"
```

>Important: Make sure to add it in your .gitignore file, which is not saved to source control, so that you aren't putting potentially sensitive information at risk. 

To set password or secret keys in environment variable on Linux(and Mac) or Windows, see the following link: https://dev.to/biplov/handling-passwords-and-secret-keys-using-environment-variables-2ei0


To access these variables in Windows in our python script, we need to import the os module.
We can do that by using os.environ.get() method and passing the key we want to access.

Now, you need to install python-dotenvpackage. python-dotenv is a Python package that lets your Python app read a .env file. This package will search for a .env and if it finds one, will expose the variables in it to the app.

Example:

```py
from dotenv import load_dotenv   #for python-dotenv method
load_dotenv()                    

import os 

user_name = os.environ.get('USER')
password = os.environ.get('password')
```

**Step 4:** 

Import Tweepy


```python
#import tweepy
```

**Step 5:** 

Make a connection with API v2. Use the variables in the function tweepy.Client(). Use the following documentation for guidance on the parameters: https://docs.tweepy.org/en/stable/client.html


```python
#import requests


# Use tweepy.Client()




```

**Step 6:** 

Make a query. Search tweets that have the hashtag #100daysofcode and the word python or react, from the last 7 days (search_recent_tweets). 

Do not include retweets. Limit the result to a maximum of 100 Tweets.

Also include some additional information with tweet_fields (author id, when the tweet was created, the language of the tweet text).

You can use this link for guidance on how to create the query: https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query


```python
# Define query


# get max. 100 tweets


```

**Step 7:** 

Convert to pandas Dataframe


```python
#import pandas


# Save data as dictionary


# Extract "data" value from dictionary
 

# Transform to pandas Dataframe


```

**Step 8:** 

Take a look at your dataframe


```python
#your code
```

**Step 9:**

Save data in a csv file named 'coding-tweets'


```python
# save df

```

**Step 10:** 

Now that you have your DataFrame of tweets set up, you're going to do a bit of text analysis to count how many tweets contain the words 'react', and 'python'. Define the following function word_in_text(), which will tell you whether the first argument (a word) occurs within the 2nd argument (a tweet). 

>Make sure to convert any word or tweet text into lowercase.
>You can use the re python library (regular expression operations). See the documentation for guidance: https://docs.python.org/3/library/re.html#



```python
#import re


#define your function here


```

**Step 11:**

Iterate through dataframe rows counting the number of tweets in which react and python are mentioned, using your word_in_text() function.


```python
# Initialize list to store tweet counts


# Iterate through df, counting the number of tweets in which each(react and python) is mentioned.

```

**Step 12:** 

Visualize the data


```python
# Import packages


# Set seaborn style


# Create a list of labels:cd


# Plot the bar chart


```

Source: 

https://www.kirenz.com/post/2021-12-10-twitter-api-v2-tweepy-and-pandas-in-python/twitter-api-v2-tweepy-and-pandas-in-python/

https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query
