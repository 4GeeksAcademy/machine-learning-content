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

Import Tweepy


```python
# your code here

```

**Step 4:** 

You need to provide the Twitter keys and tokens in order to use the API v2.

Create a simple python file called keys.py to store all passwords. Include the keys.py file in a .gitignore file so that your credentials are not uploaded to Github.

```py
consumer_key="insert your API key"
consumer_secret="insert your API secret"
access_token="insert your access token"
access_token_secret="insert your access token secret"
bearer_token="insert your bearer token"
```

**Step 5:** 

Make a connection with API v2. Import the keys and use them in the function tweepy.Client(). Use the following documentation for guidance on the parameters: https://docs.tweepy.org/en/stable/client.html


```python
#import all data from the keys file

```


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
