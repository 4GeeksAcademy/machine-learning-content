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
! pip install tweepy
```

    Collecting tweepy
      Downloading tweepy-4.10.0-py3-none-any.whl (94 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m94.4/94.4 kB[0m [31m6.9 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: requests<3,>=2.27.0 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from tweepy) (2.27.1)
    Collecting requests-oauthlib<2,>=1.2.0
      Downloading requests_oauthlib-1.3.1-py2.py3-none-any.whl (23 kB)
    Collecting oauthlib<4,>=3.2.0
      Downloading oauthlib-3.2.0-py3-none-any.whl (151 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m151.5/151.5 kB[0m [31m21.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: charset-normalizer~=2.0.0 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from requests<3,>=2.27.0->tweepy) (2.0.12)
    Requirement already satisfied: idna<4,>=2.5 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from requests<3,>=2.27.0->tweepy) (3.3)
    Requirement already satisfied: certifi>=2017.4.17 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from requests<3,>=2.27.0->tweepy) (2022.5.18.1)
    Requirement already satisfied: urllib3<1.27,>=1.21.1 in /home/gitpod/.pyenv/versions/3.8.13/lib/python3.8/site-packages (from requests<3,>=2.27.0->tweepy) (1.26.9)
    Installing collected packages: oauthlib, requests-oauthlib, tweepy
    Successfully installed oauthlib-3.2.0 requests-oauthlib-1.3.1 tweepy-4.10.0
    [33mWARNING: There was an error checking the latest version of pip.[0m[33m
    [0m

**Step 3:** 

You need to provide the Twitter keys and tokens in order to use the API v2.

To do it in a safe way, you should store the secrets in a seperate .env file.
A dotenv file contains only text, where it has one environment variable assignment per line.
Create a .env file in your project and add your secret keys or passwords: 

```py
consumer_key="insert your API key"
consumer_secret="insert your API secret"
access_token="insert your access token"
access_token_secret="insert your access token secret"
bearer_token="insert your bearer token"
```


>Important: Make sure to add it in your .gitignore file, which is not saved to source control, so that you aren't putting potentially sensitive information at risk. 

Now, you need to install python-dotenvpackage. python-dotenv is a Python package that lets your Python app read a .env file. This package will search for a .env and if it finds one, will expose the variables in it to the app.

Example:

```py
from dotenv import load_dotenv   #for python-dotenv method
load_dotenv()                    

import os 

consumer_key = os.environ.get('CONSUMER_KEY')
consumer_secret = os.environ.get('CONSUMER_SECRET')
access_token = os.environ.get('ACCESS_TOKEN')
access_token_secret = os.environ.get('ACCESS_TOKEN_SECRET')
bearer_token = os.environ.get('BEARER_TOKEN')

```

To set password or secret keys in environment variable on Linux(and Mac) or Windows, see the following link: https://dev.to/biplov/handling-passwords-and-secret-keys-using-environment-variables-2ei0

**Step 4:** 

Import Tweepy


```python
# your code here
import tweepy
```

**Step 5:** 

Make a connection with API v2. Use your in the function tweepy.Client(). Use the following documentation for guidance on the parameters: https://docs.tweepy.org/en/stable/client.html


```python
#import requests
import requests

# Use tweepy.Client()
client = tweepy.Client( bearer_token=bearer_token, 
                        consumer_key=consumer_key, 
                        consumer_secret=consumer_secret, 
                        access_token=access_token, 
                        access_token_secret=access_token_secret, 
                        return_type = requests.Response,
                        wait_on_rate_limit=True)
```

**Step 6:** 

Make a query. Search tweets that have the hashtag #100daysofcode and the word python or react, from the last 7 days (search_recent_tweets). 

Do not include retweets. Limit the result to a maximum of 100 Tweets.

Also include some additional information with tweet_fields (author id, when the tweet was created, the language of the tweet text).

You can use this link for guidance on how to create the query: https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query


```python
# Define query
query = '#100daysofcode (react OR python) -is:retweet'

# get max. 100 tweets
tweets = client.search_recent_tweets(query=query, 
                                    tweet_fields=['author_id','created_at','lang'],
                                     max_results=100)
```

**Step 7:** 

Convert to pandas Dataframe


```python
import pandas as pd

# Save data as dictionary
tweets_dict = tweets.json() 

# Extract "data" value from dictionary
tweets_data = tweets_dict['data'] 

# Transform to pandas Dataframe
df = pd.json_normalize(tweets_data) 
```

**Step 8:** 

Take a look at your dataframe


```python
df
```

**Step 9:**

Save data in a csv file named 'coding-tweets'


```python
# save df
df.to_csv("coding-tweets.csv")
```

**Step 10:** 

Now that you have your DataFrame of tweets set up, you're going to do a bit of text analysis to count how many tweets contain the words 'react', and 'python'. Define the following function word_in_text(), which will tell you whether the first argument (a word) occurs within the 2nd argument (a tweet). 

>Make sure to convert any word or tweet text into lowercase.
>You can use the re python library (regular expression operations). See the documentation for guidance: https://docs.python.org/3/library/re.html#



```python
#import re
import re

#define your function here
def word_in_text(word, text):
    word = word.lower()
    text = text.lower()
    match = re.search(word, text)

    if match:
        return True
    return False

```

**Step 11:**

Iterate through dataframe rows counting the number of tweets in which react and python are mentioned, using your word_in_text() function.


```python
# Initialize list to store tweet counts
[react, python] = [0, 0]

# Iterate through df, counting the number of tweets in which each(react and python) is mentioned.
for index, row in df.iterrows():
    react += word_in_text('react', row['text'])
    python += word_in_text('python', row['text'])
```

**Step 12:** 

Visualize the data


```python
# Import packages
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(color_codes=True)

# Create a list of labels:cd
cd = ['react', 'python']

# Plot the bar chart
ax = sns.barplot(cd, [react, python])
ax.set(ylabel="count")
plt.show()
```

Source: 

https://www.kirenz.com/post/2021-12-10-twitter-api-v2-tweepy-and-pandas-in-python/twitter-api-v2-tweepy-and-pandas-in-python/

https://developer.twitter.com/en/docs/twitter-api/tweets/search/integrate/build-a-query

https://www.indeed.com/q-Remote-Entry-Level-Human-Resource-Position-jobs.html?vjk=f66e30024cc1150f

https://www.codegrepper.com/code-examples/python/how+to+install+dotenv+python

https://dev.to/jakewitcher/using-env-files-for-environment-variables-in-python-applications-55a1

https://dev.to/biplov/handling-passwords-and-secret-keys-using-environment-variables-2ei0
