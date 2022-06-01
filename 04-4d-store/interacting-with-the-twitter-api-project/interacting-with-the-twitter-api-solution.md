# Interacting with the Twitter API

Twitter can be used as a data source for various data science projects.

In this exercise we will learn how to interact with the Twitter API. We will practice storing tweets in a dataframe and saving them into a csv file.

Tweepy is a Python library to access the Twitter API. You’ll need to set up a twitter application at dev.twitter.com to obtain a set of authentication keys to use with the API. 

**Step 1:** Create an App in the developer account: https://developer.twitter.com/ . 
Make sure to get the bearer_token, consumer_key, consumer_secret, access_token, access_token_secret and have them in a safe place.
These can be generated in your developer portal, under the “Keys and tokens” tab for your developer App.

Guidance on how to create a Twitter app (step 1 and 2): https://developer.twitter.com/en/docs/tutorials/step-by-step-guide-to-making-your-first-request-to-the-twitter-api-v2

**Step 2:** Create a new project folder with a new .py file. 
Next, we need to install tweepy.


```python
! pip install tweepy
```

**Step 3:** Import Tweepy


```python
# your code here
```

**Step 4:** You need to provide the Twitter keys and tokens in order to use the API v2.

Create a simple python file called keys.py to store all passwords. Include the keys.py file in .gitignore so that your credentials are not uploaded to Github.

```py
consumer_key="insert your API key"
consumer_secret="insert your API secret"
access_token="insert your access token"
access_token_secret="insert your access token secret"
bearer_token="insert your bearer token"
```

**Step 5:** Make a connection with API v2. Import the keys and use them in the function tweepy.Client:


```python
#import all data from the keys file

from keys import *
```


```python
import requests

client = tweepy.Client( bearer_token=bearer_token, 
                        consumer_key=consumer_key, 
                        consumer_secret=consumer_secret, 
                        access_token=access_token, 
                        access_token_secret=access_token_secret, 
                        return_type = requests.Response,
                        wait_on_rate_limit=True)
```

**Step 6:** Make a query. Search Tweets that have the hashtag #100daysofcode from the last 7 days (search_recent_tweets).
Limit the result to a maximum of 100 Tweets.
Also include some additional information with tweet_fields (author id and when the Tweet was created).


```python
# Define query
query = 'has:#100daysofcode has:react has:python'

# get max. 100 tweets
tweets = client.search_recent_tweets(query=query, 
                                    tweet_fields=['author_id','created_at','lang','geo','possibly_sensitive'],
                                     max_results=100)
```

**Step 7:** Convert to pandas Dataframe


```python
import pandas as pd

# Save data as dictionary
tweets_dict = tweets.json() 

# Extract "data" value from dictionary
tweets_data = tweets_dict['data'] 

# Transform to pandas Dataframe
df = pd.json_normalize(tweets_data) 
```

**Step 8:** Print dataframe


```python
print(df)
```

**Step 9:** Save data in a csv file


```python
# save df
df.to_csv("tweets-code.csv")
```

**Step 10:** Iterate through dataframe rows counting the number of tweets in which react and python are mentioned.


```python
# Initialize list to store tweet counts
[react, python] = [0, 0]

# Iterate through df, counting the number of tweets in which each(react and python) is mentioned.
for index, row in df.iterrows():
    react += word_in_text('react', row['text'])
    python += word_in_text('python', row['text'])
```

**Step 11:** Visualize the data


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
