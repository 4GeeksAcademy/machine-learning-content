# Interacting with the Twitter API

Twitter can be used as a data source for various data science projects, including Geo-spatial analysis (where are users tweeting about certain subjects?) and sentiment analysis (how do users feel about certain subjects?).

In this exercise we will learn how to stream real-time Twitter data. We will practice storing it in a dataframe to get some visualizations, as well as storing the data in a SQLite database, and building a web-app using Streamlit. Let's enumerate the tasks needed by dividing it into 3 areas:

1. Database set-up: This can be done directly in the RDBMS of your choice, however we choose to use SQLite in this example.

2. Tweepy: Credentials are required to interact with the Tweepy API. Once these have been obtained from dev.twitter.com we can set up a stream with keyword filters.

3. Streamlit: Once we have our data stream working, we’ll need to set up our web app using Streamlit. This is surprisingly simple and can be done within a single python file!

### Authentication

Tweepy is a Python library to access the Twitter API. You’ll need to set up a twitter application at dev.twitter.com to attain a set of authentication keys to use with the API. Streaming with Tweepy comprises of three objects; Stream, StreamListener, OAuthHandler. The latter simply handles API authentication and requires the unique keys from the creation of your Twitter app. As Tweepy has been recently updated, and in order to avoid version conflicts, the streamlistener code will be provided to work with version ... of Tweepy. 


```python
# Import package
import tweepy,json

# Store OAuth authentication credentials in relevant variables

access_token = ACCESS_TOKEN
access_token_secret = ACCESS_TOKEN_SECRET
consumer_key = CONSUMER_KEY
consumer_secret = CONSUMER_SECRET

# Pass OAuth details to tweepy's OAuth handler

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
```

### Streaming Tweets


```python
from stream import TweetListener

# Initialize Stream listener
l = TweetListener()

# Create your Stream object with authentication
stream = tweepy.Stream(auth, l)

# Filter Twitter Streams to capture data by the keywords:
stream.filter(['russia', 'ukraine'])
```

### Exploring the data


```python
from stream import TweetListener

# Initialize Stream listener
l = TweetListener()

# Create your Stream object with authentication
stream = tweepy.Stream(auth, l)

# Filter Twitter Streams to capture data by the keywords:
stream.filter(['russia', 'ukraine'])
```

### Building a dataframe with our Twitter data


```python
# Import package
import pandas as pd

# Build DataFrame of tweet texts and languages
df = pd.DataFrame(tweets_data, columns=['text', 'lang'])

# Print head of DataFrame
print(df.head())
```

### Analizing some text


```python
# Initialize list to store tweet counts
[russia, ukraine] = [0, 0, 0, 0]

# Iterate through df, counting the number of tweets in which
# each candidate is mentioned
for index, row in df.iterrows():
    russia += word_in_text('russia', row['text'])
    ukraine += word_in_text('ukraine', row['text'])
```

### Visualizing the data


```python
# Import packages
import matplotlib.pyplot as plt
import seaborn as sns

# Set seaborn style
sns.set(color_codes=True)

# Create a list of labels:cd
cd = ['russia', 'ukraine']

# Plot the bar chart
ax = sns.barplot(cd, [russia, ukraine])
ax.set(ylabel="count")
plt.show()
```
