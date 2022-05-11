
# The Twitter API

### API authentication

1. We need to create a Twitter app in https://apps.twitter.com/

2. Import the package tweepy (it handles the Twitter API authentication)

3. Store OAuth authentication credentials in relevant variables

4. Pass OAuth details to tweepy's OAuth handler and assign it in OAuth handler 'auth'

5. Apply to it the method set_access_token(), along with arguments access_token and access_token_secret.


```python
#Defining the tweet stream listener

class MyStreamListener (tweepy.StreamListener):
    def __init__(self, api = None):
        super(MyStreamListener, self).__init__()
        self.num_tweets = 0
        self.file_name = "tweets.txt"
        #self.file = open("tweets.txt", "w")

    def on_status(self, status):
        tweet = status._json
        with open(self.file_name, 'a') as file:
            file.write(json.dumps(tweet) + '\n')
        self.num_tweets += 1
        if self.num_tweets < 100:
            return True
        else:
            return False

    def on_error(self, status):
        print(status)
```

6. Let's initialize Stream listener

7. Let's create our Stream object with authentication by passing tweepy.Stream() the authentication handler auth and the Stream listener variable

8. Filter Twitter Streams to capture data by keywords, by using stream.filter(track=[keywords])

9. Let's store pur Twitter data in a text file

10. Import json

11. Assing our text file to a path

12. Initialize empty list to store tweets

13. Open connection to file with  open(path, "r")

14. Read in tweets and store in our empty list, by using the for loop to load each tweet into a variable (using json.loads()), then append the variable value to the empty list.

15. Close the connection and print the dictionary keys of the first tweet.

16. Convert our new list of dictionaries into a dataframe where each row is a tweet with two columns, 1 for the text, and the other for the language.

17. Let's print the head of the new dataframe.

18. Define a function that counts how many tweets contain our keywords.

19.  Initialize an empty list to store tweet counts for each keyword.

20. Use a for loop to iterate through each row of the dataframe to make the count of tweets for each keyword.

21. Create a chart that best represents the analysis, using visualization tools.





