consumer_key = 'JLsYLzgbgimd9AUO49d9afnmm'
consumer_secret = 'a8JGJvtJJBpm009JiTcUs1KfD6lLop9YzFECh7yGBA91T0OimM'
access_token = '1514283971130974212-adKXE1JrGjd7EnVRSZsihOb8TfP6mC'
access_token_secret = '2Xz3HE4xMWn39XtX8doTVkq5KEK5AV9Qh4JAmGeLOTVX7'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAJrVbQEAAAAAskzdAWYVNdGmvxlhSPp2MCP0mM8%3DfheyaxCiWxGKybkok6QDG81CHnViUQqLm7tZb19Df07XmrpD8n'

#pip install tweepy
import tweepy

#client = tweepy.Client(consumer_key=consumer_key, consumer_secret=consumer_secret, access_token=access_token, access_token_secret=access_token_secret)
client = tweepy.Client(bearer_token)
response = client.search_recent_tweets("AAPL")
# The method returns a Response object, a named tuple with data, includes,
# errors, and meta fields
print(response.meta)

tweets = response.data

# Each Tweet object has default ID and text fields
for tweet in tweets:
    #print(tweet.id)
    print(tweet.text)
