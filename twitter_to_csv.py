#In this file we are scraping twitter for tweets about $AAPL, then we put the tweets into a CSV where we will manually determine if they reflect positive
#or negative sentiment. This will be the basis for creating a custom training set for sentiment analysis.

import tweepy
import csv
import re
import numpy as np
from textblob import TextBlob

#setting up twitter API
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAJrVbQEAAAAAskzdAWYVNdGmvxlhSPp2MCP0mM8%3DfheyaxCiWxGKybkok6QDG81CHnViUQqLm7tZb19Df07XmrpD8n'
client = tweepy.Client(bearer_token)

#get tweets
response = client.search_recent_tweets("AAPL", max_results=100)
#response = client.search_all_tweets("AAPL")
tweets = response.data

#clean tweets
stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]
def clean_tweet(tweet):
    if type(tweet) == np.float:
        return ""
    temp = tweet.lower()
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp)
    temp = re.sub("#[A-Za-z0-9_]+","", temp)
    temp = re.sub(r'http\S+', '', temp)
    temp = re.sub('[()!?]', ' ', temp)
    temp = re.sub('\[.*?\]',' ', temp)
    temp = re.sub("[^a-z0-9]"," ", temp)
    temp = temp.split()
    temp = [w for w in temp if not w in stopwords]
    temp = " ".join(word for word in temp)
    return temp

#iterate through tweets adding to a dictionary
mydict = {}
for tweet in tweets:
    cleaned_tweet = clean_tweet(tweet.text)
    mydict[cleaned_tweet] = 0
    print(cleaned_tweet)
#print(mydict)

#writing csv
with open('combine4.csv', 'w') as f:
    for key in mydict.keys():
        f.write("%s,%s\n"%(key,mydict[key]))
