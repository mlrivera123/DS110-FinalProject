#In this file we are scraping twitter for tweets about $AAPL, then we put the tweets into a CSV where we will manually determine if they reflect positive
#or negative sentiment. This will be the basis for creating a custom training set for sentiment analysis.

import tweepy
import csv
from textblob import TextBlob

#setting up twitter API
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAJrVbQEAAAAAskzdAWYVNdGmvxlhSPp2MCP0mM8%3DfheyaxCiWxGKybkok6QDG81CHnViUQqLm7tZb19Df07XmrpD8n'
client = tweepy.Client(bearer_token)

#get tweets
response = client.search_recent_tweets("AAPL")
tweets = response.data

#iterate through tweets adding to a dictionary
mydict = {}
for tweet in tweets:
    blob = TextBlob(str(tweet.text))
    sentences = blob.sentences
    for sentence in sentences: #this part iterates through the sentences within the tweet
        if str(sentence) not in mydict:
            mydict[str(sentence)] = 0

print(mydict)

#setting up fields for csv file we want to write
fields = ['tweet', 'rating']
filename = "aapltweets.csv"

# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv dict writer object 
    writer = csv.DictWriter(csvfile, fieldnames = fields) 
        
    # writing headers (field names) 
    writer.writeheader() 
        
    # writing data rows 
    writer.writerows(mydict)
