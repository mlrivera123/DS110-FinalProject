consumer_key = 'JLsYLzgbgimd9AUO49d9afnmm'
consumer_secret = 'a8JGJvtJJBpm009JiTcUs1KfD6lLop9YzFECh7yGBA91T0OimM'
access_token = '1514283971130974212-adKXE1JrGjd7EnVRSZsihOb8TfP6mC'
access_token_secret = '2Xz3HE4xMWn39XtX8doTVkq5KEK5AV9Qh4JAmGeLOTVX7'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAJrVbQEAAAAAskzdAWYVNdGmvxlhSPp2MCP0mM8%3DfheyaxCiWxGKybkok6QDG81CHnViUQqLm7tZb19Df07XmrpD8n'

import tweepy
import requests
import csv
from bs4 import BeautifulSoup
from textblob import TextBlob
import re
import numpy as np
import pandas as pd
import torch
import nltk
import transformers as ppb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

#client = tweepy.Client(consumer_key=consumer_key, consumer_secret=consumer_secret, access_token=access_token, access_token_secret=access_token_secret)
client = tweepy.Client(bearer_token)
response = client.search_recent_tweets("AAPL")

#Cleaning Tweets
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

#Test printing sentiment
def getSentiment():
    tweets = response.data
    for tweet in tweets:
        cleaned_tweet = clean_tweet(tweet.text)
        blob = TextBlob(str(cleaned_tweet))
        sentences = blob.sentences
        for sentence in sentences:
            print(sentence.sentiment, tweet.text)

# Based on tutorial at
# http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/
# and including some code from there

# Location of SST2 sentiment dataset
SST2_LOC = 'https://github.com/clairett/pytorch-sentiment-classification/raw/master/data/SST2/train.tsv'

WEIGHTS = 'distilbert-base-uncased'
# Performance on whole 6920 sentence set is very similar, but takes rather longer
SET_SIZE = 10

# Download the dataset from its Github location, return as a Pandas dataframe
def get_dataframe():
    #df = pd.read_csv('combine4.csv', header=None)
    df = pd.read_csv(SST2_LOC, delimiter='\t', header=None)
    return df[:SET_SIZE]

# Extract just the labels from the dataframe
def get_labels(df):
    return df[1]

# Get a trained tokenizer for use with BERT
def get_tokenizer():
    return ppb.DistilBertTokenizer.from_pretrained(WEIGHTS)

# Convert the sentences into lists of tokens
def get_tokens(dataframe, tokenizer):
    return dataframe[0].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))

# We want the sentences to all be the same length; pad with 0's to make it so
def pad_tokens(tokenized):
    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)
    padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
    return padded

# Grab a trained DistiliBERT model
def get_model():
    return ppb.DistilBertModel.from_pretrained(WEIGHTS)

# This step takes a little while, since it actually runs the model on all sentences.
# Get model with get_model(), 0-padded token lists with pad_tokens() on get_tokens().
# Only returns the [CLS] vectors representing the whole sentence, corresponding to first token.
def get_bert_sentence_vectors(model, padded_tokens):
    # Mask the 0's padding from attention - it's meaningless
    mask = torch.tensor(np.where(padded_tokens != 0, 1, 0))
    with torch.no_grad():
        word_vecs = model(torch.tensor(padded_tokens).to(torch.int64), attention_mask=mask)
    # First vector is for [CLS] token, represents the whole sentence
    return word_vecs[0][:,0,:].numpy()


# To separate into train and test:
# train_features, test_features, train_labels, test_labels = train_test_split(vecs, labels)
def train_knn(train_features, train_labels):
    knc = KNeighborsClassifier()
    knc.fit(train_features, train_labels)
    return knc

# General purpose scikit-learn classifier evaluator.  The classifier is trained with .fit()
def evaluate(classifier, test_features, test_labels):
    return classifier.score(test_features, test_labels)


def get_tokens_from_sentence(sentence):
  df = pd.DataFrame([[sentence]])
  return get_tokens(df,get_tokenizer())

def get_bert_vecs_from_sentence(sentence):
  tokens = get_tokens_from_sentence(sentence)
  model = get_model()
  vecs =  get_bert_sentence_vectors(model, pad_tokens(tokens))
  return vecs

def predict_from_sentence(clf, sentence):
  vecs = get_bert_vecs_from_sentence(sentence)
  return clf.predict(vecs)

df = get_dataframe()
df.head()

labels = get_labels(df)
tokenizer = get_tokenizer()
tokens = get_tokens(df, tokenizer)
padded = pad_tokens(tokens)
model = get_model()
vecs = get_bert_sentence_vectors(model, padded)

train_features, test_features, train_labels, test_labels = train_test_split(vecs, labels)

random_forest = RandomForestClassifier()
random_forest.fit(train_features, train_labels)

#import logistic regression
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(max_iter=3000) # avoid warning about not compiling enough
lr_clf.fit(train_features, train_labels)
lr_clf.score(test_features, test_labels)

def MachinePredict(type):
    tweets = response.data
    for tweet in tweets:
        cleaned_tweet = clean_tweet(tweet.text)
        blob = TextBlob(str(cleaned_tweet))
        sentences = blob.sentences
        for sentence in sentences: #this part iterates through the sentences within the tweet
            if(type == "random_forest"):
                print(predict_from_sentence(random_forest, str(sentence)), str(sentence))
            elif(type == "logistic_regression"):
                print(predict_from_sentence(lr_clf, str(sentence)), str(sentence))
            else:
                print("sorry we don't have that yet!")

#getSentiment()
#MachinePredict("random_forest")
MachinePredict("logistic_regression")
