# DS110-FinalProject
This program uses a twitter API to search recent tweets for stock symbols. 
After collecting tweets, twitter_to_csv cleans the text and puts the tweet text into a dictionary which is then converted to a csv.
At this point the user has to manually rate the sentiment in the CSV as either positive 1 or negative 0.
Twitterbot uses the CSV as an input to train a BERT natural language processing model.
BERT converts each tweet into a vector of information.
Twitterbot then runs the vectors through the trained model and can use K nearest neighbors or random forrest to classify the sentiment.
To run this program, go to Twitterbot.py and uncomment which method you want use to classify tweets.