import pykafka
import json
import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
from Credential import apikey, consumer_key, consumer_secret, access_token, secret_token
import time
from newsapi import NewsApiClient
from nltk.corpus import stopwords
from nltk import sent_tokenize, ne_chunk, pos_tag, word_tokenize

#News API AUTH
client = NewsApiClient(api_key = apikey)

#TWITTER API AUTH
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, secret_token)

api = tweepy.API(auth)

#Twitter Stream Listener
class KafkaPushListener(StreamListener):          
	def __init__(self):
		#localhost:9092 = Default Zookeeper Producer Host and Port Adresses
		self.client = pykafka.KafkaClient("localhost:9092")
		
		#Get Producer that has topic name is Twitter
		self.producer = self.client.topics[bytes("twitter", "ascii")].get_producer()
  
	def on_data(self, data):
		#Producer produces data for consumer
		#Data comes from Twitter
		self.producer.produce(bytes(data, "ascii"))
		return True
                                                                                                                                           
	def on_error(self, status):
		print(status)
		return True

#Twitter Stream Config
twitter_stream = Stream(auth, KafkaPushListener())

#Produce Data that has Game of Thrones hashtag (Tweets)

while True:
	headlines = client.get_top_headlines(language = 'en', country = 'us')
	titles = []
	for article in headlines['articles']:
    		temp_title = article['title'].encode('ascii', 'ignore').decode('ascii')
    		print(temp_title)
    		titles.append(temp_title)
	#titles = titles[:3]

	keywords = []
	for text in titles:    
    		for sent in sent_tokenize(text):
        		for chunk in ne_chunk(pos_tag(word_tokenize(sent))):
            			if hasattr(chunk, 'label'):
                			if chunk.label() == 'PERSON' or chunk.label() == 'ORGANIZATION' or chunk.label() == 'LOCATION':
                    				keywords.append(' '.join(c[0] for c in chunk.leaves()))
	print(keywords)

	if twitter_stream.running is True:
		twitter_stream.disconnect()
		time.sleep(5)

	if not keywords:
		print('no keywords to listen to')
	else:
		twitter_stream.filter(track = keywords, languages = ['en'], async = True)
	time.sleep(115)
