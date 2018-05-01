"""
RUNNING PROGRAM;

1-Start Apache Kafka
./kafka/kafka_2.11-0.11.0.0/bin/kafka-server-start.sh ./kafka/kafka_2.11-0.11.0.0/config/server.properties

2-Run kafka_push_listener.py (Start Producer)
ipython >> run kafka_push_listener.py

3-Run kafka_twitter_spark_streaming.py (Start Consumer)
PYSPARK_PYTHON=python3 bin/spark-submit --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:2.2.0 ~/Documents/kafka_twitter_spark_streaming.py

"""

from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import json
from operator import add
from textblob import TextBlob
import re
from time import sleep
import matplotlib.pyplot as plt

def parseText(line):
    if 'text' in line:
        return line['text']
    else:
        return ' '

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing 
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

def cumu(newvalue, count):
    if count is None:
        count = 0
    return sum(newvalue, count)

def Stream():
    sc = SparkContext(appName = 'NewsTwitter')
    ssc = StreamingContext(sc, 10)
    ssc.checkpoint('checkpoint')
    kafkaStream = KafkaUtils.createStream(ssc, 'localhost:2181', 'spark-streaming', {'twitter':1})

    parsed = kafkaStream.map(lambda v: json.loads(v[1])).map(parseText)
    parsed.cache()
    tweets_saver = parsed.map(lambda tweet: tweet + '\n').reduce(lambda x, y: x + y)
    tweets_saver.saveAsTextFiles('file:///home/lmh/Downloads/temp/lmh/text/tweets/t')
    sentiment = parsed.map(lambda tweet: [analize_sentiment(tweet), 1])
    sentiment_count = sentiment.reduceByKey(add)
    sentiment_count.cache()
    sentiment_count.saveAsTextFiles('file:///home/lmh/Downloads/temp/lmh/text/sentiment/s')
    sentiment_count.pprint()
    sentiment_fig_data = sentiment_count.updateStateByKey(cumu)

    counts = []
    sentiment_fig_data.foreachRDD(lambda s, rdd: counts.append(rdd.collect()))

    ssc.start()
    ssc.awaitTerminationOrTimeout(60)
    ssc.stop(stopGraceFully = True)

    return counts

def make_plot(count):
    s_final = count[-1]
    s_list = [0] * 3
    for k, v in s_final:
        s_list[k] = v

    s_list.reverse()
    plt.figure()
    plt.plot([-1, 0, 1], s_list)
    plt.savefig(img_path + 'stat.png')
    plt.close()

if __name__ == "__main__":

    img_path = '/home/lmh/Downloads/temp/lmh/img/'

    while True:
        count = Stream()
        make_plot(count)
        sleep(10)


 
