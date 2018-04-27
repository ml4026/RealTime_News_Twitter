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
import matplotlib.pyplot as plt

def make_plot(counts):
    """
    This function plots the counts of positive and negative words for each timestep.
    """
    positiveCounts = []
    negativeCounts = []
    time = []

    for val in counts:
        positiveTuple = val[0]
        positiveCounts.append(positiveTuple[1])
        negativeTuple = val[1]
        negativeCounts.append(negativeTuple[1])

    for i in range(len(counts)):
        time.append(i)

    posLine = plt.plot(time, positiveCounts,'bo-', label='Positive')
    negLine = plt.plot(time, negativeCounts,'go-', label='Negative')
    plt.axis([0, len(counts), 0, max(max(positiveCounts), max(negativeCounts))+50])
    plt.xlabel('Time step')
    plt.ylabel('Word count')
    plt.legend(loc = 'upper left')
    plt.show()

def makePlot(sentiment_list):
    sentiment_count = [0] * 3
    sentiment_count[sentiment_list[0]] = sentiment_list[1]
    sentiment_count[sentiment_list[2]] = sentiment_list[3]
    sentiment_count[sentiment_list[4]] = sentiment_list[5]
    plt.figure()
    plt.plot(sentiment_count)
    plt.savefig('img_lib/count.png')
    plt.close()

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

def sendRecord(record):
    connection = createNewConnection()
    connection.send(record)
    connection.close()

if __name__ == "__main__":
    '''
    path = '/home/lmh/Downloads/temp/'

    '''
    sc = SparkContext(appName = 'NewsTwitter')
    ssc = StreamingContext(sc, 10)
    kafkaStream = KafkaUtils.createStream(ssc, 'localhost:2181', 'spark-streaming', {'twitter':1})
    parsed = kafkaStream.map(lambda v: json.loads(v[1])).map(lambda line: line['text'])
    sentiment = parsed.map(lambda tweet: (analize_sentiment(tweet), 1))
    sentiment_list = sentiment.reduceByKey(add).map(lambda x, y: [x, y]).reduce(x, y: x + y)
    sentiment_list.map(makePlot)

    ssc.start()
    ssc.awaitTerminationOrTimeout(30)
    ssc.stop(stopGraceFully = True)


