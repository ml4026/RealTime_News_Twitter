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
import operator
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

    
def load_wordlist(filename):
    """ 
    This function returns a list or set of words from the given filename.
    """ 
    words = {}
    f = open(filename, 'rU')
    text = f.read()
    text = text.split('\n')
    for line in text:
        words[line] = 1
    f.close()
    return words


def wordSentiment(word,pwords,nwords):
    if word in pwords:
        return ('positive', 1)
    elif word in nwords:
        return ('negative', 1)


def updateFunction(newValues, runningCount):
    if runningCount is None:
       runningCount = 0
    return sum(newValues, runningCount) 


def sendRecord(record):
    connection = createNewConnection()
    connection.send(record)
    connection.close()
if __name__ == "__main__":
    path = '/home/lmh/Downloads/temp/'
	#Create Spark Context to Connect Spark Cluster
    pwords = load_wordlist(path + "Dataset/positive.txt")
    nwords = load_wordlist(path + "Dataset/negative.txt")
    sc = SparkContext(appName="PythonStreamingKafkaTweetCount")

	#Set the Batch Interval is 10 sec of Streaming Context
    ssc = StreamingContext(sc, 10)

	#Create Kafka Stream to Consume Data Comes From Twitter Topic
	#localhost:2181 = Default Zookeeper Consumer Address
    kafkaStream = KafkaUtils.createStream(ssc, 'localhost:2181', 'spark-streaming', {'twitter':1})
    
    #Parse Twitter Data as json
    parsed = kafkaStream.map(lambda v: json.loads(v[1]))
    words = parsed.flatMap(lambda line:line['text'].split(" "))
    #words.pprint()
    positive = words.map(lambda word: ('Positive', 1) if word in pwords else ('Positive', 0))
    negative = words.map(lambda word: ('Negative', 1) if word in nwords else ('Negative', 0))
    allSentiments = positive.union(negative)
    sentimentCounts = allSentiments.reduceByKey(lambda x,y: x+y)
    sentimentCounts.pprint()
    # runningSentimentCounts = sentimentCounts.updateStateByKey(updateFunction)
    # runningSentimentCounts.pprint()
    #text = parsed.map(lambda tweet: (tweet['text'], 1)).reduceByKey(lambda x,y: x + y)
    #text.pprint()
	#Count the number of tweets per User
    #user_counts = parsed.map(lambda tweet: (tweet['text']["screen_name"], 1)).reduceByKey(lambda x,y: x + y)

	#Print the User tweet counts
    #user_counts.pprint()

	#Start Execution of Streams
    ssc.start()
    ssc.awaitTerminationOrTimeout(30)
    ssc.stop(stopGraceFully = True)


