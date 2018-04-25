# RealTime_News_Twitter

## Dependencies
pip install tweepy<br />
pip install pykafka<br />
pip install newsapi-python<br />
pip install matplotlib<br />
pip install -U textblob<br />
export NLTK_DATA=<directory> (optional, if you wish to install nltk data in a different directory other than root)<br />
python -m textblob.download_corpora

## Run
Start Zookeeper server<br />
<Kafka path>/bin/zookeeper-server-start.sh config/zookeeper.properties<br />
Start Kafka server<br />
<Kafka path>/bin/kafka-server-start.sh config/server.properties<br />
Run KafkaListener<br />
Run Spark Stream Processing Program<br />
<Spark path>/bin/spark-submit --packages org.apache.spark:spark-streaming-kafka-0-8_2.11:<Spark version> <program path><br />
