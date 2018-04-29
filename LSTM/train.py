
import re
import tensorflow as tf
import numpy as np

inputText=[]
label=[]

for i in range(24):
	inputText.append("bad")
	label.append([0,1])

print(len(inputText))



strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
numDimensions = 300
maxSeqLength = 250
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000

wordsList = np.load('wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')


def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def getSentenceMatrix(sentence):
    arr = np.zeros([batchSize, maxSeqLength])
    size = len(sentence)
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    for i in range(size):

    
        cleanedSentence = cleanSentences(sentence[i])
        split = cleanedSentence.split()
        for indexCounter,word in enumerate(split):
            try:
                sentenceMatrix[i,indexCounter] = wordsList.index(word)
            except ValueError:
                sentenceMatrix[i,indexCounter] = 399999 #Vector for unkown words
    return sentenceMatrix








tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)




sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

inputMatrix = getSentenceMatrix(inputText)

sess.run(optimizer, {input_data: inputMatrix, labels: label})

def savemodel():

    save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
    print("saved to %s" % save_path)
