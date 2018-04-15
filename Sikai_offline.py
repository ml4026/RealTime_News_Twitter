
# coding: utf-8

# In[1]:


import csv, re, random

FULLDATA = 'training.1600000.processed.noemoticon.csv'
TESTDATA = 'testdata.manual.2009.06.14.csv'
OUTFILE = 'test.csv'
#Format
#Data file format has 6 fields:
#0 - the polarity of the tweet (0 = negative, 2 = neutral, 4 = positive)
#1 - the id of the tweet (2087)
#2 - the date of the tweet (Sat May 16 23:58:44 UTC 2009)
#3 - the query (lyx). If there is no query, then this value is NO_QUERY.
#4 - the user that tweeted (robotickilldozr)
#5 - the text of the tweet (Lyx is cool)

POLARITY= 0 # in [0,5]
TWID    = 1
DATE    = 2
SUBJ    = 3 # NO_QUERY
USER    = 4
TEXT    = 5

def get_class( polarity ):
    if polarity in ['0', '1']:
        #negative
        return 0
    elif polarity in ['3', '4']:
        #positive
        return 1
    elif polarity == '2':
        #neural
        return 2
    else:
        #error
        return 3
    
def getNormalisedTweets(in_file):
    fp = open(in_file , 'r',encoding='gbk',errors='ignore')
    rd = csv.reader(fp, delimiter=',', quotechar='"' )
    #print in_file, countlines( in_file )
    tweets = []
    for row in rd:
        row[0]=get_class(row[0])
        tweets.append(row)
    return tweets

def deleteNeu(inputlist):
    for i in inputlist:
        if i[0]==2 or i[0]==3:
            inputlist.remove(i)
    return inputlist



# In[2]:


testdata=getNormalisedTweets(TESTDATA)
testdataNoNeu=deleteNeu(testdata)
traindata=getNormalisedTweets(FULLDATA)
traindataNoNeu=deleteNeu(traindata)
print(len(testdata))
print(testdataNoNeu)


# In[3]:


import re

# Hashtags
hash_regex = re.compile(r"#(\w+)")
def hash_repl(match):
    return '__HASH_'+match.group(1).upper()

# Handels
hndl_regex = re.compile(r"@(\w+)")
def hndl_repl(match):
    return '__HNDL_'+match.group(1).upper()

# URLs
url_regex = re.compile(r"(http|https|ftp)://[a-zA-Z0-9\./]+")

# Spliting by word boundaries
word_bound_regex = re.compile(r"\W+")

# Repeating words like hurrrryyyyyy
rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE);
def rpt_repl(match):
    return match.group(1)+match.group(1)

# Emoticons
emoticons=[('__EMOT_SMILEY',[':-)', ':)', '(:', '(-:', ] ),('__EMOT_LAUGH',[':-D', ':D', 'X-D', 'XD', 'xD', ] ),
           ('__EMOT_LOVE',['<3', ':\*', ] ),
           ('__EMOT_WINK',[';-)', ';)', ';-D', ';D', '(;', '(-;', ] ),
           ('__EMOT_FROWN',[':-(', ':(', '(:', '(-:', ] ),
           ('__EMOT_CRY',[':,(', ':\'(', ':"(', ':(('] )]



# Punctuations
punctuations = [('__PUNC_EXCL',['!', '¡', ] ),
                ('__PUNC_QUES',['?', '¿', ] ),
                ('__PUNC_ELLP',['...', '…', ] )]

#For emoticon regexes
def escape_paren(arr):
    return [text.replace(')', '[)}\]]').replace('(', '[({\[]') for text in arr]
def regex_union(arr):
    return '(' + '|'.join( arr ) + ')'
emoticons_regex = [ (repl, re.compile(regex_union(escape_paren(regx))) ) for (repl, regx) in emoticons ]

#For punctuation replacement
def punctuations_repl(match):
    text = match.group(0)
    repl = []
    for (key, parr) in punctuations :
        for punc in parr :
            if punc in text:
                repl.append(key)
    if( len(repl)>0 ) :
        return ' '+' '.join(repl)+' '
    else :
        return ' '

def processHashtags( text, subject='', query=[]):
    return re.sub( hash_regex, hash_repl, text )

def processHandles( text, subject='', query=[]):
    return re.sub( hndl_regex, hndl_repl, text )

def processUrls( text, subject='', query=[]):
    return re.sub( url_regex, ' __URL ', text )

def processEmoticons( 	text, subject='', query=[]):
    for (repl, regx) in emoticons_regex :
        text = re.sub(regx, ' '+repl+' ', text)
    return text

def processPunctuations( text, subject='', query=[]):
    return re.sub( word_bound_regex , punctuations_repl, text )

def processRepeatings( 	text, subject='', query=[]):
    return re.sub( rpt_regex, rpt_repl, text )


def processAll(text, subject='', query=[]):
    if(len(query)>0):
        query_regex = "|".join([ re.escape(q) for q in query])
        text = re.sub( query_regex, '__QUER', text, flags=re.IGNORECASE )
    text = re.sub( hash_regex, hash_repl, text )
    text = re.sub( hndl_regex, hndl_repl, text )
    text = re.sub( url_regex, ' __URL ', text )
    for (repl, regx) in emoticons_regex :
        text = re.sub(regx, ' '+repl+' ', text)
    text = text.replace('\'','')
    text = re.sub( word_bound_regex , punctuations_repl, text )
    text = re.sub( rpt_regex, rpt_repl, text )
    return text

testext=processAll('#SikaiZHou 10 tips for healthy eating ? ResultsBy Fitness Blog :: Fitness ... http://bit.ly/62gFn. ;-)')
print(type(testext))


# In[5]:


import nltk
import sys
def getData(data,feature_set):
    add_ngram_feat = feature_set.get('ngram', 1)
    add_negtn_feat = feature_set.get('negtn', False)
    
    stemmer=nltk.stem.PorterStemmer()
    all_tweets=[]
    for i in data:
        label=i[0]
        text=i[1]
        
        words=[word if(word[0:2]=='__') else word.lower() for word in text.split() if len(word)>=3]
        words=[stemmer.stem(w) for w in words]
        all_tweets.append((words,label))
        
    train_tweets = [x for i,x in enumerate(all_tweets) if i % 5 !=0]
    test_tweets  = [x for i,x in enumerate(all_tweets) if i % 5 ==0] 
    

        
    
    def get_word_features(words):
        bag = {}
        words_uni = [ 'has(%s)'% ug for ug in words ]
        if( add_ngram_feat>=2 ):
            words_bi  = [ 'has(%s)'% ','.join(map(str,bg)) for bg in nltk.bigrams(words) ]
        else:
            words_bi  = []
        if( add_ngram_feat>=3 ):
            words_tri = [ 'has(%s)'% ','.join(map(str,tg)) for tg in nltk.trigrams(words) ]
        else:
            words_tri = []
        for f in words_uni+words_bi+words_tri:
            bag[f] = 1
        #bag = collections.Counter(words_uni+words_bi+words_tri)
        return bag
    
    
    negtn_regex = re.compile( r"""(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$)|n't""", re.X)
    def get_negation_features(words):
        INF = 0.0
        negtn = [ bool(negtn_regex.search(w)) for w in words ]
        left = [0.0] * len(words)
        prev = 0.0
        for i in range(0,len(words)):
            if( negtn[i] ):
                prev = 1.0
            left[i] = prev
            prev = max( 0.0, prev-0.1)
        right = [0.0] * len(words)
        prev = 0.0
        for i in reversed(range(0,len(words))):
            if( negtn[i] ):
                prev = 1.0
            right[i] = prev
            prev = max( 0.0, prev-0.1)
        return dict( zip(['neg_l('+w+')' for w in  words] + ['neg_r('+w+')' for w in  words],left + right ) )
    
    def extract_features(words):
        features = {}
        word_features = get_word_features(words)
        features.update( word_features )
        if add_negtn_feat :
            negation_features = get_negation_features(words)
            features.update( negation_features )
        sys.stderr.write( '\rfeatures extracted for ' + str(extract_features.count) + ' tweets' )
        return features

    extract_features.count = 0
    v_train = nltk.classify.apply_features(extract_features,train_tweets)
    v_test  = nltk.classify.apply_features(extract_features,test_tweets)
    return (v_train, v_test)

rawdata=[[1,'#SikaiZHou 10 tips for healthy eating ? ResultsBy Fitness Blog :: Fitness ... http://bit.ly/62gFn. ;-)']]
featuresetting={'ngram': 1,'negtn':True}
print(getData(rawdata,featuresetting))


# In[6]:


def train_classify(tweets,feature_set):
    (v_train,v_test)=getData(tweets,feature_set)
    CLASSIFIER = nltk.classify.NaiveBayesClassifier
    classifier_used=CLASSIFIER.train(v_train)
    accuracy_used = nltk.classify.accuracy(classifier_used, v_test)
    print(accuracy_used)
    return classifier_used


# In[8]:


print(traindataNoNeu[1])


# In[9]:


rawdata=[]
for i in traindataNoNeu:
    rawdata.append([i[0],processAll(i[-1])])
print(rawdata[0])


# In[10]:


featuresetting={'ngram': 1,'negtn':True}
bayes=train_classify(rawdata,featuresetting)


# In[15]:


test={'has(#sikaizh)': 1, 'has(tip)': 1, 'has(for)': 1, 'has(healthi)': 1, 'has(eat)': 1, 'has(resultsbi)': 1, 'has(fit)': 1, 'has(blog)': 1, 'has(...)': 1, 'has(http://bit.ly/62gfn.)': 1, 'has(;-))': 1, 'has(#sikaizh,tip)': 1, 'has(tip,for)': 1, 'has(for,healthi)': 1, 'has(healthi,eat)': 1, 'has(eat,resultsbi)': 1, 'has(resultsbi,fit)': 1, 'has(fit,blog)': 1, 'has(blog,fit)': 1, 'has(fit,...)': 1, 'has(...,http://bit.ly/62gfn.)': 1, 'has(http://bit.ly/62gfn.,;-))': 1, 'neg_l(#sikaizh)': 0.0, 'neg_l(tip)': 0.0, 'neg_l(for)': 0.0, 'neg_l(healthi)': 0.0, 'neg_l(eat)': 0.0, 'neg_l(resultsbi)': 0.0, 'neg_l(fit)': 0.0, 'neg_l(blog)': 0.0, 'neg_l(...)': 0.0, 'neg_l(http://bit.ly/62gfn.)': 0.0, 'neg_l(;-))': 0.0, 'neg_r(#sikaizh)': 0.0, 'neg_r(tip)': 0.0, 'neg_r(for)': 0.0, 'neg_r(healthi)': 0.0, 'neg_r(eat)': 0.0, 'neg_r(resultsbi)': 0.0, 'neg_r(fit)': 0.0, 'neg_r(blog)': 0.0, 'neg_r(...)': 0.0, 'neg_r(http://bit.ly/62gfn.)': 0.0, 'neg_r(;-))': 0.0}
print(bayes.classify(test))


# In[16]:


def processinput(data,feature_set):
    add_ngram_feat = feature_set.get('ngram', 1)
    add_negtn_feat = feature_set.get('negtn', False)
    
    stemmer=nltk.stem.PorterStemmer()
    all_tweets=[]
    for i in data:
        label=0
        text=i
        
        words=[word if(word[0:2]=='__') else word.lower() for word in text.split() if len(word)>=3]
        words=[stemmer.stem(w) for w in words]
        all_tweets.append((words,label))
        
    train_tweets = [x for i,x in enumerate(all_tweets)]
    

        
    
    def get_word_features(words):
        bag = {}
        words_uni = [ 'has(%s)'% ug for ug in words ]
        if( add_ngram_feat>=2 ):
            words_bi  = [ 'has(%s)'% ','.join(map(str,bg)) for bg in nltk.bigrams(words) ]
        else:
            words_bi  = []
        if( add_ngram_feat>=3 ):
            words_tri = [ 'has(%s)'% ','.join(map(str,tg)) for tg in nltk.trigrams(words) ]
        else:
            words_tri = []
        for f in words_uni+words_bi+words_tri:
            bag[f] = 1
        #bag = collections.Counter(words_uni+words_bi+words_tri)
        return bag
    
    
    negtn_regex = re.compile( r"""(?:^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)$)|n't""", re.X)
    def get_negation_features(words):
        INF = 0.0
        negtn = [ bool(negtn_regex.search(w)) for w in words ]
        left = [0.0] * len(words)
        prev = 0.0
        for i in range(0,len(words)):
            if( negtn[i] ):
                prev = 1.0
            left[i] = prev
            prev = max( 0.0, prev-0.1)
        right = [0.0] * len(words)
        prev = 0.0
        for i in reversed(range(0,len(words))):
            if( negtn[i] ):
                prev = 1.0
            right[i] = prev
            prev = max( 0.0, prev-0.1)
        return dict( zip(['neg_l('+w+')' for w in  words] + ['neg_r('+w+')' for w in  words],left + right ) )
    
    def extract_features(words):
        features = {}
        word_features = get_word_features(words)
        features.update( word_features )
        if add_negtn_feat :
            negation_features = get_negation_features(words)
            features.update( negation_features )
        sys.stderr.write( '\rfeatures extracted for ' + str(extract_features.count) + ' tweets' )
        return features

    extract_features.count = 0
    v_train = nltk.classify.apply_features(extract_features,train_tweets)
    return v_train


# In[22]:


test1='#SikaiZHou 10 tips for healthy eating ? ResultsBy Fitness Blog :: Fitness ... http://bit.ly/62gFn. ;-)'
test11=processAll(test1)
test11=[test11]
test111=processinput(test11,featuresetting)
for i in test111:
    feature,label=i[0],i[1]
    print(feature)
    print('predictlabel',bayes.classify(feature))


# In[23]:





# In[25]:


import pickle
f = open('my_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()

