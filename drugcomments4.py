# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:18:18 2024

@author: pdeo
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation # topic analysis


# text analytics packages
from wordcloud import WordCloud  # word clouds
from wordcloud import STOPWORDS
from textblob import TextBlob   # sentiments analysis
import re  # regular expression


# natural language tool package
import nltk
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
import unicodedata
# Download necessary resources if not already downloaded
nltk.download('stopwords')  # for common stopwords
nltk.download('punkt')      # for word tokenizer
nltk.download('wordnet')    # for word map and links
nltk.download('omw-1.4')    # for multilingual wordnet

# Add your additional stop words here
additional_stopwords = ['ha','le','u','wa']

# Get the default English stop words from NLTK
stop_words = nltk.corpus.stopwords.words('english')

# Extend the default stop words list with additional stop words
stop_words.extend(additional_stopwords)


plt.style.use('fivethirtyeight')

####################################### set display option on the console
pd.set_option('display.max_colwidth', None)
pd.set_option("display.max_rows", 25)
pd.set_option("display.expand_frame_repr", True)
pd.set_option('display.width', 1000)

###############################################

############################################ get dataset  and check for accuracy
datafile = 'C:/Users/pdeo/Downloads/7304/Final project/drugComments.csv'
#datafile = 'pmi50A.csv'
twDF = pd.read_csv (datafile, encoding= 'unicode_escape')
print (twDF['benefitsReview'].head(7))
print (twDF.head(7))
print (twDF.shape)
twDF.columns


############################# make a copy of the data and
############################## work with a copy to go through the cleaning process
############################### in case of issues return to the original
twDF_PP = twDF.copy()
twDF_PP['benefitsReview'].head()
twDF_PP.info()


"""
Clean the data by removing  hashtags,  URLs
numbers, special characters like puntuation


"""
twDF_PP['benefitsReview']=twDF_PP['benefitsReview'].str.lower()
twDF_PP['benefitsReview'].head()


def cleanText(text):
    if isinstance(text, str):
        text = re.sub(r'#', '', text)  
        text = re.sub(r'https?:\/\/\S+', '', text)
        text = re.sub(r'[0-9]+|[^\w\s]', '', text)
        return text
    else:
        return text

twDF_PP['benefitsReview'] = twDF_PP['benefitsReview'].apply(cleanText)
twDF_PP['benefitsReview'].head()

############################### use the SKLearn to convert to tokens/words
################# bag of words
# tokenization  of the text
class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

cv = CountVectorizer(tokenizer=LemmaTokenizer(), lowercase=False, stop_words='english', max_df=0.75)
# Replace np.nan values with empty strings
twDF_PP['benefitsReview'] = twDF_PP['benefitsReview'].fillna('')

# Proceed with the CountVectorizer
counts = cv.fit_transform(twDF_PP['benefitsReview'])



DT_matrix = pd.DataFrame(counts.toarray(), columns = cv.get_feature_names_out())
print (DT_matrix.head(7))

TD_matrix = DT_matrix.transpose()

print(TD_matrix.head(7))

TD_matrix['total_count'] = TD_matrix.sum(axis=1)

## focus on the top 25 words


TD_matrix_TOP25 = TD_matrix.sort_values(by ='total_count',ascending=False)[:25]

TOP25 = TD_matrix_TOP25.iloc[:,-1:]

TOP25.info()

print(TOP25)

TOP25.plot.bar(y='total_count', color = {"red", "blue", "green"})
plt.show()


############### exclude top 10

TD_matrix_EX10 = TD_matrix.sort_values(by ='total_count',ascending=False)[10:25]

EXTOP10 = TD_matrix_EX10.iloc[:,-1:]

EXTOP10.info()

print(EXTOP10)

EXTOP10.plot.bar(y='total_count', color = {"red", "blue", "green"})
plt.show()





#################################### more insights using n-gram
"""
ngrams can provide more insigths than a just a single word
"""


def get_ngrams(text, ngram_from=2, ngram_to=2, n=None):
   
    vec = CountVectorizer(ngram_range = (ngram_from, ngram_to),
                          stop_words='english').fit(text)
    bag_of_words = vec.transform(text)
    sum_words = bag_of_words.sum(axis = 0)
    words_freq = [(word, sum_words[0, i]) for word, i in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
    return words_freq[:n]


unigrams = get_ngrams(twDF_PP['benefitsReview'], ngram_from=3, ngram_to=3, n=50)
unigrams_df = pd.DataFrame(unigrams)
unigrams_df.columns=["Trigram", "Frequency"]
unigrams_df.head(10)


######################################### N-gram with NLTK


"""
  A simple function to clean up the data. All the words that
  are not designated as a stop word is then lemmatized after
  encoding and basic regex parsing are performed.
"""
def basic_clean(text):
  wnl = nltk.stem.WordNetLemmatizer()
  stopwords = nltk.corpus.stopwords.words('english')
  text = (unicodedata.normalize('NFKD', text)
    .encode('ascii', 'ignore')
    .decode('utf-8', 'ignore')
    .lower())
  words = re.sub(r'[^\w\s]', '', text).split()
  return [wnl.lemmatize(word) for word in words if word not in stopwords]

words = basic_clean(''.join(str(twDF_PP['benefitsReview'].tolist())))



pd.Series(nltk.ngrams(words, 2)).value_counts()[:10]
pd.Series(nltk.ngrams(words, 3)).value_counts()[:10]

words[:20]  # bag of word

def ngram_filter(doc, word, n):
    tokens = doc.split()
    all_ngrams = ngrams(tokens, n)
    filtered_ngrams = [x for x in all_ngrams if word in x]
    return filtered_ngrams

words = ''.join(str(twDF_PP['benefitsReview'].tolist()))
ngram_filter(words, 'project', 3)


#################### the word cloud ..


stop_words = ["project", "PMP", "Project"] + list(STOPWORDS)

data = dict(zip(TD_matrix.index.tolist(), TD_matrix['total_count'].tolist()))


wc = WordCloud(stopwords=stop_words,width=500, height = 300, random_state = 21, max_font_size = 119).generate_from_frequencies(data)

plt.figure(figsize=(8, 12))

plt.imshow(wc, interpolation = "bilinear")
plt.axis('off')
plt.show()

################################### topic analysis


tf = cv.fit_transform(twDF_PP['benefitsReview']).toarray()
   
tf_feature_names = cv.get_feature_names_out()
 
lda = LatentDirichletAllocation(n_components=6, learning_method="batch", max_iter=25, random_state=0)
 
lda.fit(tf)    
 
 
def display_topics(lda, feature_names, no_top_words):
    topic_dict = {f"Topic {i} words": [feature_names[index] for index in topic.argsort()[:-no_top_words - 1:-1]] for i, topic in enumerate(lda.components_)}
    return pd.DataFrame(topic_dict)
   
no_top_words = 10
display_topics(lda, tf_feature_names, no_top_words)


##############################################################
## sentiment and polarity

def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getPolarity(text):
    return TextBlob(text).sentiment.polarity

twDF_PP['Subjectivity'] = twDF_PP['benefitsReview'].apply(getSubjectivity)
twDF_PP['Polarity'] = twDF_PP['benefitsReview'].apply(getPolarity)
twDF_PP

def getPolarityScore(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
   
twDF_PP['Sentiments'] = twDF_PP['Polarity'].apply(getPolarityScore)  
twDF_PP['Sentiments'].value_counts()


plt.title('Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
twDF_PP['Sentiments'].value_counts().plot(kind= 'bar')
plt.show()


twDF_PP['Polarity'].head()

PositiveDF = twDF_PP.query("Polarity > 0")["benefitsReview"]
NegativeDF = twDF_PP.query("Polarity < 0")["benefitsReview"]

PositiveDF.to_csv('positive.csv')
NegativeDF.to_csv('negative.csv')

import os
print(os.getcwd())