# Twitter senitment analysis

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
import warnings
# %matplotlib inline
warnings.filterwarnings('ignore')

"""## loading dataset"""

df = pd.read_csv('training_data.csv')
df.head()

df.info()

"""## Preprocessing the dataset"""

# removing pattern in the input text
def remove_pattern(input_text, pattern):
  r = re.findall(pattern, input_text)
  for word in r:
    input_text = re.sub(word, "", input_text)
  return input_text

# removing twitter handles (user)
df['clean_tweet'] = np.vectorize(remove_pattern)(df['tweet'], "@[\w]*")

#removing special char, punctuaions and numbers
df['clean_tweet'] = df['clean_tweet'].str.replace("[^a-zA-Z#]", " ")

# removing shorter words
df['clean_tweet'] = df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if  len(w)>3]) )

df.head()

# tokenizing words
tokenized_tweet = df['clean_tweet'].apply( lambda x: x.split())
tokenized_tweet.head()

# stemming
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply( lambda sentence: [stemmer.stem(words) for words in sentence])
tokenized_tweet.head()

# combining words into sentence
for i in range(len(tokenized_tweet)):
  tokenized_tweet[i] = " ".join(tokenized_tweet[i])

df['clean_tweet'] = tokenized_tweet
df.head()

"""## Exploratory data analysis"""

# visualizing frequent words
all_words = " ".join([sentence for sentence in df['clean_tweet'] ])

from wordcloud import WordCloud
wordcloud = WordCloud( width=800, height=500, random_state=42, max_font_size=100 ).generate(all_words)

# plotting graph
plt.figure(figsize= (14,6 ))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# frequent word visualizing +ve
all_words = " ".join([sentence for sentence in df['clean_tweet'][df['label']==0] ])

wordcloud = WordCloud( width=800, height=500, random_state=42, max_font_size=100 ).generate(all_words)

# plotting graph
plt.figure(figsize= (14,6 ))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# frequent word visualizing -ve
all_words = " ".join([sentence for sentence in df['clean_tweet'][df['label']==1] ])

wordcloud = WordCloud( width=800, height=500, random_state=42, max_font_size=100 ).generate(all_words)

# plotting graph
plt.figure(figsize= (14,6 ))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Extracting hashtag
def extract_hashtag(tweets):
  hashtags = []
  # looping words in tweet
  for tweet in tweets:
    ht = re.findall(r"#(\w+)", tweet)
    hashtags.append(ht)
  return hashtags

# extracting hashtags for postive tweets
ht_positive = extract_hashtag(df['clean_tweet'][df['label']==0] )

# extracting hashtags for negative tweets
ht_negative = extract_hashtag(df['clean_tweet'][df['label']==1] )

ht_positive[:10]

# unnesting list
ht_positive = sum(ht_positive, [])
ht_negative = sum(ht_negative, [])

print("Positive Hastags: ",ht_positive[:9])
print("Negative Hastags: ",ht_negative[:9])

# Positive tweets counting
freq = nltk.FreqDist(ht_positive)
d = pd.DataFrame( {'Hashtag': list(freq.keys()), 'Count': list(freq.values()) } )

d.head()

# Plotting top 10 Positive Hashtags
d = d.nlargest(columns='Count', n=10)
plt.figure(figsize=(12, 4))
ax = sns.barplot(data=d, x='Hashtag', y='Count')
ax.set(xlabel='Positive Hashtags', ylabel='Count')
# plt.xticks(rotation=45)
plt.show()

# Negative tweets counting
freq = nltk.FreqDist(ht_negative)
d = pd.DataFrame( {'Hashtag': list(freq.keys()), 'Count': list(freq.values()) } )

d.head()

# Plotting top 10 Negative Hashtags
d = d.nlargest(columns='Count', n=10)
plt.figure(figsize=(12, 4))
ax = sns.barplot(data=d, x='Hashtag', y='Count')
ax.set(xlabel='Negative Hashtags', ylabel='Count')
# plt.xticks(rotation=45)
plt.show()

"""## Input Split"""

# feature extraction
from sklearn.feature_extraction.text import CountVectorizer
bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(df['clean_tweet'])

# bow[0].toarray()

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(bow, df['label'], random_state=42, test_size=0.25)

"""## Model Training"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

# training
model = LogisticRegression()
model.fit(x_train, y_train)

# testing
pred = model.predict(x_test)
f1_score(y_test, pred)

accuracy_score(y_test,pred)

# use probability to get output
pred_prob = model.predict_proba(x_test)
pred = pred_prob[:, 1] >= 0.3
pred = pred.astype(np.int)

f1_score(y_test, pred)

accuracy_score(y_test,pred)

pred_prob[0][1] >= 0.3

