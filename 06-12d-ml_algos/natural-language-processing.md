# Natural Language Processing (NLP)

**First of all, what is text classification?**

Text Classification is an automated process of classification of text into predefined categories. We can classify Emails into spam or non-spam, news articles into different categories like Politics, Stock Market, Sports, etc.

This can be done with the help of Natural Language Processing and different Classification Algorithms like Naive Bayes, Support Vector Machine and even Neural Networks in Python.

**What is Natural language processing?**

Natural Language Processing (NLP) is an Artificial Intelligence (AI) field that enables computer programs to recognize, interpret, and manipulate human languages.

## Steps to build an NLP model

- **Step 1:** Add the required libraries.

- **Step 2:** Set a random seed. This is used to reproduce the same result every time if the script is kept consistent otherwise each run will produce different results. The seed can be set to any number.

- **Step 3:** Load the dataset

- **Step 4:** Preprocess the content of each text. This is a very important step.

Real-world data is often incomplete, inconsistent, and/or lacking in certain behaviors or trends, and is likely to contain many errors. Data pre-processing is a proven method of resolving such issues.This will help in getting better results through the classification algorithms.

- **Step 5:** Separate train and test datasets

- **Step 6:** Encoding

Label encode the target variable — This is done to transform Categorical data of string type in the data set into numerical values which the model can understand.

- **Step 7:** Bag of words (vectorization)

It is the process of converting sentence words into numerical feature vectors. It is useful as models require data to be in numeric format. So if the word is present in that particular sentence then we will put 1 otherwise 0. The most popular method is called TF-IDF. It stands for “Term Frequency — Inverse Document” Frequency. TF-IDF are word frequency scores that try to highlight words that are more interesting, e.g. frequent in a document but not across documents.We can also decide to remove stop words by adding a parameter called “stop_words” in “TFidfVectorizer”.


- **Step 8:** Use the ML Algorithm to Predict the outcome

### Detail of data pre-processing steps

You can always add or remove steps which best suits the data set you are dealing with:

1. Remove Blank rows in Data, if any

```py
Dataset['text'].dropna(inplace=True)
```

2. Change all the text to lower case because python interprets upper and lower case differently.

```py
Dataset['text'] = [entry.lower() for entry in Dataset['text']]
```

3. Word Tokenization: It is the process of breaking a stream of text up into words, phrases, symbols, or other meaningful elements called tokens. The list of tokens becomes input for further processing. NLTK Library has word_tokenize and sent_tokenize to easily break a stream of text into a list of words or sentences, respectively.

```py
Dataset['text']= [word_tokenize(entry) for entry in Dataset['text']]
```

4. Remove Stop words: It removes all the frequently used words such as “I, or, she, have, did, you, to”.

5. Remove Non-alpha text and punctuation characters

6. Word Lemmatization/ Stemming: It is the process of reducing the inflectional forms of each word into a common base or root.
The difference between lemma and stem is that a stemmer operates on a single word without knowledge of the context, and therefore cannot discriminate between words which have different meanings depending on part of speech. However, stemmers are typically easier to implement and run faster, and the reduced accuracy may not matter for some applications. For example, if the message contains some error word like “frei” which might be misspelled for “free”. Stemmer will stem or reduce that error word to its root word i.e. “fre”. As a result, “fre” is the root word for both “free” and “frei”.

### Use Case

```py
# Step 1
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score

# Step 2
np.random.seed(500)

# Step 3
Corpus = pd.read_csv(r"C:\Users\gunjit.bedi\Desktop\NLP Project\corpus.csv",encoding='latin-1')

# Step 4
## Load our stopwords and punctuation and take a look at their content.

import string
import nltk

nltk.download('stopwords')
nltk.download('punkt')
stopwords = nltk.dataset.stopwords.words('english')
punctuation = string.punctuation
print(stopwords[:5])
print(punctuation)
>>>   ['i', 'me', 'my', 'myself', 'we']
>>>   !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~

## Define a pre-processing function, that results in a list of tokens without punctuation, stopwords or capital letters.
## We’ll use lambda to apply the function and store it as an additional column named “processed” in our data frame.

def pre_process(sms):
   remove_punct = "".join([word.lower() for word in sms if word not 
                  in punctuation])
   tokenize = nltk.tokenize.word_tokenize(remove_punct)
   remove_stopwords = [word for word in tokenize if word not in
                       stopwords]
   return remove_stopwords
data['processed'] = data['sms'].apply(lambda x: pre_process(x))
print(data['processed'].head())
>>>   0    [go, jurong, point, crazy, available, bugis, n...
>>>   1    [ok, lar, joking, wif, u, oni]
>>>   2    [free, entry, 2, wkly, comp, win, fa, cup, fin...
>>>   3    [u, dun, say, early, hor, u, c, already, say]
>>>   4    [nah, dont, think, goes, usf, lives, around, t...

# Step 5
x_train , x_test , y_train , y_test = ttsplit(x_new,y_new,test_size=0.2,shuffle=True)

# Step 6
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)

# Step 7

# Step 8
## fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
## predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
## Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
```















