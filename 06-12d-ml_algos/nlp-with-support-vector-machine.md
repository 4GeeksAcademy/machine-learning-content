# Support Vector Machine

Support Vector Machine (SVM) is a supervised learning algorithm so we need to have a labeled dataset to be able to use SVM. It can be used for regression and classification problems and it can be of linear and non linear type. The main objective of SVM is to find a hyperplane in an N( total number of features)-dimensional space that differentiates the data points. So we need to find a plane that creates the maximum margin between two data point classes, which means finding the line for which the distance of the closest points is the farthest possible.

Let's see some graphs where the algorithm is trying to find the maximum distance between the closest points:

![svm](../assets/svm.jpg)

![svm2](../assets/svm2.jpg)

![svm3](../assets/svm3.jpg)

We can see that the third graph between lines is the greatest distance we can observe in order to put between our two groups.

To completely understand the Support Vector Machine terminology, let's look at the parts of it:

![svm_terminology](../assets/svm_terminology.jpg)

The margin needs to be equal on both sides. 

To sum up some possible uses of SVM models, they can be used for:

-Linear classification

-Non linear classification

-Linear Regression

-Non linear Regression

## Effect of small margins

The maximum margin classifier will separate the categories. If you are familiar with the bias-variance trade off, you will know that if you are putting a line that is very well adjusted for your dataset, you are reducing the bias and increasing the variance. If you are not familiar yet, what it says is that when you have high bias, it means that your model might not be fully tuned to your dataset so will have more misclassification, but if you have a low bias and high variance it means that your model will be very very well tuned to your dataset and that can lead to overfitting meaning that it will perform very well on your training set but in your testing set you will have more error rate.

Because your margins are so small, you may increase the chance of misclassify new data points. T

How do we deal with that?

-Increase margins by including some misclassified data points between your margins. Increase bias and reduce variance can be done by controlling one of the SVM hyperparameters, the 'C' parameter.

A low C value increases bias and decreases variance. If values are too extreme you may underfit.

A high C value decreases bias and increases variance. If values are too extreme you may overfit.

We can determine the best C value by doing cross validation or tuning with validation set.

**What common kernels can be used for SVM?**

1. Linear 

2. Polynomial

3. Gaussian RBF

4. Sigmoid

**Why is it important to scale features before using SVM?**

SVM tries to fit the widest gap between all classes, so unscaled features can cause some features to have a significantly larger or smaller impact on how the SVM split is created.

**Can SVM produce a probability score along with its classification prediction?**

No.

## Natural Language Processing (NLP)?

**First of all, what is text classification?**

Text Classification is an automated process of classification of text into predefined categories. We can classify Emails into spam or non-spam, news articles into different categories like Politics, Stock Market, Sports, etc.

This can be done with the help of Natural Language Processing and different Classification Algorithms like Naive Bayes, Support Vector Machine and even Neural Networks in Python.

**What is Natural language processing?**

Natural Language Processing (NLP) is an Artificial Intelligence (AI) field that enables computer programs to recognize, interpret, and manipulate human languages.

### Steps to preprocess a NLP model

- **Step 1:** Add the required libraries.

```py
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
```

- **Step 2:** Set a random seed. This is used to reproduce the same result every time if the script is kept consistent otherwise each run will produce different results. The seed can be set to any number.

```py
np.random.seed(500)
```

- **Step 3:** Load the dataset

```py
Corpus = pd.read_csv(r"C:\Users\gunjit.bedi\Desktop\NLP Project\corpus.csv",encoding='latin-1')
```

- **Step 4:** Preprocess the content of each text. This is a very important step.

Real-world data is often incomplete, inconsistent, and/or lacking in certain behaviors or trends, and is likely to contain many errors. Data pre-processing is a proven method of resolving such issues.This will help in getting better results through the classification algorithms.

Here’s the complete data pre-processing steps, you can always add or remove steps which best suits the data set you are dealing with:

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

Let's see an example on how to load our stopwords and punctuation and take a look at their content:

```py
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
```

Now let's define a pre-processing function, that results in a list of tokens without punctuation, stopwords or capital letters.
We’ll use lambda to apply the function and store it as an additional column named “processed” in our data frame.

```py
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
```

- **Step 5:** Separate train and test datasets

```py
x_train , x_test , y_train , y_test = ttsplit(x_new,y_new,test_size=0.2,shuffle=True)
```

- **Step 6:** Encoding

Label encode the target variable — This is done to transform Categorical data of string type in the data set into numerical values which the model can understand.

```py
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)
```

- **Step 7:** Bag of words (vectorization)

It is the process of converting sentence words into numerical feature vectors. It is useful as models require data to be in numeric format. So if the word is present in that particular sentence then we will put 1 otherwise 0. The most popular method is called TF-IDF. It stands for “Term Frequency — Inverse Document” Frequency. TF-IDF are word frequency scores that try to highlight words that are more interesting, e.g. frequent in a document but not across documents.We can also decide to remove stop words by adding a parameter called “stop_words” in “TFidfVectorizer”.


- **Step 8:** Use the ML Algorithm to Predict the outcome

```py
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,Train_Y)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, Test_Y)*100)
```



Source:

https://medium.com/analytics-vidhya/how-to-build-a-simple-sms-spam-filter-with-python-ee777240fc

https://projectgurukul.org/spam-filtering-machine-learning/

https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34

