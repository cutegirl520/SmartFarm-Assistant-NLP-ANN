
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import re
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import load_model
import tensorflow




def trainIntentModel():
    # Load the dataset and prepare it to the train the model

    # Importing dataset and splitting into words and labels
    dataset = pd.read_csv('datasets/intent.csv', names=["Query", "Intent"])

    X = dataset["Query"]
    y = dataset["Intent"]

    unique_intent_list = list(set(y))

    print("Intent Dataset successfully loaded!")
    
    # Clean and prepare the intents corpus
    queryCorpus = []
    ps = PorterStemmer()

    for query in X:
        query = re.sub('[^a-zA-Z]', ' ', query)

        # Tokenize sentence
        query = query.split(' ')

        # Lemmatizing
        tokenized_query = [ps.stem(word.lower()) for word in query]
