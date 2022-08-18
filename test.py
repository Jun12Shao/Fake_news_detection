import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from nltk.stem import PorterStemmer
import re
import joblib

ps = PorterStemmer()


def filter_function(str):
    if str=='' or str=='\n':
        return False
    else:
        return True

def data_cleaning(dataframe):
    n=len(dataframe)
    data=np.empty(n, dtype=object)
    for i in tqdm(range(n)):
        text=dataframe['title'][i]+' '+dataframe['author'][i]+' '+dataframe['text'][i]
        text=text.lower()
        text=list(filter(filter_function, re.split('[^a-zA-Z]', text)))
        text =' '.join([ps.stem(word) for word in text])
        data[i]=text
    return data

## data loading
test_data_path = "data/test.csv"
test_label_path="data/labels.csv"
train_path = "data/train.csv"

train_data = pd.read_csv(train_path,keep_default_na=False)
test_data =  pd.read_csv(test_data_path,keep_default_na=False)
test_label = pd.read_csv(test_label_path, keep_default_na=False)

## data cleaning
train_data=data_cleaning(train_data)
test_data=data_cleaning(test_data)

## extract tf-df frequency features
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english")

train_data = vectorizer.fit_transform(train_data)
X_test= vectorizer.transform(test_data)
y_test= test_label['label']

## model loading

filenames = ['svm_cls.sav','decision_tree.sav','random_forest.sav','nn_cls.sav']

## testing
for filename in filenames:
    print(filename)

    ## model loading
    loaded_model = joblib.load('result/'+filename)

    y_pred=loaded_model.predict(X_test)
    f1_score = metrics.f1_score(y_test, y_pred)
    print('f1:',f1_score)

    result = loaded_model.score(X_test, y_test)
    print('Accuracy:', result)







