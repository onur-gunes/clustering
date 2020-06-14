import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import re

df = pd.read_excel('titanic.xls')
df.drop(['body', 'name', 'ticket', 'home.dest', 'boat', 'body'], 1, inplace=True)
df['embarked'] = df['embarked'].fillna('S')
df['fare'] = df['fare'].fillna(-99999)
df['fare'] = df['fare'].astype(int)
df['age'] = df['age'].fillna(-99999)
df['age'] = df['age'].astype(int)
df['cabin'] = df['cabin'].fillna('Unknown')
decks = ['A','B','C','D','E','F','G','Unknown']

for i in range(len(df['cabin'])):
    for word in re.findall(r"\w+", df['cabin'][i]):
        for deck in decks:
            if deck in word:
                df['cabin'].replace(df['cabin'][i], deck, inplace=True)

for i in range(len(df['cabin'])):
    if df['cabin'][i] not in decks:
        df['cabin'].replace(df['cabin'][i], 'Unknown', inplace=True)

df.fillna(-99999, inplace=True)
df = pd.get_dummies(df, columns=["sex", 'embarked', 'cabin'])

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct +=1
print(correct/len(X))
