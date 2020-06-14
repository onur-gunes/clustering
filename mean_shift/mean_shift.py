import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing
import pandas as pd
import re

df = pd.read_excel('titanic.xls')
original_df = pd.DataFrame.copy(df)

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

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan

for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i]

n_clusters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group']) == float(i) ]
    survival_cluster = temp_df[(temp_df['survived'] == 1)]
    survival_rate = len(survival_cluster) / len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)

for i in range(len(survival_rates)):
    print(original_df[ (original_df['cluster_group'] == i) ].describe())
