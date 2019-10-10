import os
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
# %matplotlib inline
import pickle
import warnings

warnings.filterwarnings('ignore')

import seaborn

train = pd.read_csv('train.csv')
Y = train.target
#test = pd.read_csv('test.csv.zip')
#test_ID = test.ID
#%%

def autolabel(arrayA):
    ''' label each colored square with the corresponding data value.
    If value > 20, the text is in black, else in white.
    '''
    arrayA = np.array(arrayA)
    for i in range(arrayA.shape[0]):
        for j in range(arrayA.shape[1]):
            plt.text(j, i, "%.2f" % arrayA[i, j], ha='center', va='bottom', color='w')


def hist_it(feat):
    plt.figure(figsize=(16, 4))
    feat[Y == 0].hist(bins=range(int(feat.min()), int(feat.max() + 2)), normed=True, alpha=0.8)
    feat[Y == 1].hist(bins=range(int(feat.min()), int(feat.max() + 2)), normed=True, alpha=0.5)
    plt.ylim((0, 1))


def gt_matrix(feats, sz=16):
    a = []
    for i, c1 in enumerate(feats):
        b = []
        for j, c2 in enumerate(feats):
            mask = (~train[c1].isnull()) & (~train[c2].isnull())
            if i >= j:
                b.append((train.loc[mask, c1].values >= train.loc[mask, c2].values).mean())
            else:
                b.append((train.loc[mask, c1].values > train.loc[mask, c2].values).mean())

        a.append(b)

    plt.figure(figsize=(sz, sz))
    plt.imshow(a, interpolation='None')
    _ = plt.xticks(range(len(feats)), feats, rotation=90)
    _ = plt.yticks(range(len(feats)), feats, rotation=0)
    autolabel(a)


# Number of NaNs for each object
train.isnull().sum(axis=1).head(15)
# %% md DATASET Cleaning
traintest = pd.concat([train, test], axis=0)
# `dropna = False` makes nunique treat NaNs as a distinct value
feats_counts = train.nunique(dropna=False)
# feats_counts.sort_values()[:10]
constant_features = feats_counts.loc[feats_counts == 1].index.tolist()
print(constant_features)

traintest.drop(constant_features, axis=1, inplace=True)

# %% Remove duplicated features

traintest.fillna('NaN', inplace=True)
train_enc = pd.DataFrame(index=train.index)

for col in tqdm_notebook(traintest.columns):
    train_enc[col] = train[col].factorize()[0]
# train_enc[col] = train[col].map(train[col].value_counts())

dup_cols = {}

for i, c1 in enumerate(tqdm_notebook(train_enc.columns)):
    for c2 in train_enc.columns[i + 1:]:
        if c2 not in dup_cols and np.all(train_enc[c1] == train_enc[c2]):
            dup_cols[c2] = c1

# dup_cols


pickle.dump(dup_cols, open('dup_cols.p', 'w'), protocol=pickle.HIGHEST_PROTOCOL)

# %% Determine types
nunique = train.nunique(dropna=False)
mask = (nunique.astype(float) / train.shape[0] > 0.8)
# train.loc[:, mask]
mask = (nunique.astype(float) / train.shape[0] < 0.8) & (nunique.astype(float) / train.shape[0] > 0.4)
# train.loc[:25, mask]

train['VAR_0015'].value_counts()

cat_cols = list(train.select_dtypes(include=['object']).columns)
num_cols = list(train.select_dtypes(exclude=['object']).columns)

train.replace('NaN', -999, inplace=True)

# select first 42 numeric features
feats = num_cols[:42]

# build 'mean(feat1 > feat2)' plot
gt_matrix(feats, 16)

# %%
hist_it(train['VAR_0002'])
plt.ylim((0, 0.05))
plt.xlim((-10, 1010))

hist_it(train['VAR_0003'])
plt.ylim((0, 0.03))
plt.xlim((-10, 1010))

train['VAR_0004_mod50'] = train['VAR_0004'] % 50
hist_it(train['VAR_0004_mod50'])
plt.ylim((0, 0.6))

# %% Categorical features
train.loc[:, cat_cols].head().T

date_cols = [u'VAR_0073', 'VAR_0075',
             u'VAR_0156', u'VAR_0157', u'VAR_0158', 'VAR_0159',
             u'VAR_0166', u'VAR_0167', u'VAR_0168', u'VAR_0169',
             u'VAR_0176', u'VAR_0177', u'VAR_0178', u'VAR_0179',
             u'VAR_0204',
             u'VAR_0217']

for c in date_cols:
    train[c] = pd.to_datetime(train[c], format='%d%b%y:%H:%M:%S')
    test[c] = pd.to_datetime(test[c], format='%d%b%y:%H:%M:%S')

mask = (~train[c1].isnull()) & (~train[c2].isnull())
# sc2(train.loc[mask, c1].values, train.loc[mask, c2].values, c=train.loc[mask, 'target'].values)
