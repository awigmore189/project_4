import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import sklearn as sk
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.grid_search import GridSearchCV
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier


sns.set_style('white')
%matplotlib inline


# general cleaning
data = pd.read_csv('evergreen.tsv', delimiter='\t')
data = data.replace('?', np.nan)

data['alchemy_category_score'] = pd.to_numeric(data['alchemy_category_score'])
data['is_news'] = pd.to_numeric(data['is_news'])
data['news_front_page'] =pd.to_numeric(data['news_front_page'])
data.dtypes

data.dropna(axis = 0, inplace=True)

# separating variable types
maskNumermical = ['alchemy_category_score', 'avglinksize', 'commonlinkratio_1',
       'commonlinkratio_2', 'commonlinkratio_3', 'commonlinkratio_4',
       'compression_ratio', 'embed_ratio', 'frameTagRatio',
       'html_ratio', 'image_ratio', 'linkwordscore',
       'non_markup_alphanum_characters', 'numberOfLinks', 'numwords_in_url',
       'parametrizedLinkRatio', 'spelling_errors_ratio']

maskCat = ['alchemy_category','lengthyLinkDomain',
            'news_front_page', 'framebased', 'hasDomainLink']


# modeling with only numerical
lr = LogisticRegression()
y = data['label']
X = data[maskNumermical]
X_train, X_test, y_train, y_test = train_test_split(X, y)

lr.fit(X_train, y_train)
lr.predict(X_test)
lr.score(X_test, y_test)

# modeling with categorical
dummies = pd.get_dummies(data['alchemy_category'])
second_model = pd.concat([X, dummies], axis = 1)

X2_train, X2_test, y2_train, y2_test = train_test_split(second_model, y)
lr2 = LogisticRegression()
lr2.fit(X2_train, y2_train)
lr2.predict(X2_test)
lr2.score(X2_test, y2_test)

# modeling with cross_validation
lrCV = LogisticRegressionCV()
lrCV.fit(X2_train, y2_train)
lrCV.predict(X2_test)
lrCV.score(X2_test, y2_test)

# models with pre normalized values & inclusion of ALL categorical variables
dummies2 = pd.get_dummies(data[maskCat])
data2 = pd.concat([X, dummies2], axis = 1)
data2 = normalize(data2, norm = 'l2')

X3_train, X3_test, y3_train, y3_test = train_test_split(data2, y)
lr3 = LogisticRegression()
lr3.fit(X3_train, y3_train)
lr3.predict(X3_test)
lr3.score(X3_test, y3_test)

lrCV2 = LogisticRegressionCV()
lrCV2.fit(X3_train, y3_train)

lrCV2.score(X3_test, y3_test)


# Part 5 - gridsearch with Logistic Regression
model = LogisticRegression()
param_grid = {'penalty':('l1', 'l2'),'C':(.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0)}
logGrid = GridSearchCV(model, param_grid)

logGrid.fit(X3_train, y3_train)
logGrid.score(X3_test, y3_test)

# validation of part 5
logVal = LogisticRegression(penalty = 'l2', C = 1.0)
logVal.fit(X3_train, y3_train)
logVal.score(X3_test, y3_test)
# validation successful! woot!


# part 6 - gridsearch with k nearest neighbors
neighbors = KNeighborsClassifier()
neighbors.fit(X3_train, y3_train)
neighbors.score(X3_test, y3_test)

# gridsearch nearest neighbors
neighborsRange = range(1,51)
param_grid2 = {'n_neighbors':(neighborsRange)}
model2 = KNeighborsClassifier()
neighborsGrid = GridSearchCV(model2, param_grid2)
neighborsGrid.fit(X3_train, y3_train)
neighborsGrid.best_estimator_
neighborsGrid.score(X3_test, y3_test)

# part 7... new target variable
data['alchemy_category'].value_counts()
len(data)
# recreation is around 25% of all of the data

# maskNumermical = ['alchemy_category_score', 'avglinksize', 'commonlinkratio_1',
#        'commonlinkratio_2', 'commonlinkratio_3', 'commonlinkratio_4',
#        'compression_ratio', 'embed_ratio', 'frameTagRatio',
#        'html_ratio', 'image_ratio', 'linkwordscore',
#        'non_markup_alphanum_characters', 'numberOfLinks', 'numwords_in_url',
#        'parametrizedLinkRatio', 'spelling_errors_ratio']
#
maskCat2 = ['lengthyLinkDomain',
            'news_front_page', 'framebased', 'hasDomainLink']

data3 = data
data3.dropna(inplace=True)


# fitting binary values to the recreation category & regularization/normalization of predictors
data3['new_pred'] = data3['alchemy_category'].apply(lambda x: 1 if x =='recreation' else 0)

y4 = data3['new_pred']

X4 = data3[maskNumermical]

dummies3 = pd.get_dummies(data[maskCat2])
X4 = pd.concat([X4, dummies3], axis = 1)
X4 = normalize(data2, norm = 'l2')

# gridsearch logistic with new predictors
new_model = LogisticRegression()
param_grid = {'penalty':('l1', 'l2'),'C':(.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0)}
logGrid2 = GridSearchCV(new_model, param_grid)

X4_train, X4_test, y4_train, y4_test = train_test_split(X4,y4)

logGrid2.fit(X4_train, y4_train)
logGrid2.score(X4_test, y4_test)

logGrid2.grid_scores_

# logistic regression with optimal parameters
newLogReg = LogisticRegression(penalty = 'l1', C = .1)
newLogReg.fit(X4_train, y4_train)
newLogReg.coef_
# coefficents all appear to be 0....

# I was sadly unable to figure out the process of gridsearching for precision over
# accuracy... looking forward to seeing the answer!
