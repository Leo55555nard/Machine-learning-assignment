# Import 
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error
import json

# Import training data
train = pd.DataFrame.from_records(json.load(open('train.json'))).fillna(" ")

# Drop abstract and editor
train = train.drop(["editor"], axis=1)
train = train.drop_duplicates(subset=['title'])

year = train["year"]

# Remove year from training data
train = train.drop(["year"], axis=1)

# Create dummies for entrytype
train = pd.get_dummies(train, columns=['ENTRYTYPE'])

# Adjust the columns 
X = train
y = year

# Convert y to numeric values
y = pd.to_numeric(y, errors="raise")

# Create a ColumnTransformer for vectorizing title, publisher, and author separately
featurizer = ColumnTransformer(
    transformers=[
        ("title", TfidfVectorizer(), "title"),
        ("abstract", TfidfVectorizer(), "abstract"),
        ("publisher", TfidfVectorizer(), "publisher"),
        ("author", TfidfVectorizer(analyzer=lambda x: x), "author")
    ],
    remainder='passthrough'
)

# Ridge Regression
# Create a pipeline with the featurizer and Ridge Regression
pipeline_R = make_pipeline(featurizer, Ridge(alpha=1.7))
pipeline_R.fit(X, y)

# Stochastic gradient descent
# Create a pipeline with the featurizer and SGDRegressor
pipeline_SGD = make_pipeline(featurizer, SGDRegressor(epsilon=0.01,max_iter=1000, alpha=0.0001, penalty='l1', random_state=42,learning_rate='adaptive',loss='squared_epsilon_insensitive'))
pipeline_SGD.fit(X, y)

# Import test file and drop editor
test = pd.DataFrame.from_records(json.load(open('test.json'))).fillna(' ')
test = test.drop(["editor"], axis=1)

# Create dummies for entrytype
test = pd.get_dummies(test, columns=['ENTRYTYPE'])
pred_R = pipeline_R.predict(test)
pred_SGD = pipeline_SGD.predict(test)
pred_avg = (0.5*pred_R + 0.5*pred_SGD)

# Create new column with predictions
test['year'] = pred_avg
test.to_json("predicted.json", orient='records', indent=2)