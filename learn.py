from sklearn import preprocessing,pipeline
from sklearn.pipeline import make_pipeline
from sklearn.cross_validation import cross_val_score
import numpy as np
import csv

def read_csv(fname):
    data = []
    with open(fname,'r') as f:
        reader = csv.reader(f,delimiter=',')
        for row in reader:
            data.append(row)
    return data

def format_data(data):
    # Returns a list of dictionaries with keys 'attack-count', 'win-chance', and 'data'
    # Ignores the first row, which contains column headers
    return [{'attack-count': row[0], 'win-chance': row[1], 'data': row[2:]} for row in data[1:]]

def unweighted_classifier_data(formatted_data):
    X = np.array([row['data'] for row in formatted_data]).astype(np.float)
    def cat(y):
        if y == '0.0':
            return 0
        elif y == '1.0':
            return 1
        return 2
    y = np.array([cat(row['win-chance']) for row in formatted_data])
    return X,y

def sample_weights(formatted_data):
    return np.array([row['attack-count'] for row in formatted_data]).astype(np.float)

def scaler_pipeline(clf):
    # Adds a StandardScaler to the front of a pipeline, followed by clf. Returns the pipeline.
    return make_pipeline(preprocessing.StandardScaler(), clf)

def score_classifier(X, y, clf):
    pipe = scaler_pipeline(clf)
    scores = cross_val_score(pipe, X, y, cv=5, n_jobs=8)
    return scores
