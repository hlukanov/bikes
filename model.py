# -*- coding: utf-8 -*-
"""
@author: jjoachims
"""

# import necessary packages
import sys, pickle, numpy as np, pandas as pd;
from sklearn.model_selection import train_test_split; # import train_test_split function
from sklearn.calibration import CalibratedClassifierCV;
from sklearn.svm import LinearSVC;

# set necessary paths
one_hot_path = './models/one_hot_encoder.pkl';
csv_path = './data/train_test_data.csv';
model_path = './models/main_model.pkl';

# load one-hot encoder
enc = pickle.load(open(one_hot_path, 'rb'));

def loadData(csv_path, one_hot_path):
    # load preprocessed data
    data = pd.read_csv(csv_path, sep=",", encoding="utf-8");
    df = pd.DataFrame(data);
    
    # specify columns by index
    start_stations = df.iloc[:,3].to_numpy(dtype=str); # start stations, fourth column
    X = df.iloc[:,:3]; # day of week, hour, minute, load first three columns
    y = df.iloc[:,-1]; # end stations/classes, no need to train end_stations

    # convert start station IDs to one-hot vectors
    start_stations = enc.transform(start_stations.reshape(-1,1)).toarray();

    # concatenate features (datetime + start_stations one hot vectors) after encoding
    # to obtain training data
    X = np.concatenate((X, start_stations), axis=1);

    # split dataset into training set and test set, takes a random number once but then sticks
    # to the random number, necessary for reproducability
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1); # 70% training and 30% test

    print('Data loaded');

    return [X_train, X_test, y_train, y_test.to_numpy()];

def train(model_path, X_train, y_train):
    print('Training...');

    # initialize a multiclass SVM classifier
    clf = LinearSVC(random_state=0, tol=1e-5, verbose=1, dual=False);

    # train the SVM with Cross-validation, it's possible to use different classifiers, but with this
    # it's possible to give accuracy probabilities for the answers even if that means it needs three times
    # the training time 
    cclf = CalibratedClassifierCV(base_estimator=clf, cv=3);
    cclf.fit(X_train, y_train);#takes five hours

    # save the model
    with open(model_path, 'wb') as f:
        pickle.dump(cclf, f);

    return cclf;

# test if the right class is in the top K answers
def test_top(k, cclf, X_test, y_test):
    # list of probabilities for all classes
    res = cclf.predict_proba(X_test);

    topK = res.argsort()[:,-k:]; # get indices of the top K answers, which class is more probable
    topK_score = 0;

    # for each sample...
    for i in range(0,len(topK)):
        # ...for each answer
        for c in topK[i]:
            # ...check if the answer is the correct class
            if cclf.classes_[c] == y_test[i]:
                topK_score += 1;
                break;

    # return top K accuracy as percentage
    return round(topK_score * 100 / len(topK));

# load data
X_train, X_test, y_train, y_test = loadData(csv_path, one_hot_path);

# train or test, if you give the property train then it trains the model, else or default is test
if len(sys.argv) == 2 and sys.argv[1] == 'train':
    cclf = train(model_path, X_train, y_train);
else:
    # load pretrained model
    cclf = pickle.load(open(model_path, 'rb'));

print('Testing...');

# test accuracy of the whole test set
print('\nAccuracy test on the test set:');

accuracy_top_1 = int(round(cclf.score(X_test, y_test) * 100));
accuracy_top_3 = test_top(3, cclf, X_test, y_test);
accuracy_top_5 = test_top(5, cclf, X_test, y_test);

print('Top answer: {0}%'.format(accuracy_top_1));
print('Top 3 answers: {0}%'.format(accuracy_top_3));
print('Top 5 answers: {0}%'.format(accuracy_top_5));