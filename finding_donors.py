
import numpy as np
import pandas as pd
from time import time
from IPython.display import display


import visuals as vs

get_ipython().run_line_magic('matplotlib', 'inline')

data = pd.read_csv("census.csv")

display(data.head(n=1))


n_records = len(data)

n_greater_50k = data.loc[data.income=='>50K', 'income'].count()

n_at_most_50k = data.loc[data.income=='<=50K', 'income'].count()

greater_percent = n_greater_50k*100/n_records

print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

vs.distribution(data)

skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

vs.distribution(features_log_transformed, transformed = True)


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

display(features_log_minmax_transform.head(n = 5))


features_final = pd.get_dummies(features_log_minmax_transform)

income = income_raw.eq('>50K').mul(1)

encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))



TP=np.sum(income)
FP=income.count()
TN=0
FN=0

accuracy = (TP+TN)/(FP)
recall = (TP)/(TP+FN)
precision = (TP)/(FP)

fscore = (1+pow(0.5,2))*(precision*recall)/((pow(0.5,2)*precision)+recall)

print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


from sklearn.metrics import fbeta_score, accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    
    results = {}
    
    start = time()
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()
    
    results['train_time'] = end-start
        
    start = time()
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()
    
    results['pred_time'] = end-start
            
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)
        
    results['acc_test'] = accuracy_score(y_test, predictions_test)
    
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)
        
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=0.5)
       
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    return results


from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

clf_A = GaussianNB()
clf_B = DecisionTreeClassifier(random_state=42)
clf_C = SVC(random_state=42)

samples_100 = len(y_train)
samples_10 = int(10*samples_100/100)
samples_1 = int(1*samples_100/100)

results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] =         train_predict(clf, samples, X_train, y_train, X_test, y_test)

vs.evaluate(results, accuracy, fscore)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

clf = SVC()

parameters = {'C':[0.1, 1, 10], 'kernel':['rbf'], 'gamma':[15, 20, 25]}

scorer = make_scorer(fbeta_score,beta=0.5)

grid_obj = GridSearchCV(clf, parameters, scoring=scorer)

grid_fit = grid_obj.fit(X_train, y_train)

best_clf = grid_fit.best_estimator_

predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))


from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV

adaboost = AdaBoostClassifier()
parms = {'n_estimators':[1,10,100,200], 'learning_rate':[0.001,0.01,0.1,0.5]}

model = RandomizedSearchCV(adaboost, param_distributions=parms).fit(X_train, y_train)

importances = model.best_estimator_.feature_importances_

vs.feature_plot(importances, X_train, y_train)


from sklearn.base import clone

X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

clf = (clone(SVC())).fit(X_train_reduced, y_train)

reduced_predictions = clf.predict(X_test_reduced)

print("Final Model trained on full data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
print("\nFinal Model trained on reduced data\n------")
print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))


#get_ipython().getoutput('jupyter nbconvert *.ipynb')
