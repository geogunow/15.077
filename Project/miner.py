import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import *
from random import *
from copy import deepcopy as copy
from sklearn.linear_model import *
from sklearn.cross_validation import StratifiedKFold
from sklearn.naive_bayes import GaussianNB as NaiveBayes
from sklearn.svm import SVC, LinearSVC
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pydot
from termcolor import colored
import cart

pdf_num = 3

# allocate X,Y data
X = list()
Y = list()

# load data from file
with open('formed_data.csv','r') as fh:
    lines = fh.readlines()

    header = lines[0].split('\n')
    header = header[0].split(',')

    for line in lines[1:]:
        xvector = list()
        words = line.split(',')
        for word in words[:-1]:
            xvector.append(float(word))
        
        Y.append(float(words[-1]))
        X.append(xvector)

# convert to numpy format
X = np.array(X)
Y = np.array(Y)

# split into train, test, validate sets
N = len(Y)
indexes = np.arange(N)
np.random.seed(10)
np.random.shuffle(indexes)
cut1 = int(0.5*N)
cut2 = int(0.8*N)
ind_train = indexes[:cut1]
ind_val = indexes[cut1:cut2]
ind_test = indexes[cut2:]
X_train = X[ind_train,:]
Y_train = Y[ind_train]
X_val = X[ind_val,:]
Y_val = Y[ind_val]
X_test = X[ind_test,:]
Y_test = Y[ind_test]


# funciton to find hyper parameter for pruning
def find_opt_prune(model_fn, alphas, X_val, Y_val):

    opt_error = float('inf')
    opt_alpha = 0

    for alpha in alphas:

        # create new model
        model = model_fn(alpha)
        Y_p = model.predict(X_val)

        # calculate misclassification error
        err = sum(abs(Y_val - Y_p)) / len(Y_val)

        if err <= opt_error:
            opt_error = err
            opt_alpha = alpha

    return opt_alpha

# funciton to find hyper parameter
def find_opt_svm(model_fn, C_vals, gamma_vals, X_val, Y_val, 
        X_train, Y_train, kernel):

    opt_error = float('inf')
    opt_C = 0
    opt_gamma = 0

    if kernel == 'linear':
        gamma_vals = [1]

    for c in C_vals:
        for g in gamma_vals:

            # create new model
            if kernel == 'linear':
                model = model_fn(C=c, penalty='l1', dual=False)
            else:
                model = model_fn(C=c, gamma=g, kernel=kernel)
            model.fit(X_train, Y_train)
            Y_p = model.predict(X_val)

            # calculate misclassification error
            err = sum(abs(Y_val - Y_p)) / len(Y_val)

            if err <= opt_error:
                opt_error = err
                opt_C = c
                opt_gamma = g

    return (opt_C, opt_gamma)

# funciton to find hyper parameter
def find_opt_lr(model_fn, C_vals, X_val, Y_val, X_train, Y_train):

    opt_error = float('inf')
    opt_C = 0

    for c in C_vals:

        model = model_fn(C=c)
        model.fit(X_train, Y_train)
        Y_p = model.predict(X_val)

        # calculate misclassification error
        err = sum(abs(Y_val - Y_p)) / len(Y_val)

        if err <= opt_error:
            opt_error = err
            opt_C = c

    return opt_C


# function to output train, validate, and test errors
def output_errors(model, X_train, Y_train, X_val, Y_val, X_test, Y_test):

    # test model on validation data
    Y_p = model.predict(X_test)
    err = sum(abs(Y_test - Y_p)) / len(Y_test)
    print "Optimal Test Error = ", err

    # test model on validation data
    Y_p = model.predict(X_val)
    err = sum(abs(Y_val - Y_p)) / len(Y_val)
    print "Optimal Validation Error = ", err

    # test model on training data
    Y_p = model.predict(X_train)
    err = sum(abs(Y_train - Y_p)) / len(Y_train)
    print "Optimal Training Error = ", err

    return

'''
first create tree model
'''
'''
print "\nTree Model:"

# train tree model
clf = cart.DecisionTree(min_size=10)
clf.fit(X_train, Y_train)

# calculate optimal complexity penalty
alphas = np.logspace(-3,3,100)
opt_alpha = find_opt_prune(clf.pruned_tree, alphas, X_val, Y_val)
print "alpha = ", opt_alpha

# create final model
clf = clf.pruned_tree(opt_alpha)
output_errors(clf, X_train, Y_train, X_val, Y_val, X_test, Y_test)

# visualize graph
categories = ['Yes', 'No']
dot_data = clf.dot_export(var_names=header[:-1])
graph = pydot.graph_from_dot_data(dot_data)
graph.write_pdf("tree" + str(pdf_num) + ".pdf")
'''
'''
Scale data
'''
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

'''
use random forest classifier
'''
print "\nRandom Forest Model:"

# determine hyperparameters
best_l = 0
best_acc = 0
for l in np.linspace(5,25,10):
    l = int(l)
    rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=l, 
            oob_score=False)
    rf.fit(X_train, Y_train)
    acc = rf.score(X_val, Y_val)
    if acc > best_acc:
        best_acc = acc
        best_l = l

# fit model
print "Using", best_l, "min leaf samples"
rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=best_l,
        oob_score=False)
rf.fit(X_train, Y_train)

# print errors
output_errors(rf, X_train, Y_train, X_val, Y_val, X_test, Y_test)

'''
logistic regression w/ L-2 penalty
'''
print "\nLogisitc Regression Model:"

# find optimal generalization value
c_vals = np.logspace(-6,6,100)
C = find_opt_lr(LogisticRegression, c_vals, X_val, Y_val,
        X_train, Y_train)

# logistic regression with L-2 penalty
lr = LogisticRegression(C=C)
lr.fit(X_train, Y_train)
output_errors(lr, X_train, Y_train, X_val, Y_val, X_test, Y_test)

# print classification matrix
print "Logistic regression classification matrix"
Y_p = lr.predict(X_test)
print "pred = 0, acutal = 0:", sum( (Y_test == 0) & (Y_p == 0) )
print "pred = 1, acutal = 0:", sum( (Y_test == 0) & (Y_p == 1) )
print "pred = 0, acutal = 1:", sum( (Y_test == 1) & (Y_p == 0) )
print "pred = 1, acutal = 1:", sum( (Y_test == 1) & (Y_p == 1) )
print len(Y_test)
prob = lr.predict_proba(X_test)

prob_vals = list()
success = list()
for i in range(len(prob)):
    if prob[i][1] > prob[i][0]:
        if Y_test[i] == 1:
            print colored("Upset " + str(prob[i][1]), 'green')
            success.append(i)
        else:
            print colored("Upset " + str(prob[i][1]), 'red')
        prob_vals.append(prob[i][1])
    else:
        if Y_test[i] == 0:
            print colored("Favored " + str(prob[i][0]), 'green')
            success.append(i)
        else:
            print colored("Favored " + str(prob[i][0]), 'red')
        prob_vals.append(prob[i][0])

indexes = range(len(prob_vals))
indexes = np.array([s for (y,s) in sorted(zip(prob_vals, indexes))])

ordered = np.zeros(len(indexes))
val = 1
for index in indexes:
    ordered[index] = val
    val += 1

print ordered
print prob_vals
total = sum(ordered[success])
print "Point total = ", total
print "Total possible = ", sum(ordered)

ordered_temp = np.array(ordered)
temp = np.zeros(10**5)
for i in range(10**5):
    np.random.shuffle(ordered_temp)
    temp[i] = sum(ordered_temp[success])

print "Average arangement -> ", float(sum(temp))/len(temp), "points"

plt.hist(temp, bins=50, normed=True)
plt.xlabel('Point Total')
plt.ylabel('Frequency')
plt.show()

exit()

'''
Naive bayes gaussian model
'''
print "\nNaive Bayes:"

# Naive bayes model
nb = NaiveBayes()
nb.fit(X_train, Y_train)
output_errors(nb, X_train, Y_train, X_val, Y_val, X_test, Y_test)


'''
use simple rbf support vector classifier
'''
print "\nGaussian Kernel SVM Model:"
c_vals = np.logspace(2,5,25)
gamma_vals = np.logspace(-9,-5,25)
C, gamma = find_opt_svm(SVC, c_vals, gamma_vals, X_val, Y_val, 
        X_train, Y_train, 'rbf')
print "C = ", C
print "gamma = ", gamma
svm = SVC(C=C, gamma=gamma)
svm.fit(X_train, Y_train)

# print errors
output_errors(svm, X_train, Y_train, X_val, Y_val, X_test, Y_test)

'''
use simple linear support vector classifier
'''
print "\nLinear SVM Model:"
c_vals = np.logspace(-8,8,20)
gamma_vals = np.logspace(-10,-5,4)
C, gamma = find_opt_svm(LinearSVC, c_vals, gamma_vals, X_val, Y_val, 
        X_train, Y_train, 'linear')
print "C = ", C
svm = LinearSVC(C=C, penalty='l1', dual=False)
svm.fit(X_train, Y_train)

# print errors
output_errors(svm, X_train, Y_train, X_val, Y_val, X_test, Y_test)

# output coefficients from linear model
print "\nLinear SVM Lasso Coefficients"
coefs = svm.coef_
non_zero_coefs = list()
for i, coef in enumerate(coefs[0]):
    if coef != 0:
        non_zero_coefs.append(i)
        print header[i], coef

'''
Unscale data and trim
'''
X_train = scaler.inverse_transform(X_train)
X_val = scaler.inverse_transform(X_val)
X_test = scaler.inverse_transform(X_test)

X_train = X_train[:, non_zero_coefs]
X_val = X_val[:, non_zero_coefs]
X_test = X_test[:, non_zero_coefs]

new_header = list()
non_zero_coefs = set(non_zero_coefs)
for i, val in enumerate(header):
    if i in non_zero_coefs:
        new_header.append(val)
header = new_header

'''
train new tree using only non-zero lasso coefficients
'''
print "\nTree Model on Trimmed Data:"

# train tree model
clf = cart.DecisionTree(min_size=10)
clf.fit(X_train, Y_train)

# calculate optimal complexity penalty
alphas = np.logspace(-3,3,100)
opt_alpha = find_opt_prune(clf.pruned_tree, alphas, X_val, Y_val)
print "alpha = ", opt_alpha

# create final model
clf = clf.pruned_tree(opt_alpha)
output_errors(clf, X_train, Y_train, X_val, Y_val, X_test, Y_test)

# visualize graph
categories = ['Yes', 'No']
dot_data = clf.dot_export(var_names=header)
graph = pydot.graph_from_dot_data(dot_data)
graph.write_pdf("tree" + str(pdf_num) + "_v2.pdf")

'''
Rescale data and downselect
'''
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

'''
Train new gaussian kernel SVM using only nonzero lasso coefficents
'''
print "\nGaussian Kernel SVM Model on Trimmed Data:"
c_vals = np.logspace(2,8,25)
gamma_vals = np.logspace(-9,2,25)
C, gamma = find_opt_svm(SVC, c_vals, gamma_vals, X_val, Y_val, 
        X_train, Y_train, 'rbf')
print "C = ", C
print "gamma = ", gamma
svm = SVC(C=C, gamma=gamma)
svm.fit(X_train, Y_train)
output_errors(svm, X_train, Y_train, X_val, Y_val, X_test, Y_test)

'''
Train new random forest classifier
'''
print "\nRandom Forest Model:"

# determine hyperparameters
best_l = 0
best_acc = 0
for l in np.linspace(5,25,10):
    l = int(l)
    rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=l, 
            oob_score=False)
    rf.fit(X_train, Y_train)
    acc = rf.score(X_val, Y_val)
    if acc >= best_acc:
        best_acc = acc
        best_l = l

# fit model
print "Using", best_l, "min leaf samples"
rf = RandomForestClassifier(n_estimators=150, min_samples_leaf=best_l,
        oob_score=False)
rf.fit(X_train, Y_train)

# print errors
output_errors(rf, X_train, Y_train, X_val, Y_val, X_test, Y_test)


'''
Train new logistic regression model using trimmed data
'''
print "\nLogisitc Regression Model"

# find optimal generalization value
c_vals = np.logspace(-6,6,100)
C = find_opt_lr(LogisticRegression, c_vals, X_val, Y_val,
        X_train, Y_train)

# logistic regression with L-2 penalty
lr = LogisticRegression(C=C)
lr.fit(X_train, Y_train)
output_errors(lr, X_train, Y_train, X_val, Y_val, X_test, Y_test)

'''
Naive bayes gaussian model using trimmed data
'''
print "\nNaive Bayes:"

# Naive bayes model
nb = NaiveBayes()
nb.fit(X_train, Y_train)
output_errors(nb, X_train, Y_train, X_val, Y_val, X_test, Y_test)


