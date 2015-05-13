import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import *
from random import *
from copy import deepcopy as copy
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import *
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pydot
import cart

font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size' : 15}
plt.rc('font', **font)

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
D = len(X[0])
N = len(X)

# create dictionary for header
header_vals = dict()
for i, name in enumerate(header[:-1]):
    header_vals[name] = i


'''
Principal Component Analysis
'''
# get two main principal components
decomp = PCA(n_components=2)
X_decomp = decomp.fit_transform(X)
XA = X_decomp[:,0]
XB = X_decomp[:,1]
XA1 = list()
XA2 = list()
XB1 = list()
XB2 = list()
for i in range(N):
    if Y[i] == 0:
        XA1.append(XA[i])
        XB1.append(XB[i])
    else:
        XA2.append(XA[i])
        XB2.append(XB[i])
plt.plot(XA1, XB1, 'k.')
plt.plot(XA2, XB2, 'r.')
plt.legend(['Favor Win', 'Upset'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
exit()

'''
Plotting single statistics
'''
# choose home team rush offense
quant = 'Home Team allRushYdsOff'
ind = header_vals[quant]
X1 = list()
X2 = list()
for i in range(N):
    if Y[i] == 0:
        X1.append(X[i,ind])
    else:
        X2.append(X[i,ind])
X1 = np.array(X1)
X2 = np.array(X2)



'''
plt.hist(X[:,ind], bins=N/40, normed=normalize, color='k')
plt.xlabel(quant)
plt.ylabel('Frequency')
#plt.show()
'''
normalize = True
plt.hist(X1, bins=N/40, normed=normalize, color='b')
plt.hist(X2, bins=N/40, normed=normalize, color='r')
plt.xlabel('Home Team Rushing Yards per Game')
plt.ylabel('Frequency')
plt.legend(['Visitor Win', 'Home Team Win'])
plt.show()

X1 = np.log(X1)
X2 = np.log(X2)

s_X1 = sorted(X1)
k = np.array(range( len(s_X1) )) + 1.0
uniform = k / (len(s_X1) + 1)
ordered_X1 = norm.ppf(uniform)
plt.plot(ordered_X1, s_X1, 'b.')
s_X2 = sorted(X2)
k = np.array(range( len(s_X2) )) + 1.0
uniform = k / (len(s_X2) + 1)
ordered_X2 = norm.ppf(uniform)
plt.plot(ordered_X2, s_X2, 'r.')
plt.legend(['Visitor Win', 'Home Team Win'])
plt.xlabel('Normal Quantiles')
plt.ylabel('Home Team Rushing Yards per Game')
plt.show()

# repeat for visitor rush defense
quant = 'Visitor allRushYdsDef'
ind = header_vals[quant]
X1 = list()
X2 = list()
for i in range(N):
    if Y[i] == 0:
        X1.append(X[i,ind])
    else:
        X2.append(X[i,ind])
X1 = np.array(X1)
X2 = np.array(X2)

normalize = True
plt.hist(X1, bins=N/40, normed=normalize, color='b')
plt.hist(X2, bins=N/40, normed=normalize, color='r')
plt.xlabel('Visitor Rushing Yards Against per Game')
plt.ylabel('Frequency')
plt.legend(['Visitor Win', 'Home Team Win'])
plt.show()

s_X1 = sorted(X1)
k = np.array(range( len(s_X1) )) + 1.0
uniform = k / (len(s_X1) + 1)
ordered_X1 = norm.ppf(uniform)
plt.plot(ordered_X1, s_X1, 'b.')
s_X2 = sorted(X2)
k = np.array(range( len(s_X2) )) + 1.0
uniform = k / (len(s_X2) + 1)
ordered_X2 = norm.ppf(uniform)
plt.plot(ordered_X2, s_X2, 'r.')
plt.legend(['Visitor Win', 'Home Team Win'])
plt.xlabel('Normal Quantiles')
plt.ylabel('Visitor Rushing Yards Against per Game')
plt.show()

'''
Interaction between two chosen features
'''
# choose home rush offense, visitor rush defense
quant1 = 'Home Team allRushYdsOff'
quant2 = 'Visitor allRushYdsDef'
ind1 = header_vals[quant1]
ind2 = header_vals[quant2]
for i in range(N):
    if Y[i] == 0:
        plt.plot(X[i][ind1], X[i][ind2], 'k.')
    else:
        plt.plot(X[i][ind1], X[i][ind2], 'r.')

plt.legend(['Visitor Win', 'Home Team Win'])
plt.xlabel("Home Team Rushing Offense Yards per Game")
plt.ylabel("Visitor Rushing Yards Against per Game")
plt.show()

# plot transformed data
quant1 = 'Home Team allRushYdsOff'
quant2 = 'Visitor allRushYdsDef'
ind1 = header_vals[quant1]
ind2 = header_vals[quant2]
for i in range(N):
    if Y[i] == 0:
        plt.plot(np.log(X[i][ind1]), X[i][ind2], 'k.')
    else:
        plt.plot(np.log(X[i][ind1]), X[i][ind2], 'r.')

plt.legend(['Visitor Win', 'Home Team Win'])
plt.xlabel("Home Team Rushing Offense Yards per Game")
plt.ylabel("Visitor Rushing Yards Against per Game")
plt.show()

'''
quant1 = 'Home Team allRushYdsDef'
quant2 = 'Visitor allRushYdsOff'
ind1 = header_vals[quant1]
ind2 = header_vals[quant2]
for i in range(N):
    if Y[i] == 0:
        plt.plot(X[i][ind1], X[i][ind2], 'k.')
    else:
        plt.plot(X[i][ind1], X[i][ind2], 'r.')

plt.xlabel(quant1)
plt.ylabel(quant2)
plt.show()


quant1a = 'Home Team allRushYdsOff'
quant1b = 'Home Team allRushYdsDef'
quant2a = 'Visitor allRushYdsOff'
quant2b = 'Visitor allRushYdsDef'
ind1a = header_vals[quant1a]
ind1b = header_vals[quant1b]
ind2a = header_vals[quant2a]
ind2b = header_vals[quant2b]
for i in range(N):
    if Y[i] == 0:
        plt.plot(X[i][ind1a] + X[i][ind2b], X[i][ind2a] + X[i][ind1b], 'b.')
    else:
        plt.plot(X[i][ind1a] + X[i][ind2b], X[i][ind2a] + X[i][ind1b], 'r.')

plt.xlabel('Net ' + quant1a)
plt.ylabel('Net ' + quant2a)
plt.show()

# search for stuff
for j in range(D/2):
    for i in range(N):
        if Y[i] == 0:
            plt.plot(X[i][2*j], X[i][2*j+1], 'k.')
        else:
            plt.plot(X[i][2*j], X[i][2*j+1], 'r.')

    plt.xlabel(header[2*j])
    plt.ylabel(header[2*j+1])
    plt.show()
'''
