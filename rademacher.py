from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future
import numpy as np
import pandas as pd
import math
import sys
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import validation_curve
import matplotlib.patches as mpatches
from scipy.interpolate import make_interp_spline, BSpline


# Randomly choose impurity measure from "Gini" & "Entropy"
def impurity_measure():
    return np.random.choice(np.array(["gini", "entropy"]), size = 1, p = [0.5, 0.5])[0]


class Subsampling:
    def __init__(self, n, k, t):
        self.n = n  # Number of samples to draw
        self.k = k  # Number of trees to build
        self.t = t

    def estimate(self, h, X):
        m, p = X.shape
        r_t = []
        for t in range(self.t):
            # Generate Rademacher random varaible
            sigma = np.random.choice(np.array([-1, 1]), size = m, p = [0.5, 0.5])
            r_k = []
            for k in range(self.k):
                idx1 = np.random.choice(m, size = self.n, replace=False)
                idx2 = np.random.choice(p, size = min(h+10, p), replace=False)
                X_train = X[idx1, :][:, idx2]
                Y_train = sigma[idx1]
                # Fit decision tree
                tree = DecisionTreeClassifier(criterion = impurity_measure(), max_depth = h)
                tree.fit(X_train, Y_train)
                # Calculate error
                Y_hat = tree.predict(X[:, idx2])
                r_k.append(sigma.T.dot(Y_hat))

            r_t.append(max(r_k)/m)

        return sum(r_t)/self.t


class Adasampling:
    def __init__(self, k, t):
        self.k = k
        self.t = t

    def estimate(self, h, X):
        m, _ = X.shape
        r_t = []
        for t in range(self.t):
            # Generate Rademacher random varaible
            sigma = np.random.choice(np.array([-1, 1]), size = m, p = [0.5, 0.5])
            # Initialize weights
            W = np.ones(m)/m
            r_k = []
            impurity = impurity_measure()

            for k in range(self.k):
                tree = DecisionTreeClassifier(criterion = impurity, max_depth = h)
                tree.fit(X, sigma, sample_weight = W)
                # In Sample Prediction
                Y_hat = tree.predict(X)
                err = W.T.dot(sigma != Y_hat)
                if err == 0:
                    continue
                alpha = 0.5 * (np.log(1 - err) - np.log(err))
                # Update weights
                W = W * np.exp(-alpha * sigma * Y_hat)
                W = W/sum(W)

                # Calculate error
                r_k.append(sigma.T.dot(Y_hat))

            r_t.append(max(r_k)/m)

        return sum(r_t)/self.t



if __name__== "__main__":
    dataName = sys.argv[1]
    minDepth = int(sys.argv[2])
    maxDepth = int(sys.argv[3])

    data = pd.read_csv("/Users/guanghongfu/Desktop/projects/cos511/data/" + dataName, header = None)
    X = np.array(data.loc[:, 1:])
    Y = np.array(data.loc[:, 0])
    m, _ = X.shape

    # Split Dataset
    idx = np.array(range(0, m))
    np.random.shuffle(idx)
    n = math.floor(m * 0.8)
    X_train = X[idx[0:n], ]
    Y_train = Y[idx[0:n]]
    X_test = X[idx[n:], ]
    Y_test = Y[idx[n:]]

    # Calculate Rademacher Complexity
    rad_sub = []
    rad_ada = []
    for h in range(minDepth, maxDepth+1):
        print(h)
        sub = Subsampling(math.ceil(n*0.2), 100, 1000)
        rad_sub.append(sub.estimate(h, X_train))

        ada = Adasampling(50, 1000)
        rad_ada.append(ada.estimate(h, X_train))

    d = {'height': range(minDepth, maxDepth+1), 
         'subsampling': rad_sub,
         'adasampling': rad_ada}
    df = pd.DataFrame(data=d)
    # df.to_csv(dataName + "_rademacher.csv", index=False)

    # rademacher vs height plot
    plt.figure(0)
    line1 = plt.plot(range(minDepth, maxDepth+1), rad_sub, color = "cyan")
    line2 = plt.plot(range(minDepth, maxDepth+1), rad_ada, color = "yellow")
    cyan_patch = mpatches.Patch(color='cyan', label='subsampling')
    yellow_patch = mpatches.Patch(color='yellow', label='adasampling')
    plt.legend(handles=[cyan_patch, yellow_patch])

    plt.savefig(dataName + '_radmacher_vs_height.png')

    # Training Error & Testing Error
    train_scores, valid_scores = validation_curve(DecisionTreeClassifier(), 
                                                  np.array(X), 
                                                  np.array(Y), 
                                                  'max_depth',range(minDepth,maxDepth + 1), cv=5)
    train_err = 1 - np.mean(train_scores, axis=1)
    test_err = 1 - np.mean(valid_scores, axis=1)
    e = {'height': range(minDepth, maxDepth+1), 
         'training error': train_err,
         'testing error': test_err}
    df = pd.DataFrame(data=e)
    # df.to_csv(dataName + "_error.csv", index=False)

    plt.figure(1)
    xnew = np.linspace(minDepth, maxDepth+1, 200)

    spl_train = make_interp_spline(range(minDepth, maxDepth+1), train_err, k=3)
    smooth_train = spl_train(xnew)
    line3 = plt.plot(xnew, smooth_train, color = "magenta")

    spl_test = make_interp_spline(range(minDepth, maxDepth+1), test_err, k=3)
    smooth_test = spl_test(xnew)
    line4 = plt.plot(xnew, smooth_test, color = "yellow")

    spl_ada = make_interp_spline(range(minDepth, maxDepth+1), train_err + np.array(rad_sub), k=3)
    smooth_ada = spl_ada(xnew)
    line5 = plt.plot(xnew, smooth_ada, color = "cyan")

    magenta_patch = mpatches.Patch(color='magenta', label='training_err')
    yellow_patch = mpatches.Patch(color='yellow', label='testing_err')
    cyan_patch = mpatches.Patch(color='cyan', label='est_rada_gen_err')
    plt.legend(handles=[magenta_patch, yellow_patch, cyan_patch])

    plt.savefig(dataName + '_error_vs_height.png')




    
    











