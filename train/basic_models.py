import numpy as np
from scipy.ndimage import shift
import random
    
class Progressive():
    def __init__(self, offset, ytest):
        self.offset = offset
        self.ytest = ytest.copy(deep=True)

    def fit(self, xtrain = [], ytrain=[], sample_weight=[], eval_set=[]):
        pass
    
    def predict(self, xtest):
        res = self.ytest.shift(self.offset).values
        res = np.nan_to_num(res, nan=0)
        return res

    def predict_proba(self, xtest):
        res = np.zeros((len(xtest), 2))
        res[:,1] = self.ytest.shift(self.offset).values[:,0]
        res[:,1] = np.nan_to_num(res[:,1], nan=0)
        res[:,0] = 1 - res[:,1]
        return res
    
    def score(self, x, y):
        return 0
    
class Bernouilli():
    def __init__(self, proba, seed, n_class):
            self.proba = proba
            self.seed = seed
            self.n_class = n_class
            random.seed(seed)

    def fit(self, xtrain = [], ytrain=[], sample_weight=[], eval_set=[]):
        pass
    
    def predict(self, xtest):
        leni = len(xtest)
        res = np.zeros(leni)
        for i in range(leni):
            res[i] = random.random() < self.proba
        return res

    def predict_proba(self, xtest):
        leni = len(xtest)
        res = np.zeros(shape=(leni, self.n_class),)
        res[:,1] = self.proba
        res[:,0] = 1 - res[:, 1]
        return res

    def score(self, x, y):
        return 0