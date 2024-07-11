import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

class SigmoidCalibrator:
    def __init__(self, prob_pred, prob_true, model):
        self.regressor = LogisticRegression()
        self.model = model
        self.regressor.fit(prob_pred, prob_true)

    def predict(self, x):
        probabilities = self.model.predict_proba(x)[:,1].reshape(-1,1)
        return self.regressor.predict(probabilities)
    
    def predict_proba(self, x):
        probabilities = self.model.predict_proba(x)[:,1].reshape(-1,1)
        return self.regressor.predict_proba(probabilities)

class IsotonicCalibrator:
    def __init__(self, prob_pred, prob_true):
        self.regressor = IsotonicRegression(out_of_bounds="clip")
        self.regressor.fit(prob_pred, prob_true)

    def predict(self, probabilities):
        return self.regressor.predict(probabilities)