import pandas as pd
import numpy as np
import xgboost
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import griddata
from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from pandas.plotting import table
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import pickle
import random
import math
from sklearn.model_selection import cross_validate
import CRPS.CRPS as pscore
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, \
roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, brier_score_loss, log_loss, mean_squared_error, mean_pinball_loss
from pandas.plotting import table
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
from sklearn.calibration import calibration_curve
from skopt import BayesSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import numpy as np
from calibration import SigmoidCalibrator

def Myplot(dataset, dir_output, departement):
    if len(dataset) == 0:
        return
    clusterId = dataset.cluster.unique()
    fig, ax = plt.subplots(int(len(clusterId)), figsize = (20,20))
    
    for i, cID in enumerate(clusterId):
        dataset252_ = dataset[(dataset['cluster'] == cID)]
        dataset252_Fire = dataset252_[dataset252_['is1NATURELSFire_1'] > 0]
        ax[i].set_ylim([0, 1])
        ax[i].scatter(dataset252_Fire.index.values, dataset252_Fire.calibratedProbaFor1NATURELSFiretest.values, color='black', label='IsFire')
        ax[i].plot(dataset252_.index.values, dataset252_.calibratedProbaFor1NATURELSFiretest.values, label='Calibrated Proba')
        ax[i].plot(dataset252_.index.values, dataset252_.proba.values, label='Regression')
        ax[i].set_ylabel('Probability')
        ax[i].set_xlabel('Day of year')
        
    plt.tight_layout()
    
    plt.legend()
    outname  = departement+'.png'
    plt.savefig(dir_output / outname)
    plt.close('all')

def my_log_loss(ytrue, ypred, sample_weights, pad):
    res = []
    leni = len(ypred)
    i = 0
    while i < leni:
        res.append(log_loss(ypred[i:i + pad], ytrue[i:i + pad], sample_weight=sample_weights[i:i+pad], labels=[0,1]))
        i += pad
    return np.array(res)


def root_mean_squared_error(ytrue, ypred, sample_weight=None):
    return math.sqrt(mean_squared_error(ytrue, ypred, sample_weight=sample_weight))


def quantile_loss(y_true, y_pred, q):
    return np.max([q*(y_true - y_pred), (1-q)*(y_pred-y_true)], axis=0)

def expected_calibration_error(true_labels_series, samples, bins=5):
    true_labels = true_labels_series.values
    samples = samples.reshape(-1)
    bin_count, bin_edges = np.histogram(samples, bins = 'auto')
    n_bins = len(bin_count)
    # uniform binning approach with M number of bins
    bin_boundaries = bin_edges
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = samples

    # get a boolean list of correct/false predictions
    accuracies = true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prob_in_bin = in_bin.mean()

        if prob_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prob_in_bin
    return n_bins, ece[0] * 100

def check_and_create_path(path: Path):
    """
    Creer un dossier s'il n'existe pas
    """
    path_way = path.parent if path.is_file() else path

    path_way.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

def callback(self):
    print('Iteration number:')

class Launcher():

    def __init__(self, dir_output, model, train, val, name,
                 variables, config, saving):
        self.train = train
        self.val = val
        self.variables = variables
        self.config = config
        self.trainScore = 0
        self.testScore = 0
        self.name = name
        self.saving = saving
        self.dir_output = dir_output
        self.model = model
        self.iterTest = 0
        
        if self.config['prefit']:
            outputname = self.name + '.joblib'
            self.model = joblib.load(dir_output/outputname)

        check_and_create_path((self.dir_output))

    def __train__(self, tuning, searchParams, fitparams):
        if tuning == 'bayes':
            self.trainBaysearch(searchParams, fitparams)
        elif tuning == 'grid':
            self.trainGridSearch(searchParams, fitparams)
        else:
            self.trainVanilla(fitparams)

    def train_cross_validation(self, fitparams):
        xtrain = self.train[self.variables]
        ytrain = self.train[self.config['target']]

        scores = cross_validate(self.model, xtrain, ytrain, cv=5, return_estimator=True, fit_params=fitparams)
        bestIndice = np.argmax(scores['test_score'])
        self.model = scores['estimator'][bestIndice]

    def trainVanilla(self, fitparams):
        print('Training vanilla {}'.format(self.name))
        #if self.config['prefit']:
        #    return
        xtrain = self.train[self.variables]
        ytrain = self.train[self.config['target']]
        self.model = self.model.fit(xtrain, ytrain, **fitparams)
        outputname = self.name + '.joblib'
        joblib.dump(self.model, self.dir_output / outputname)

    def trainBaysearch(self, bayesSearchParams, fitparams):
        print('Training bayes {}'.format(self.name))
        self.iter = 0

        xtrain = self.train[self.variables]
        ytrain = self.train[self.config['target']]
    
        bayes_cv_tuner = BayesSearchCV(**bayesSearchParams, estimator=self.model)

        bayes_cv_tuner.fit(xtrain, ytrain, **fitparams)

        self.model = bayes_cv_tuner.best_estimator_
        self.trainScore = bayes_cv_tuner.best_score_

        outputname = self.name + '.joblib'
        joblib.dump(self.model, self.dir_output / outputname)

    def trainGridSearch(self, gridSearchParams, fitparams):
        print('Training grid {}'.format(self.name))
        self.iter = 0 

        xtrain = self.train[self.variables]
        ytrain = self.train[self.config['target']]
    
        grid_cv_tuner = GridSearchCV(**gridSearchParams, estimator=self.model)

        grid_cv_tuner.fit(xtrain, ytrain, **fitparams)

        self.model = grid_cv_tuner.best_estimator_
        self.trainScore = grid_cv_tuner.best_score_

        outputname = self.name + '.joblib'
        joblib.dump(self.model, self.dir_output / outputname)

    def calibrate(self, mode):
        if self.config['prefit']:
            self.name += '_sigmoid'
            outputname = self.name + '.joblib'
            self.model = joblib.load(self.dir_output/outputname)
        else:
            xval = self.val[self.variables]
            yval = self.val[self.config['target']]
            #calibrator = CalibratedClassifierCV(self.model, cv='prefit', method=mode)
            #calibrator.fit(xval, yval)
            calibrator = SigmoidCalibrator(self.model.predict_proba(xval)[:,1].reshape(-1,1), yval, self.model)
            self.model = calibrator
            self.name += '_sigmoid'
            self.iterTest = 0
            outputname = self.name + '.joblib'
            joblib.dump(self.model, self.dir_output / outputname)

    def __test__(self, test, testname, weights='noWeights', name=''):
        print('----Testing {}, len test {}:'.format(name, len(test)))
        if weights == 'noWeights':
            test['noWeights'] = 1

        self.iterTest += 1
        self.res_metrics = {}
        self.test = test
        ytest = self.test[self.config['target']]
        xtest = self.test[self.variables]

        if self.config['type'] == 'binary':
            ypred = self.model.predict(xtest)
            self.proba = self.model.predict_proba(xtest)[:,1]
            self.pred = ypred
            self.test['proba'] = self.proba
            self.test['pred'] = ypred

        elif self.config['type'] == 'quantile':
            ypred = self.model.predict(xtest)
            for i, al in enumerate(self.config['alpha']):
                
                lam = lambda x : x if x < 1 else 1

                ypred[:,i] = np.array(list(map(lam, ypred[:,i])))

                lam = lambda x : x if x > 0 else 0

                ypred[:,i] = np.array(list(map(lam, ypred[:,i])))

                self.test['proba_'+str(al)] = ypred[:,i]
                self.test['pred_'+str(al)] = ypred[:,i]
                
            self.pred = ypred
            self.proba = ypred

        elif self.config['type'] == 'fit':
            ypred = self.model.predict(xtest)
            lam = lambda x : x if x < 1 else 1

            ypred = np.array(list(map(lam, ypred)))

            lam = lambda x : x if x > 0 else 0

            ypred = np.array(list(map(lam, ypred)))

            self.proba = ypred
            self.pred = ypred
            self.test['proba'] = self.proba
            self.test['pred'] = ypred

        elif self.config['type'] == 'multiregression' or self.config['type'] == 'multibinary':
            ypred = np.moveaxis(self.model.predict(xtest), 0, 1)
            
            lam = lambda x : x if x < 1 else 1

            ypred = np.array([np.array(list(map(lam, yp))) for yp in ypred])

            lam = lambda x : x if x > 0 else 0

            ypred = np.array([np.array(list(map(lam, yp))) for yp in ypred])

            self.proba = ypred
            self.pred = ypred
            for i in range(11):
                self.test['proba_'+str(i)] = self.proba[i]
                self.test['pred_'+str(i)] = ypred[i]

        self.score(ytest, self.pred, self.proba, weights, testname)

    def score(self, ytrue, ypred, proba, sample_weight, testname):
        metrics = self.config['metrics']
        for key in metrics:
            self.res_metrics[key] = []
            func = metrics[key]['func']
            params = metrics[key]['params']
            use_proba = metrics[key]['use_proba']
            use_weights = metrics[key]['use_weights']
            use_nonfire = metrics[key]['use_nonfire']

            if use_nonfire:
                test = self.test.reset_index(drop=True)
            else:
                test = self.test[self.test['is1NATURELSFire_1'] == 1].reset_index(drop=True)
            
            if use_weights:
                params['sample_weight'] = test[sample_weight]
            if self.config['type'] == 'binary' or self.config['type'] == 'fit':
                res = func(test[metrics[key]['target']], test.proba.values, **params)
                self.res_metrics[key].append(res)
            elif self.config['type'] == 'quantile':
                pos_al = metrics[key]['pos_alpha']
                res = func(ytrue, ypred[:,pos_al], **params)
                self.res_metrics[key].append(res)
            elif self.config['type'] == 'multiregression' or self.config['type'] == 'multibinary':
                res = []
                for i in range(1, 11):
                        res = func(test[metrics[key]['target']+'_'+str(i)], proba[i], **params)
                        self.res_metrics[key+'_'+'target'+'_'+str(i)] = [res]

            print('Calculating {}, Result with {}\n'.format(key, res))

        if self.config['type'] == 'binary':
            self.confusion_matrix = confusion_matrix(ytrue, ypred, normalize='true')

        self.res_metrics = pd.DataFrame.from_dict(self.res_metrics)

        if self.saving:
            self.save(testname)

    def save(self, testname):

        ### Save metrics
        outputname = self.name+'_'+testname+"_metrix.png"
        _ = plt.figure(figsize=(20,15))
        plot = plt.subplot(111, frame_on=False)
        table(plot, self.res_metrics,loc='upper right')
        plt.savefig(self.dir_output / outputname)
        outputname = self.name+'_'+testname+"_metrix.csv"
        self.res_metrics.to_csv(self.dir_output / outputname, index=False)
        
        #### Save confusion matrix
        if self.config['type'] == 'type':
            outputname = self.name+'_'+testname+"_matrix.png"
            if self.config['return_proba']:
                ConfusionMatrixDisplay(self.confusion_matrix).plot()
                plt.savefig(self.dir_output / outputname)
                plt.close('all')
        
        if self.config['type'] != 'quantile' and self.config['type'] != 'multiregression' and self.config['type'] != 'multibinary':
            ### Save calibration
            fig = plt.figure(figsize=(10,10))
            gs = GridSpec(4, 2)

            ax_calibration_curve = fig.add_subplot(gs[:2, :2])
            ax_calibration_curve.plot([0, 1], [0, 1], linestyle = '--', label = 'Perfect calibration')
            hist, bin_edge = np.histogram(self.proba, bins='auto')
            n_bins = len(bin_edge)
            y_means, proba_means = calibration_curve(self.test['is1NATURELSFire_1'], self.proba, n_bins=n_bins, strategy='quantile')
            ax_calibration_curve.plot(proba_means, y_means, label=self.name)

            ax_calibration_curve.set(xlabel="Mean predicted probability of positive class", ylabel="Fraction of positive class")
            ax_calibration_curve.grid()
            ax_calibration_curve.legend()
            outputname = self.name+'_'+testname+'_calibration_curve.png'
            plt.savefig(self.dir_output / outputname)

        if self.config['type'] == 'multiregression' or self.config['type'] == 'multibinary':
            fig = plt.figure(figsize=(10,10))
            gs = GridSpec(4, 2)

            ax_calibration_curve = fig.add_subplot(gs[:2, :2])
            ax_calibration_curve.plot([0, 1], [0, 1], linestyle = '--', label = 'Perfect calibration')

            for i in range(11):
                hist, bin_edge = np.histogram(self.proba[i], bins='auto')
                n_bins = len(bin_edge)
                y_means, proba_means = calibration_curve(self.test['is1NATURELSFire_'+str(i)], self.proba[i], n_bins=n_bins, strategy='quantile')
                ax_calibration_curve.plot(proba_means, y_means, label=self.name)

            ax_calibration_curve.set(xlabel="Mean predicted probability of positive class", ylabel="Fraction of positive class")
            ax_calibration_curve.grid()
            ax_calibration_curve.legend()
            outputname = self.name+'_'+testname+'_calibration_curve.png'
            plt.savefig(self.dir_output / outputname)

        try:
            fig, ax = plt.subplots(figsize=(10,5))
            xgboost.plot_importance(self.model.get_booster(), ax=ax, max_num_features = 100)
            outputname = self.name+'_'+testname+'features_importances.png'
            plt.savefig(self.dir_output / outputname)
        except:
            pass

        Ain = np.arange(0,10)
        rhone = np.arange(680,690)
        Doubs = np.arange(240, 250)
        Yvelines = np.arange(770,780)

        if testname == 'test2023':
            Myplot(self.test[self.test['cluster'].isin(Ain)], self.dir_output, 'ain'+self.name )
            Myplot(self.test[self.test['cluster'].isin(Doubs)], self.dir_output, 'Doubs'+self.name )
            Myplot(self.test[self.test['cluster'].isin(Yvelines)], self.dir_output, 'Yvelines'+self.name )
        if testname == 'test69':
            Myplot(self.test[self.test['cluster'].isin(rhone)], self.dir_output, 'rhone'+self.name )

        """#### Save prediction
        if self.config['type'] != 'quantile':
            
            fig = plt.figure(figsize = (15,5))

            plt.plot(self.test.index.values, self.test.isFire.values, label='IsFire')
            plt.plot(self.test.index.values, self.proba, label='prediction')
            #plt.plot(self.test.index.values, self.proba, label='predictProba')

            outputname = self.name+'_'+testname+'_prediction.png'
            plt.savefig(self.dir_output / outputname)
        else:
            pass"""