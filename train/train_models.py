import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from scipy.interpolate import griddata
from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, \
roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, brier_score_loss, log_loss
from pandas.plotting import table
from scipy import stats
import pickle
import random
import math
#import CRPS.CRPS as pscore

### Models
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from ngboost import NGBClassifier, NGBRegressor


### Metrics
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, \
roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, brier_score_loss, log_loss, mean_squared_error, mean_pinball_loss

from pandas.plotting import table
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats
from sklearn.calibration import calibration_curve
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
from tools import Launcher, root_mean_squared_error, expected_calibration_error
from variables import variables

random.seed(42)
    
def pipeline(launcher, tuning, doCalibration, tests, searchParams, fitparams):

    #try:
        launcher.__train__(tuning, searchParams=searchParams, fitparams=fitparams)

        for test in tests:
            if len(test['df']) == 0:
                continue
            launcher.__test__(test['df'], test['name'], test['weights'], test['name'])

        if doCalibration:
            launcher.calibrate('sigmoid')
            for test in tests:
                if len(test['df']) == 0:
                    continue
                launcher.__test__(test['df'], test['name'], test['weights'], test['name'])
    #except Exception as e:
    #    print(e)

def train_val_test_split_sequence(dataset, testsize, valsize, offset, seq_len):
    clustersID = dataset.cluster.unique()
    
    train = []
    test = []
    val = []

    randomChoice = [1,2,3]
    randomWeight = [1 - testsize - valsize, valsize, testsize]

    for j, cluster in enumerate(clustersID):
        i = 0
        print(cluster)
        data = dataset[dataset['cluster'] == cluster].reset_index(drop=True)
        indexMax = data.index[-1]
        while i < indexMax:
            choice = random.choices(randomChoice, randomWeight)[0]

            if choice == 1:
                train.append(data[i:i+seq_len])
            elif choice == 2:
                val.append(data[i:i+seq_len])
            else:
                test.append(data[i:i+seq_len])
            
            i += seq_len + offset

    if len(train) > 0:
        train = pd.concat(train).sort_values('creneau').reset_index(drop=True)
        train.sort_values('creneau', inplace=True)
        train = train.reset_index(drop=True)

    if len(val) > 0:
        val = pd.concat(val).sort_values('creneau').reset_index(drop=True)
        val.sort_values('creneau', inplace=True)
        val = val.reset_index(drop=True)
        
    if len(test) > 0:
        test = pd.concat(test).sort_values('creneau').reset_index(drop=True)
        test.sort_values('creneau', inplace=True)
        test = test.reset_index(drop=True)

    return train, val, test

def make_tuning_param(tuning, model):

    if tuning == 'bayes':

        searchParams = {'n_jobs' : -1,
            'n_iter' : 100,
            'n_points': 5,
            'verbose' : 0,
            'scoring' : None,
            'refit' : True,
            'random_state' : 42,
            'cv':5}

    if tuning == 'grid':
        searchParams = {'n_jobs' : -1,
                'verbose' : 0,
                'scoring' : None,
                'refit' : True,
                'cv':10}
    else:
        searchParams = {}
        
    if model == 'xgboost':
       """ search_spaces = {
                    "learning_rate": (0.01, 1.0, "log-uniform"),
                    "min_child_weight": (1, 10),
                    "max_depth": (1, 15),
                    "max_delta_step": (5, 20),
                    "subsample": (0.20, 0.80, "uniform"),
                    "colsample_bytree": (0.20, 0.80, "log-uniform"),
                    "colsample_bylevel": (0.20, 0.80, "log-uniform"),
                    "reg_lambda": (1.5, 100, "log-uniform"),
                    "reg_alpha": (1e-9, 1.0, "log-uniform"),
                    "gamma": (1e-9, 0.5, "log-uniform"),
                    "n_estimators": (5000, 20000),
                }"""
       
       search_spaces = {
                    "learning_rate": [0.01, 0.05, 0.15, 0.50, 0.75],
                    "min_child_weight": [1,5,10],
                    #"max_depth": np.linspace(5, 20, 3).astype(int),
                    'max_depth' : [6, 10, 15],
                    "max_delta_step": [5,10,15,20],
                    "subsample": np.linspace(0.20,0.8, 4),
                    "colsample_bytree": np.linspace(0.20,0.8, 4),
                    #"colsample_bylevel": np.linspace(0.20,0.8, 4),
                    "reg_lambda": np.linspace(1, 100, 20),
                    "reg_alpha": np.linspace(1e-9, 1, 10),
                    #"gamma": np.linspace(1e-9, 1, 10),
                    #"n_estimators": np.linspace(5000, 15000, 10).astype(int),
                    'n_estimators' : [500,1000,2000,5000,10000],
                    'early_stopping_rounds' : 15
                }

    if model == 'lightgbm':
        """search_spaces = {
            "learning_rate": (0.01, 1.0, "log-uniform"),
            "min_child_weight": (1, 10),
            'min_data_in_leaf': (5, 100),
            "max_depth": (5, 15),
            "bagging_fraction": (0.20, 0.80, "uniform"),
            "colsample_bytree": (0.20, 0.80, "log-uniform"),
            "feature_fraction_bynode": (0.20, 0.80, "log-uniform"),
            "reg_lambda": (1.5, 100, "log-uniform"),
            "reg_alpha": (1e-9, 1.0, "log-uniform"),
            "num_iterations": (5000, 50000),
            'num_leaves' : (32, 2**15)
        }"""
        search_spaces = {
                    "learning_rate": np.linspace(0.01,0.8, 5),
                    "min_child_weight": np.linspace(1,10),
                    "min_data_in_leaf": np.linspace(1,10).astype(int),
                    #"max_depth": np.linspace(5, 20, 3).astype(int),
                    'max_depth' : [6],
                    "bagging_fraction": np.linspace(0.20,0.8, 4),
                    "colsample_bytree": np.linspace(0.20,0.8, 4),
                    "feature_fraction_bynode": np.linspace(0.20,0.8, 4),
                    "reg_lambda": np.linspace(1, 100, 20),
                    "reg_alpha": np.linspace(1e-9, 1, 10),
                    "gamma": np.linspace(1e-9, 1, 10),
                    "num_iterations": np.linspace(5000, 25000, 10).astype(int),
                    'num_leaves' : [2**20]
                }

    if model == 'ngboost':
        """search_spaces = {
            "learning_rate": (0.01, 1.0, "log-uniform"),
            "minibatch_frac": (0.20, 0.8, "uniform"),
            "col_sample": (0.20, 0.8, "log-uniform"),
            "n_estimators": (1000, 5000),
        }"""

        search_spaces = {
            "learning_rate": np.linspace(0.01,0.8, 5),
            "minibatch_frac": np.linspace(0.20,0.8, 4),
            "col_sample": np.linspace(0.20,0.8, 4),
            "n_estimators": np.linspace(5000, 25000, 10).astype(int),
            #'n_estimators' : [10000]
        }

    if model == 'rf':
        """search_spaces = {
        'n_estimators': (500, 10000),
        'max_depth': (5, 30),
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999),
        }"""

        search_spaces = {
        "n_estimators": np.linspace(5000, 25000, 10).astype(int),
        "max_depth": np.linspace(5, 20, 3).astype(int),
        'min_samples_split': np.linspace(5, 20, 5).astype(int),
        'max_features': np.linspace(0.20,0.8, 4),
        }
    
    searchParams['param_grid'] = search_spaces
    return searchParams

##########################################################################

dataset_version = 'GMClustering'
model_version = 'calibratedWeights'
root = Path('/home/caron/Bureau/Model/LargeScale')
dir_dataset = Path(root / dataset_version)

# Use gpu
GPU = True

### train val test

train = pd.read_csv(dir_dataset / 'train.csv')
val = pd.read_csv(dir_dataset / 'val.csv')
generalizeTest = pd.read_csv(dir_dataset / 'test_2023.csv')

dataset = pd.read_csv(dir_dataset / 'dataset_2022_calibrated.csv')

### Departement prediction
departmentPrediction = False

ids = np.arange(0, 10)

if departmentPrediction:
    variablesToAdd = variables.copy()
    newVariables = ['month',
    'dayofyear', 'dayofweek',
    'bankHolidays', 'bankHolidaysEve', 'bankHolidaysEveEve',
    'holidays', 'holidaysEve', 'holidaysEveEve', 'holidaysLastDay', 'holidaysLastLastDay']

    for var in newVariables:
        variablesToAdd.remove(var)

    for id in ids:
        datasetCluster = dataset[dataset['cluster'] == id]
        columns = [col + '_' + str(id) for col in variablesToAdd]
        
        newVariables += columns

    variables = newVariables

print(len(train), len(val), len(generalizeTest))
train[variables] = train[variables].astype(float)
train.drop_duplicates(subset=variables, inplace=True)
train.dropna(subset=variables, inplace=True)

#test.drop_duplicates(subset=variables, inplace=True)
#test.dropna(subset=variables, inplace=True)

val[variables] = val[variables].astype(float)
val.drop_duplicates(subset=variables, inplace=True)
val.dropna(subset=variables, inplace=True)

generalizeTest[variables] = generalizeTest[variables].astype(float)
generalizeTest.drop_duplicates(subset=variables, inplace=True)
generalizeTest.dropna(subset=variables, inplace=True)

print(len(train), len(val), len(generalizeTest))

"""print('Class ', 'dataset weight', dataset['weights_reg'].unique(), '\n',
     'train weight', train['weights_reg'].unique(), '\n',
     'val weight', val['weights_reg'].unique(), '\n',
    'generalizeTest weight', generalizeTest['weights_reg'].unique())

print('Binary ', dataset['weights_binary'].unique(), '\n',
      'train weight', train['weights_binary'].unique(), '\n',
        'val weight', val['weights_binary'].unique(), '\n',
      'generalizeTest weight', generalizeTest['weights_binary'].unique())"""

### Tuning
tuning = 'grid'
doCalibration = False

### Models to train
train_basic = False
train_xgboost = True
train_lightgbm = True
train_ngboost = True
train_rf = True

#### Weight
train_weights = 'weightsRegFor1NATURELSFire_1' if model_version == 'calibratedWeights' else 'weightsBinaryFor1NATURELSFire_1'
val_weights = 'weightsRegFor1NATURELSFire_1'

test_weights = 'weightsRegFor1NATURELSFire_1'

## Quantile:
alpha = [0.05,0.5,0.95]

### Metrics
metrics_binary = {
            #'precision' : {'func' : precision_score, 'use_proba': False, 'use_weights':True, 'params' : {'average':'micro', 'labels':[0,1,2,3,4]}},
           #'recall' : {'func' : recall_score, 'use_proba': False, 'use_weights':True,'params' : {'average':'micro', 'labels':[0,1,2,3,4]}},
           #'accuracy' : {'func' : accuracy_score, 'use_proba': False, 'use_weights':True,'params' : {}},
           #'f1score' : {'func' : f1_score, 'use_proba': False, 'use_weights':True,'params' : {'average':'micro', 'labels':[0,1,2,3,4]}},
           'pinball' : {'func': mean_pinball_loss, 'use_proba': True, 'use_weights':True, 'pos_alpha' : 0, 'params' : {'alpha' : 0.5}},
           'logloss' : {'func' : log_loss, 'use_proba': True, 'use_weights':True,'pos_alpha' : 0, 'params' : {'labels':[0, 1]}},
            'ECE' : {'func' : expected_calibration_error, 'use_proba': True, 'use_weights':False,'pos_alpha' : 0, 'params' : {}}
           }

metrics_regressive = {
            'rmse' : {'func': root_mean_squared_error, 'use_proba': True, 'use_weights':False, 'use_nonfire': True, 'params' : {}, 'target' : 'calibratedProbaFor1NATURELSFiretest_1'},
            'logloss' : {'func' : log_loss, 'use_proba': True, 'use_weights':False, 'use_nonfire': True, 'pos_alpha' : 1, 'params' : {'labels':[0, 1]}, 'target' : 'is1NATURELSFire_1'},
            'firelogloss' : {'func' : log_loss, 'use_proba': True, 'use_weights':False, 'use_nonfire': False, 'pos_alpha' : 1, 'params' : {'labels':[0, 1]}, 'target' : 'is1NATURELSFire_1'},
            'ECE' : {'func' : expected_calibration_error, 'use_proba': True, 'use_weights':False, 'use_nonfire': True, 'pos_alpha' : 0, 'params' : {}, 'target' : 'is1NATURELSFire_1'},
            #'pinball_0' : {'func': mean_pinball_loss, 'use_proba': False, 'use_weights':True, 'pos_alpha' : 0, 'params' : {'alpha' : 0.05}},
            #'pinball_1' : {'func': mean_pinball_loss, 'use_proba': False, 'use_weights':True, 'pos_alpha' : 2, 'params' : {'alpha' : 0.95}},
            #'pinball_2' : {'funwc': mean_pinball_loss, 'use_proba': False, 'use_weights':True, 'pos_alpha' : 1, 'params' : {'alpha' : 0.5}}
           }

#['calibratedProbaFor1NATURELSFire'] + ['calibratedProbaFor1NATURELSFire_' + str(day) for day in range(1, 11)]
config = {'target' : 'calibratedProbaFor1NATURELSFire_1', 'return_proba': True, 'metrics': metrics_regressive,
                 'alpha': [0.5],
                 'type':'fit',
                 'prefit':False}

print(config)

train.dropna(subset=config['target'], inplace=True)
train.reset_index(drop=True, inplace=True)
generalizeTest.dropna(subset='calibratedProbaFor1NATURELSFiretest_1', inplace=True)
generalizeTest.reset_index(drop=True, inplace=True)
val.dropna(subset=config['target'], inplace=True)
val.reset_index(drop=True, inplace=True)

train['is1NATURELSFire_1'] = train['is1NATURELSFire_1'].astype(int)
val['is1NATURELSFire_1'] = val['is1NATURELSFire_1'].astype(int)
generalizeTest['is1NATURELSFire_1'] = generalizeTest['is1NATURELSFire_1'].astype(int)

Ain = np.arange(0,10)
rhone = np.arange(680,690)
Doubs = np.arange(240, 250)
Yvelines = np.arange(770,780)

test_base = [#{'name' : 'testSet', 'df': test, 'weights' : test_weights},
             #{'name' : 'testSetFire', 'df': test[test['isFire'] == 1], 'weights' : test_weights},
             {'name' : 'test2023', 'df': generalizeTest[generalizeTest['departement'] != 69], 'weights' : test_weights},
             {'name' : 'test69', 'df': generalizeTest[generalizeTest['departement'] == 69], 'weights' : test_weights},
             #{'name' : 'test2023Fire', 'df': generalizeTest[generalizeTest['isNATURELSFire'] == 1], 'weights' : test_weights},
             {'name' : 'test2023Winter', 'df': generalizeTest[generalizeTest['month'] < 6], 'weights' : test_weights},
             {'name' : 'test2023Summer', 'df': generalizeTest[generalizeTest['month'] >= 6], 'weights' : test_weights},
            {'name' : 'ain', 'df': generalizeTest[generalizeTest['cluster'].isin(Ain)], 'weights' : test_weights},
            {'name' : 'doubs', 'df': generalizeTest[generalizeTest['cluster'].isin(Doubs)], 'weights' : test_weights},
            {'name' : 'rhone', 'df': generalizeTest[generalizeTest['cluster'].isin(rhone)], 'weights' : test_weights},
            {'name' : 'yvelines', 'df': generalizeTest[generalizeTest['cluster'].isin(Yvelines)], 'weights' : test_weights}]

###################### Same ##########################
if train_basic:
    ####################### Binary #######################
    """name = 'bernouilli_binary'
    dir_output = dir_dataset / config['type'] / model_version / name
    
    model = Bernouilli(train[train['isFire'] == 1] / len(train), 42, 2)

    saving = True

    launcher = Launcher(dir_output, model, train, val,  name, variables, config, saving)
    pipeline(launcher, tuning, doCalibration, test_base, searchparams={}, fitparams={})"""

    ####################### Regressive #######################
    name = 'progressive'
    dir_output = dir_dataset / config['type'] / model_version / name

    model = Progressive(1, generalizeTest)
    saving = True

    launcher = Launcher(dir_output, model, train, val, name, variables, config, saving)
    pipeline(launcher, tuning, doCalibration, [test_base[2]], searchParams={}, fitparams={})

####################### XGBoost ########################

def config_xgboost(ty):
    params = {
        'verbosity':0,
        'early_stopping_rounds':15,
        'learning_rate' :0.01,
        'min_child_weight' : 1.0,
        'max_depth' : 6,
        'max_delta_step' : 1.0,
        'subsample' : 0.5,
        'colsample_bytree' : 0.7,
        'colsample_bylevel': 0.6,
        'reg_lambda' : 1.7,
        'reg_alpha' : 0.7,
        #'gamma' : 2.5, 
        'n_estimators' : 10000,
        'random_state': 42,
        "quantile_alpha": alpha,
        'tree_method':'hist',
        }
    
    if GPU:
        params['device']='cuda'
    
    if ty == 'multiregression':
        params['multi_strategy'] = "multi_output_tree"

    if ty == 'binary':
        #params['objective'] = 'binary:logistic',
        return XGBClassifier(**params,
                             objective = 'binary:logistic')
    else:
        return XGBRegressor(**params,
                            #objective = 'reg:quantileerror'
                            objective = 'reg:squarederror'
                            )

if train_xgboost:
    
    searchParams = make_tuning_param(tuning, 'xgboost')

    name = 'xgboost'
    dir_output = dir_dataset / config['type'] / model_version / name

    model = config_xgboost(config['type'])

    saving = True

    fitparams={
                'verbose':True,
                'eval_set':[(train[variables], train[config['target']]), (val[variables], val[config['target']])],
                #'sample_weight_eval_set': [train['weights_binary'].values, val['weights_binary'].values]
                }

    if model_version != 'noWeights':
        fitparams['sample_weight'] = train[train_weights]

    launcher = Launcher(dir_output, model, train, val,  name, variables, config, saving)
    pipeline(launcher, tuning, doCalibration, test_base, searchParams, fitparams)

####################### LightGBM ################################

def config_lightGBM(ty):
    params = {'verbosity':-1,
        #'num_leaves':64,
        'learning_rate':0.01,
        'early_stopping_rounds': 15,
        'bagging_fraction':0.7,
        'colsample_bytree':0.6,
        'max_depth' : 4,
        'num_leaves' : 2**4,
        'reg_lambda' : 1,
        'reg_alpha' : 0.27,
        #'gamma' : 2.5,
        'num_iterations' :10000,
        'random_state':42
        }

    if GPU:
        params['device'] = "gpu"

    if ty == 'binary':
        params['objective'] = 'binary',
        return LGBMClassifier(**params)
    else:
        params['objective'] = 'root_mean_squared_error',
        return LGBMRegressor(**params)

if train_lightgbm:
    searchParams = make_tuning_param(tuning, 'lightgbm')

    name = 'lightgbm'
    dir_output = dir_dataset / config['type'] / model_version / name

    model = config_lightGBM(config['type'])

    saving = True

    fitparams={
                'eval_set':[(train[variables], train[config['target']]), (val[variables], val[config['target']])],
                #'eval_sample_weight' : [train[train_binary_weight], val[train_binary_weight]]
                }

    if model_version != 'noWeights':
        fitparams['sample_weight'] = train[train_weights]

    launcher = Launcher(dir_output, model, train, val,  name, variables, config, saving)
    pipeline(launcher, tuning, doCalibration, test_base, searchParams, fitparams)

####################### NGBoost ########################################

def config_ngboost(ty):
    params  = {
        'natural_gradient':True,
        'n_estimators':1000,
        'learning_rate':0.01,
        'minibatch_frac':0.7,
        'col_sample':0.6,
        'verbose':True,
        'verbose_eval':100,
        'tol':1e-4,
    }

    if ty == 'binary':
        """Base = XGBClassifier(max_depth= 6, n_estimators= 300, verbosity= 1, objective= 
                            'binary:logistic',  booster= 'gbtree', tree_method= 'hist', n_jobs=-1, learning_rate= 0.05, gamma= 0.15,
                            reg_alpha= 0.20, 
                            reg_lambda= 0.50, random_state=42, device='cuda')"""
        #Base = LGBMClassifier(device='gpu', random_state=42, reg_lambda=0.5, reg_alpha=0.2, )
        return NGBClassifier(**params)
    else:
        Base = XGBRegressor(max_depth= 6, n_estimators= 300, verbosity= 1, objective= 
                            'reg:squarederror',  booster= 'gbtree', tree_method= 'hist', n_jobs=-1, learning_rate= 0.05, gamma= 0.15,
                            reg_alpha= 0.20, 
                            reg_lambda= 0.50, random_state=42, device='cuda')
        
        """Base = LGBMRegressor(device='gpu', random_state=42, reg_lambda=0.5, reg_alpha=0.2, )"""
        return NGBRegressor(**params, Base=Base)

if train_ngboost:
    searchParams = make_tuning_param(tuning, 'ngboost')

    name = 'ngboost'
    print(dir_dataset / config['type'] / model_version / name)
    dir_output = dir_dataset / config['type'] / model_version / name
    print(dir_output)
    model = config_ngboost(config['type'])

    saving = True

    fitparams = {
        'early_stopping_rounds':15,
        'X_val':val[variables],
        'Y_val':val[config['target']],
    }

    if model_version != 'noWeights':
        fitparams['sample_weight'] = train[train_weights]

    launcher = Launcher(dir_output, model, train, val,  name, variables, config, saving)
    pipeline(launcher, tuning, doCalibration, test_base, searchParams, fitparams)

############################## Random Forest ###########################################

def config_rf(ty):
    
    params = {'verbose':0,
        'max_depth' : 10,
        'n_estimators' : 1000,
        'min_impurity_decrease':0.01,
        'max_features' : 0.7,
        'min_samples_leaf': 1,
        'min_samples_split' : 2,
        }

    if ty == 'binary':
        params['criterion'] = 'log_loss'
        return RandomForestClassifier(**params)
    params['criterion'] = 'squared_error'
    return RandomForestRegressor()

if train_rf:
    searchParams = make_tuning_param(tuning, 'rf')

    name = 'rf'
    dir_output = dir_dataset / config['type'] / model_version / name

    model = config_rf(config['type'])

    saving = True

    fitparams = {
    }

    if model_version != 'noWeights':
        fitparams['sample_weight'] = train[train_weights]

    launcher = Launcher(dir_output, model, train, val,  name, variables, config, saving)
    pipeline(launcher, tuning, doCalibration, test_base, searchParams, fitparams)