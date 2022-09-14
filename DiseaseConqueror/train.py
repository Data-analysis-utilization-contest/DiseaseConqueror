try:
    import pandas as pd
except ImportError:
    pip.main(['install', 'pandas'])
finally:
    import pandas as pd
    
try:
    import sklearn
except ImportError:
    pip.main(['install', 'sklearn'])
finally:
    import sklearn
    
try:
    from xgboost import XGBClassifier
except ImportError:
    pip.main(['install', 'xgboost'])
finally:
    from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
except ImportError:
    pip.main(['install', 'lightgbm'])
finally:
    from lightgbm import LGBMClassifier

try:
    from catboost import CatBoostClassifier
except ImportError:
    pip.main(['install', 'catboost'])
finally:
    from catboost import CatBoostClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

import os
import time
import warnings

warnings.filterwarnings('ignore')

def modeling(data, target, models=['Random Forest', 'AdaBoost', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost'], one_hot_encoding=True, n_splits=5, test_size=0.33, random_state=42, save=True):
    """
        학습하기
        
        args
            data: pd.DataFrame()
            target: str
            models: list
            one_hot_encoding: bool
            n_splits: int
            test_size: float
            random_state: int
            save: bool
            
        return
            df: pd.DataFrame()
    """
    X = data.drop(target, axis=1)
    y = data[target]
    
    if one_hot_encoding == True:
        X = X.astype(str)
        X = pd.get_dummies(X)
    else:
        pass
    
    results = []
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    
    print('-' * 100)
    print('학습 시작')
    for model in models:
        iteration = 1
        print('-' * 100)
        if model == 'Random Forest':
            clf = RandomForestClassifier(n_estimators=100)
        elif model == 'AdaBoost':
            clf = AdaBoostClassifier(n_estimators=100)
        elif model == 'Gradient Boosting':
            clf = GradientBoostingClassifier(n_estimators=100)
        elif model == 'XGBoost':
            clf = XGBClassifier(n_estimators=100)
        elif model == 'LightGBM':
            clf = LGBMClassifier(n_estimators=100)
        elif model == 'CatBoost':
            clf = CatBoostClassifier()
                
        for train_idx, test_idx in skf.split(X,y):
            print('Model: {}, iteration: {}'.format(model, iteration))
            X_train = X.iloc[train_idx, :-1]
            X_test = X.iloc[test_idx, :-1]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            start_time = time.time()
            clf.fit(X_train, y_train)
            end_time = time.time()
            training_time = end_time - start_time

            pred = clf.predict(X_test)
            pred_proba = clf.predict_proba(X_test)    
            accuracy = accuracy_score(y_test, pred)    
            precision = precision_score(y_test, pred)
            recall = recall_score(y_test, pred)
            F1_score = f1_score(y_test, pred)
            auc = roc_auc_score(y_test, pred_proba[:,1])

            results.append([model, iteration, accuracy, precision, recall, F1_score, auc, training_time])
            
            iteration +=1
    print('-' * 100)
    print('Model: {} {}개 학습 완료'.format(models, len(models)))
    print('-' * 100)
    
    df = pd.DataFrame(
        data=results, 
        columns=['Model', 'Iteration', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC', 'Training Time'])
    
    cur_path = os.getcwd()
    if save == True:
        path = cur_path + '/results/{}'.format(target)

        if not os.path.isdir(path):
            os.makedirs(path)
            
        df.to_csv('{}/{}-fold_cv_results.csv'.format(path, n_splits), sep='\t')
        
    elif save == False:
        pass
        
    return df