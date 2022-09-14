try:
    import pandas as pd
except ImportError:
    pip.main(['install', 'pandas'])
finally:
    import pandas as pd
    
try:
    import numpy as np
except ImportError:
    pip.main(['install', 'numpy'])
finally:
    import numpy as np
    
try:
    import scipy.stats as stats
except ImportError:
    pip.main(['install', 'scipy'])
finally:
    import scipy.stats as stats
    
try:
    import matplotlib.pyplot as plt
except ImportError:
    pip.main(['install', 'matplotlib'])
finally:
    import matplotlib.pyplot as plt
    
try:
    import seaborn as sns
except ImportError:
    pip.main(['install', 'seaborn'])
finally:
    import seaborn as sns

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

import os
import warnings

plt.rcParams["font.family"] = 'NanumGothicCoding.ttf'
sns.set_theme(style="whitegrid")
warnings.filterwarnings('ignore')

def show_targets():
    """
        분석 가능한 질병 보여주기
    """
    f = open('./data/Chronic disease list.txt')
    lines = f.readlines()
    for line in lines:
        print(line)

def confusion_matrix(data, target, meta_data, features='all', visualization=False, n=None, save=True):
    """
        혼동 행렬 그림
        
        args
            data: pd.DataFrame()
            target: str
            meta_data: pd.DataFrame()
            features: 'all', list
            visualization: bool
            n: int
            save: bool
            
        return
            results: pd.DataFrame()
    """
    results = pd.DataFrame([], columns=['feature', 'Chi-squared', 'p-value'])
    if features == 'all':
        features = list(data.columns)
        features.remove(target)
        i = 0
        for feature in features:
            table = pd.crosstab(index=data[feature], columns=data[target])
            group_counts = [value for value in table.values.flatten()]
            group_percentages = ['{0:.2%}'.format(value) for value in table.values.flatten()/np.sum(table.values)]
            labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
            labels = np.asarray(labels).reshape(table.shape[0], table.shape[1])
            
            chi_squared, p_value = stats.chi2_contingency(observed=table)[:2]
            results.loc[i] = [feature, chi_squared, p_value]
            i +=1
            
            cur_path = os.getcwd()
            if visualization == True:
                print('Feature: {}'.format(feature))

                plt.figsize=(5, 5)
                plt.rcParams['axes.unicode_minus'] = False
                plt.rc('font', family='NanumGothic')
                ax = sns.heatmap(table, cmap='Blues', annot=labels, fmt='')
                plt.xlabel('{} 유병 여부'.format(target))
                if save == True:
                    path = cur_path + '/results/{}/confusion_matrix'.format(target)

                    if not os.path.isdir(path):
                        os.makedirs(path)
                        
                    plt.savefig('{}/confusion_matrix({} vs {}).png'.format(path, target, feature), dpi=300)
                plt.show()
                
            elif visualization == False:
                pass
            
    elif features != 'all':
        if n == None:
            pass
        if n != None:
            features = features[:n]
        i = 0
        for feature in features:
            table = pd.crosstab(index=data[feature], columns=data[target])
            group_counts = [value for value in table.values.flatten()]
            group_percentages = ['{0:.2%}'.format(value) for value in table.values.flatten()/np.sum(table.values)]
            labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_counts, group_percentages)]
            labels = np.asarray(labels).reshape(table.shape[0], table.shape[1])
            
            chi_squared, p_value = stats.chi2_contingency(observed=table)[:2]
            results.loc[i] = [feature, chi_squared, p_value]
            i +=1
            
            cur_path = os.getcwd()
            if visualization == True:
                print('Feature: {}'.format(feature))

                plt.figsize=(5, 5)
                plt.rcParams['axes.unicode_minus'] = False
                plt.rc('font', family='NanumGothic')
                ax = sns.heatmap(table, cmap='Blues', annot=labels, fmt='')
                plt.xlabel('{} 유병 여부'.format(target))
                if save == True:
                    path = cur_path + '/results/{}/confusion_matrix'.format(target)

                    if not os.path.isdir(path):
                        os.makedirs(path)
                        
                    plt.savefig('{}/confusion_matrix({} vs {}).png'.format(path, target, feature), dpi=300)
                plt.show()
                
            elif visualization == False:
                pass
    
    dic_eng2kor = {}
    for eng, kor in zip(meta_data['variable'], meta_data['variable description']):
        dic_eng2kor[eng] = kor
        
    results['feature_korean'] = results['feature'].map(dic_eng2kor)
    results = results[['feature', 'feature_korean', 'Chi-squared', 'p-value']]    
    results = results.sort_values(by='p-value')
    results.index = range(len(results))
    
    return results

def metric_plot(results, target, metrics=['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC', 'Training Time'], save=True):
    """
        머신러닝 지표 비교 그림
        
        args
            results: pd.DataFrame()
            target: str
            metrics: list
            save: bool
    """
    cur_path = os.getcwd()
    path = cur_path + '/results/{}/metric_plots'.format(target)

    if not os.path.isdir(path):
        os.makedirs(path)
        
    plt.figsize=(5, 5)
    for metric in metrics:
        ax = sns.barplot(x='Model', y=metric, data=results, order=results.groupby('Model').mean().sort_values(metric).index, ci='sd')
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2, height / 2, round(height, 5), ha = 'center')
        plt.xlabel('Model')
        plt.xticks(rotation=30)
        plt.ylabel(metric)
        
        if save == True:
            plt.savefig('{}/{}.png'.format(path, metric), dpi=300)
            
        elif save == False:
            pass
        
        plt.show() 
    
def factor_extraction(data, target, models=['Random Forest', 'AdaBoost', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost'], one_hot_encoding=True, n=40, random_state=42, visualization=False, save=True):
    """
        영향 요인 추출
        
        args
            data: pd.DataFrame()
            target: str
            models: list
            one_hot_encoding: bool
            n: int
            random_state: int
            visualization: bool
            save: bool
            
        return
            results: pd.DataFrame()
    """
    X = data.drop(target, axis=1)
    y = data[target]
    
    if one_hot_encoding == True:
        X = X.astype(str)
        X = pd.get_dummies(X)
    else:
        pass
    
    index = list(X.columns)
    def underbar_split(string):
        return string.rpartition('_')[0]
    index = list(set(list(map(underbar_split, index))))
    results = pd.DataFrame([], index=index)

    for model in models:
        if model == 'Random Forest':
            clf = RandomForestClassifier(n_estimators=1000)
        elif model == 'AdaBoost':
            clf = AdaBoostClassifier(n_estimators=1000)
        elif model == 'Gradient Boosting':
            clf = GradientBoostingClassifier(n_estimators=1000)
        elif model == 'XGBoost':
            clf = XGBClassifier(n_estimators=1000)
        elif model == 'LightGBM':
            clf = LGBMClassifier(n_estimators=1000)
        elif model == 'CatBoost':
            clf = CatBoostClassifier()
            
        clf.fit(X, y)
        
        df = pd.DataFrame({'feature':X.columns, 'feature importances':clf.feature_importances_})
        dic_question2importance = {}
        for i in range(df.shape[0]):
            feature, importance = df.iloc[i, :]
            question = feature.rpartition('_')[0]
            if question not in dic_question2importance:
                dic_question2importance[question] = []
            dic_question2importance[question].append(importance)

        lst_question = []
        lst_importance = []

        for question, importances in dic_question2importance.items():
            lst_question.append(question)
            lst_importance.append(np.mean(importances))

        df = pd.DataFrame({'feature':lst_question, 'feature importances':lst_importance})
        df = df.sort_values(by='feature importances', ascending=False)
        
        if visualization == True:
            plt.title('{} Feature Importances'.format(model))
            plt.xlabel('Feature Importances')
            plt.ylabel('Feature')
            ax = sns.barplot(x='feature importances', y='feature', data=df.iloc[:n])
            plt.show()
            
        elif visualization == False:
            pass
        
        df[model] = range(len(df), 0, -1)
        df.index = df['feature']
        
        results = pd.concat([results, df[model]], axis=1)
        
    results['Average Score'] = results.mean(axis=1)
    results = results.sort_values('Average Score', ascending=False)
    
    cur_path = os.getcwd()
    if save == True:
        path = cur_path + '/results/{}'.format(target)

        if not os.path.isdir(path):
            os.makedirs(path)
            
        results.to_csv('{}/{}_factors.csv'.format(path, target), sep='\t')
        
    elif save == False:
        pass
    
    return results

def score_trendplot(data, n=100):
    """
        머신러닝 모델들이 준 평균 점수 그림
        
        args
            data: pd.DataFrame()
            n: int
    """
    data = data.iloc[:n]
    ax = sns.lineplot(data=data, x=range(len(data)), y='Average Score')
    plt.show()