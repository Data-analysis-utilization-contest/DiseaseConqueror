import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
    
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

import os
import time
import warnings

warnings.filterwarnings('ignore')

def dataloader(datadir, year, target):
    """
    데이터 로드
    
    args
        datadir: 데이터 경로
        year: 연도
        target: 질병
        
    return
        data: pd.DataFrame()
        meta_data: pd.DataFrame()
    """
    year = str(year)[2:]
    data = pd.read_sas(os.path.join(datadir, 'hn{}_all.sas7bdat'.format(year)))
    meta_data = pd.read_excel(os.path.join(datadir, 'meta_data{}.xlsx'.format(year)))
    
    if target == '비만':
        data = add_target1(data)       
    elif target == '고혈압':
        data = add_target2(data)        
    elif target == '당뇨병':
        data = add_target3(data)          
    elif target == '고콜레스테롤혈증':
        data = add_target4(data)       
    elif target == '고중성지방혈증':
        data = add_target5(data)        
    elif target == '만성폐쇄성폐질환':
        print('해당년도는 만성폐쇄성폐질환 유병률 분석은 지원하지 않습니다. 다른 질병을 선택하여 주시기 바랍니다.')        
    elif target == 'B형간염':
        data = add_target6(data)           
    elif target == '빈혈':
        data = add_target7(data)        
    elif target == '만성콩팥병':
        data = add_target8(data)        
    elif target == '영구치우식':
        print('해당년도는 영구치우식 유병률 분석은 지원하지 않습니다. 다른 질병을 선택하여 주시기 바랍니다.')        
    elif target == '구강기능제한':
        data = add_target9(data)           
    elif target == '저작불편호소':
        data = add_target10(data)        
    elif target == '뇌졸중':
        data = add_target11(data)
    elif target == '천식':
        data = add_target12(data)           
    elif target == '알레르기비염':
        data = add_target13(data)        
    elif target == '아토피피부염':
        data = add_target14(data)
    
    idx = []
    columns = []
    
    for dtype in ['object', 'statistics']:
        lst_index = meta_data[meta_data['data type'] == dtype].index
        for index in lst_index:
            idx.append(index)
    idx.sort()
    
    for index in idx:
        var = meta_data['variable'].loc[index]
        columns.append(var)

    data = data.drop(columns, axis=1)
    data.index = range(len(data))
    
    print('총 인원: {}명'.format(data.shape[0]))
    print('{}: {}명'.format(target, data[target].value_counts()[1]))
    print('정상: {}명'.format(data[target].value_counts()[0]))
    print('{} 유병률: {:.2f}%'.format(target, data[target].value_counts()[1] / data.shape[0] * 100))
        
    return data, meta_data
        
def add_target1(data):
    """
    질병 선택 1
    
    - 질병: 비만
    
    args
        data: pd.DataFrame()
        
    return
        data: pd.DataFrame()
    """
    
    data = data[data['age'] >= 19]
    data = data.dropna(subset=['HE_BMI'])
    idx = []
    for index in data.dropna(subset=['HE_dprg']).index:
        idx.append(index)
    data = data.drop(idx, axis=0)
    data.index = range(len(data))
    data['비만'] = (data['HE_BMI'] >= 25).astype(int)
    
    return data
    
def add_target2(data):
    """
    질병 선택 2
    
    - 질병: 고혈압
    
    args
        data: pd.DataFrame()
        
    return
        data: pd.DataFrame()
    """
    
    data = data[data['age'] >= 19]
    data = data.dropna(subset=['HE_sbp1', 'HE_sbp2', 'HE_sbp3', 'HE_dbp1', 'HE_dbp2', 'HE_dbp3'])
    data = data[data['DI1_2'].isin([1, 2, 3, 4, 5, 8])]
    data['고혈압'] = data['HE_HP'].map({1: 0, 2: 0, 3: 1})
    
    return data
    
def add_target3(data):
    """
    질병 선택 3
    
    - 질병: 당뇨병
    
    args
        data: pd.DataFrame()
        
    return
        data: pd.DataFrame()
    """
    
    data = data[(data['age'] >= 19) & (data['HE_fst'] >= 8)]
    data = data[(data['DE1_dg'].isin([0, 1, 8])) & (data['DE1_31'].isin([0, 1, 8])) & (data['DE1_32'].isin([0, 1, 8]))]
    data = data.dropna(subset=['HE_glu', 'HE_HbA1c'])
    idx = []
    for index in data.dropna(subset=['HE_dprg']).index:
        idx.append(index)
    data = data.drop(idx, axis=0)
    data.index = range(len(data))
    data['당뇨병'] = data['HE_DM_HbA1c'].map({1: 0, 2: 0, 3: 1})
    
    return data
    
def add_target4(data):
    """
    질병 선택 4
    
    - 질병: 고콜레스테롤혈증
    
    args
        data: pd.DataFrame()
        
    return
        data: pd.DataFrame()
    """
    
    data = data[(data['age'] >= 19) & (data['HE_fst'] >= 8)]
    data = data.dropna(subset=['HE_chol', 'DI2_2'])
    data = data[data['DI2_2'].isin([1, 2, 3, 4, 5, 8])]
    data.index = range(len(data))
    lst = [0] * data.shape[0]
    index = data[(data['HE_chol'] >= 240) | (data['DI2_2'] == 1)].index
    for idx in index:
        lst[idx] = 1
    data['고콜레스테롤혈증'] = lst
    
    return data
    
def add_target5(data):
    """
    질병 선택 5
    
    - 질병: 고중성지방혈증
    
    args
        data: pd.DataFrame()
        
    return
        data: pd.DataFrame()
    """
    
    data = data[(data['age'] >= 19) & (data['HE_fst'] >= 12)]
    data = data.dropna(subset=['HE_TG'])
    data.index = range(len(data))
    data['고중성지방혈증'] = (data['HE_TG'] >= 200).astype(int)
    
    return data
    
def add_target6(data):
    """
    질병 선택 6
    
    - 질병: B형간염
    
    args
        data: pd.DataFrame()
        
    return
        data: pd.DataFrame()
    """
    
    data = data[data['age'] >= 10]
    data = data.dropna(subset=['HE_hepaB'])
    data = data[data['HE_hepaB'].isin([0, 1])]
    data.index = range(len(data))
    data['B형간염'] = data['HE_hepaB']
    
    return data

def add_target7(data):
    """
    질병 선택 7
    
    - 질병: 빈혈
    
    args
        data: pd.DataFrame()
        
    return
        data: pd.DataFrame()
    """
    
    data = data[data['age'] >= 10]
    data = data.dropna(subset=['HE_anem'])
    data = data[data['HE_anem'].isin([0, 1])]
    data.index = range(len(data))
    data['빈혈'] = data['HE_anem']
    
    return data
    
def add_target8(data):
    """
    질병 선택 8
    
    - 질병: 만성콩팥병
    
    args
        data: pd.DataFrame()
        
    return
        data: pd.DataFrame()
    """
    
    data = data[data['age'] >= 19]
    data = data.dropna(subset=['HE_crea', 'HE_Ualb','HE_Ucrea'])
    data.index = range(len(data))
    temp = data[['sex', 'age', 'HE_crea', 'HE_Ualb','HE_Ucrea']]
    lst = []
    for i in range(len(temp)):
        if temp.loc[i]['sex'] == 1:
            if temp.loc[i]['HE_crea'] <= 0.9:
                lst.append(141 * (temp.loc[i]['HE_crea']**(-0.411)) * (0.993)**(temp.loc[i]['age']))
            elif temp.loc[i]['HE_crea'] > 0.9:
                lst.append(141 * (temp.loc[i]['HE_crea']**(-1.209)) * (0.993)**(temp.loc[i]['age']))
        elif temp.loc[i]['sex'] == 2:
            if temp.loc[i]['HE_crea'] <= 0.7:
                lst.append(144 * (temp.loc[i]['HE_crea']**(-0.329)) * (0.993)**(temp.loc[i]['age']))
            elif temp.loc[i]['HE_crea'] > 0.7:
                lst.append(144 * (temp.loc[i]['HE_crea']**(-1.209)) * (0.993)**(temp.loc[i]['age']))
    temp['사구체 여과율'] = lst
    data['만성콩팥병'] = (((temp['사구체 여과율'] >= 60) & (temp['HE_Ualb'] / temp['HE_Ucrea'] >= 0.3)) | (temp['사구체 여과율'] < 60)).astype(int)
    
    return data
    
def add_target9(data):
    """
    질병 선택 9
    
    - 질병: 구강기능제한율
    
    args
        data: pd.DataFrame()
        
    return
        data: pd.DataFrame()
    """
    
    data = data[data['age'] >= 19]
    data = data.dropna(subset=['BM7', 'O_chew_d', 'BM8'])
    data = data[(data['BM7'].isin([1, 2, 3, 4, 5, 8])) & (data['BM8'].isin([1, 2, 3, 4, 5, 8]))]
    data.index = range(len(data))
    data['구강기능제한'] = (data['BM7'].isin([1, 2]) | data['BM8'].isin([1, 2])).astype(int)
    
    return data    
    
def add_target10(data):
    """
    질병 선택 10
    
    - 질병: 저작불편호소율
    
    args
        data: pd.DataFrame()
        
    return
        data: pd.DataFrame()
    """
    
    data = data[data['age'] >= 19]
    data = data.dropna(subset=['BM7', 'O_chew_d'])
    data = data[(data['BM7'].isin([1, 2, 3, 4, 5, 8]))]
    data.index = range(len(data))
    data['저작불편호소'] = data['O_chew_d']

    return data
    
def add_target11(data):
    """
    질병 선택 11
    
    - 질병: 뇌졸중
    
    args
        data: pd.DataFrame()
        
    return
        data: pd.DataFrame()
    """
    
    data = data[data['age'] >= 30]
    data = data.dropna(subset=['DI3_dg'])
    data = data[data['DI3_dg'].isin([0, 1])]
    data.index = range(len(data))
    data['뇌졸중'] = data['DI3_dg']
        
    return data
    
def add_target12(data):
    """
    질병 선택 12
    
    - 질병: 천식
    
    args
        data: pd.DataFrame()
        
    return
        data: pd.DataFrame()
    """
    
    data = data[data['age'] >= 19]
    data = data.dropna(subset=['DJ4_dg'])
    data = data[data['DJ4_dg'].isin([0, 1])]
    data.index = range(len(data))
    data['천식'] = data['DJ4_dg']
    
    return data
    
def add_target13(data):
    """
    질병 선택 13
    
    - 질병: 알레르기비염
    
    args
        data: pd.DataFrame()
        
    return
        data: pd.DataFrame()
    """
    
    data = data[data['age'] >= 19]
    data = data.dropna(subset=['DJ8_dg'])
    data = data[data['DJ8_dg'].isin([0, 1])]
    data.index = range(len(data))
    data['알레르기비염'] = data['DJ8_dg']
    
    return data
    
def add_target14(data):
    """
    질병 선택 14
    
    - 질병: 아토피피부염
    
    args
        data: pd.DataFrame()
        
    return
        data: pd.DataFrame()
    """
    
    data = data[data['age'] >= 19]
    data = data.dropna(subset=['DL1_dg'])
    data = data[data['DL1_dg'].isin([0, 1])]
    data.index = range(len(data))
    data['아토피피부염'] = data['DL1_dg']
        
    return data

def fill_exception(data, meta_data, na=-1, not_applicable=-2, unknown=-3):
    """
        결측값, 비해당, 무응답 채우기
        
        args
            data: pd.DataFrame()
            meta_data: pd.DataFrame()
            na: int
            not_applicable: int
            unknwon: int
            
        return
            data: pd.DataFrame()
    """
    data = data.fillna(na)
    
    lst_columns_na = []
    lst_columns_un = []
    lst_value_na = []
    lst_value_un = []
    
    for idx, value in zip(meta_data.index, meta_data['not applicable']):
        if np.isnan(value) == False:
            lst_columns_na.append(meta_data.loc[idx]['variable'])
            lst_value_na.append(value)
            
    for idx, value in zip(meta_data.index, meta_data['unknown']):
        if np.isnan(value) == False:
            lst_columns_un.append(meta_data.loc[idx]['variable'])
            lst_value_un.append(value)
    
    for column, value in zip(lst_columns_na, lst_value_na):
        data[column] = data[column].map(lambda x: {value: not_applicable}.get(x, x))
        
    for column, value in zip(lst_columns_un, lst_value_un):
        data[column] = data[column].map(lambda x: {value: unknown}.get(x, x))
    
    return data

def digitize(data, meta_data):
    """
        범주화 하기
        
        args
            data: pd.DataFrame()
            meta_data: pd.DataFrame()
            
        return
            data: pd.DataFrame()
    """
    variable = meta_data[(meta_data['data type'] == 'numeric') | (meta_data['data type'] == 'numeric_age')]['variable']
    variable_bins = bins = meta_data[(meta_data['data type'] == 'numeric') | (meta_data['data type'] == 'numeric_age')]['variable bins']
    for col, str_bins in zip(variable, variable_bins):
        bins = []
        for num in str_bins.split(':'):
            bins.append(float(num))
        if col in data.columns:
            data[col] = np.digitize(data[col], bins=bins)
        
    return data

def modeling(data, target, models=['RandomForest', 'AdaBoost', 'GradientBoosting', 'XGBoost', 'LightGBM'], one_hot_encoding=True, n_splits=5, test_size=0.33, random_state=42, save=True, prePath=None):
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
            prePath: str
            
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
        if model == 'RandomForest':
            clf = RandomForestClassifier(n_estimators=100)
        elif model == 'AdaBoost':
            clf = AdaBoostClassifier(n_estimators=100)
        elif model == 'GradientBoosting':
            clf = GradientBoostingClassifier(n_estimators=100)
        elif model == 'XGBoost':
            clf = XGBClassifier(n_estimators=100)
        elif model == 'LightGBM':
            clf = LGBMClassifier(n_estimators=100)
                
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
        columns=['Model', 'Iteration', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC', 'TrainingTime'])
    
    cur_path = prePath
    if save == True:
        path = os.path.join(cur_path, '{}'.format(target))

        if not os.path.isdir(path):
            os.makedirs(path)
            
        df.to_csv('{}/{}-fold_cv_results.csv'.format(path, n_splits))
        
    elif save == False:
        pass
        
    return df

def show_targets():
    """
        분석 가능한 질병 보여주기
    """
    f = open('./data/Chronic disease list.txt')
    lines = f.readlines()
    for line in lines:
        print(line)

def confusion_matrix(data, target, meta_data, features='all', visualization=False, n=None, save=True, prePath=None):
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
            prePath: str
            
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
            
            cur_path = prePath
            if visualization == True:
                print('Feature: {}'.format(meta_data[meta_data['feature'] == feature]['feature_korean']))

                plt.figsize=(5, 5)
                plt.rcParams['axes.unicode_minus'] = False
                plt.rc('font', family='NanumGothic')
                ax = sns.heatmap(table, cmap='Blues', annot=labels, fmt='')
                plt.xlabel('{} 유병 여부'.format(target))
                if save == True:
                    path = os.path.join(cur_path, '{}/confusion_matrix'.format(target))
                    
                    if not os.path.isdir(path):
                        os.makedirs(path)
                        
                    plt.savefig('{}/confusion_matrix({} vs {}).png'.format(path, target, feature), dpi=300)
                plt.show()
                plt.clf()

            elif visualization == False:
                pass
            plt.clf()

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
            
            cur_path = prePath
            if visualization == True:
                print('Feature: {}'.format(feature))

                plt.figsize=(5, 5)
                plt.rcParams['axes.unicode_minus'] = False
                plt.rc('font', family='NanumGothic')
                ax = sns.heatmap(table, cmap='Blues', annot=labels, fmt='')
                plt.xlabel('{} 유병 여부'.format(target))
                if save == True:
                    path = os.path.join(cur_path, '{}/confusion_matrix'.format(target))

                    if not os.path.isdir(path):
                        os.makedirs(path)
                        
                    plt.savefig('{}/confusion_matrix({} vs {}).png'.format(path, target, feature), dpi=300)
                plt.show()
                plt.clf()
                
            elif visualization == False:
                pass
            plt.clf()
    
    dic_eng2kor = {}
    for eng, kor in zip(meta_data['variable'], meta_data['variable description']):
        dic_eng2kor[eng] = kor
        
    results['feature_korean'] = results['feature'].map(dic_eng2kor)
    results = results[['feature', 'feature_korean', 'Chi-squared', 'p-value']]    
    results = results.sort_values(by='p-value')
    results.index = range(len(results))
    
    return results

def metric_plot(results, target, metrics=['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC', 'TrainingTime'], save=True, prePath=None):
    """
        머신러닝 지표 비교 그림
        
        args
            results: pd.DataFrame()
            target: str
            metrics: list
            save: bool
            prePath: str
    """
    cur_path = prePath
    path = os.path.join(cur_path, '{}/metric_plots'.format(target))

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
        plt.cla()
    
def factor_extraction(data, target, models=['RandomForest', 'AdaBoost', 'GradientBoosting', 'XGBoost', 'LightGBM'], one_hot_encoding=True, n=40, random_state=42, visualization=False):
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
        if model == 'RandomForest':
            clf = RandomForestClassifier(n_estimators=1000)
        elif model == 'AdaBoost':
            clf = AdaBoostClassifier(n_estimators=1000)
        elif model == 'GradientBoosting':
            clf = GradientBoostingClassifier(n_estimators=1000)
        elif model == 'XGBoost':
            clf = XGBClassifier(n_estimators=1000)
        elif model == 'LightGBM':
            clf = LGBMClassifier(n_estimators=1000)
            
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

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DiseaseConqueror Pipeline')
    parser.add_argument('-datadir', type=str, default='./data', help='Set data directory')
    parser.add_argument('-year', type=int, default=2020, choices=[2019, 2020], help='Select the year of data')
    parser.add_argument('-target', type=str, default='고혈압', choices=['비만', '고혈압', '당뇨병', '고콜레스테롤혈증', '고중성지방혈증', 'B형간염', '빈혈', '만성콩팥병', '구강기능제한', '저작불편호소', '뇌졸중', '천식', '알레르기비염', '아토피피부염'], help='Set target')
    parser.add_argument('-models', nargs='+', choices=['RandomForest', 'AdaBoost', 'GradientBoosting', 'XGBoost', 'LightGBM'], help='Choose a machine learning model to train on')
    parser.add_argument('-one_hot_encoding', type=bool, default=True, help='Choose whether to do one-hot encoding or not')
    parser.add_argument('-n_splits', type=int, default=5, help='Set number of folds')
    parser.add_argument('-test_size', type=float, default=0.33, help='Set test size')
    parser.add_argument('-random_state', type=int, default=42, help='Set random_state')
    parser.add_argument('-save', type=bool, default=True, help='Choose whether to save to a file or not')
    parser.add_argument('-metrics', nargs='+', choices=['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC', 'TrainingTime'], help='')
    parser.add_argument('-n1', type=int, default=40, help='Set n of factor_extraction function')
    parser.add_argument('-n2', type=int, default=20, help='Set n of confusion_matrix function')
    parser.add_argument('-visualization', type=bool, default=True, help='Choose whether to visualize or not')
    parser.add_argument('-prePath', help='path of output file', default='result', metavar='PREPATH', dest='PREPATH')
    args = parser.parse_args()
    
    # 1. 데이터 로드
    data, meta_data = dataloader(datadir=args.datadir, year=args.year, target=args.target)
    
    mapping = {}
    for variable, variable_description in zip(meta_data['variable'], meta_data['variable description']):
        mapping[variable] = variable_description
    
    # 2. 전처리
    data = fill_exception(data, meta_data)
    p_value1 = confusion_matrix(data, target=args.target, meta_data=meta_data, features='all', visualization=False, n=None, save=False, prePath=args.PREPATH)
    drop_columns = p_value1[p_value1['p-value'] <= 10e-300]['feature'].tolist()
    data = data.drop(drop_columns, axis=1)
    data = digitize(data, meta_data)
    
    # 3. 머신 러닝 학습
    results = modeling(data, target=args.target, models=args.models, one_hot_encoding=args.one_hot_encoding, n_splits=args.n_splits, test_size=args.test_size, random_state=args.random_state, save=args.save, prePath=args.PREPATH)
    metric_plot(results, target=args.target, metrics=args.metrics, save=args.save, prePath=args.PREPATH)
    
    # 4. 영향 요인 추출
    factor = factor_extraction(data, target=args.target, models=args.models, one_hot_encoding=args.one_hot_encoding, n=args.n1, random_state=args.random_state, visualization=args.visualization)
    p_value2 = confusion_matrix(data, target=args.target, meta_data=meta_data, features=factor.index.tolist(), visualization=args.visualization, n=args.n2, save=args.save, prePath=args.PREPATH)
    factor.index = factor.index.map(mapping)

    cur_path = os.path.join(args.PREPATH, '{}'.format(args.target)) 
    factor.to_csv('{}/{}_factors.csv'.format(cur_path, args.target))
    p_value2.to_csv('{}/p-value_of_factors.csv'.format(cur_path))