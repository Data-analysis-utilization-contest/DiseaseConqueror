try:
    import argparse
except ImportError:
    pip.main(['install', 'argparse'])
finally:
    import argparse

import os

from load import dataloader
from preprocessing import fill_exception, digitize
from train import modeling
from utils import confusion_matrix, metric_plot, factor_extraction

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='DiseaseConqueror Pipeline')
    parser.add_argument('--datadir', type=str, default='./data', help='Set data directory')
    parser.add_argument('--year', type=int, default=2020, help='Select the year of data')
    parser.add_argument('--target', type=str, default='고혈압', help='Set target')
    parser.add_argument('--models', type=list, default=['Random Forest', 'AdaBoost', 'Gradient Boosting', 'XGBoost', 'LightGBM', 'CatBoost'], help='Choose a machine learning model to train on')
    parser.add_argument('--one_hot_encoding', type=bool, default=True, help='Choose whether to do one-hot encoding or not')
    parser.add_argument('--n_splits', type=int, default=5, help='Set number of folds')
    parser.add_argument('--test_size', type=float, default=0.33, help='Set test size')
    parser.add_argument('--random_state', type=int, default=42, help='Set random_state')
    parser.add_argument('--save', type=bool, default=True, help='Choose whether to save to a file or not')
    parser.add_argument('--metrics', type=list, default=['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC', 'Training Time'], help='')
    parser.add_argument('--n1', type=int, default=40, help='Set n of factor_extraction function')
    parser.add_argument('--n2', type=int, default=20, help='Set n of confusion_matrix function')
    parser.add_argument('--visualization', type=bool, default=True, help='Choose whether to visualize or not')
    args = parser.parse_args()
    
    # 1. 데이터 로드
    data, meta_data = dataloader(datadir=args.datadir, year=args.year, target=args.target)
    
    # 2. 전처리
    data = fill_exception(data, meta_data)
    p_value1 = confusion_matrix(data, target=args.target, meta_data=meta_data, features='all', visualization=False, n=None, save=False)
    drop_columns = p_value1[p_value1['p-value'] <= 10e-300]['feature'].tolist()
    data = data.drop(drop_columns, axis=1)
    data = digitize(data, meta_data)
    
    # 3. 머신 러닝 학습
    results = modeling(data, target=args.target, models=args.models, one_hot_encoding=args.one_hot_encoding, n_splits=args.n_splits, test_size=args.test_size, random_state=args.random_state, save=args.save)
    metric_plot(results, target=args.target, metrics=args.metrics, save=args.save)
    
    # 4. 영향 요인 추출
    factor = factor_extraction(data, target=args.target, models=args.models, one_hot_encoding=args.one_hot_encoding, n=args.n1, random_state=args.random_state, visualization=args.visualization, save=args.save)
    p_value2 = confusion_matrix(data, target=args.target, meta_data=meta_data, features=factor.index.tolist(), visualization=args.visualization, n=args.n2, save=args.save)
    p_value2.to_csv('./results/{}/p-value_of_factors.csv'.format(args.target))