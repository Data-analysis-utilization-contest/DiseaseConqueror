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

import os
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