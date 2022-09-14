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