import pickle
import numpy as np
import pandas as pd
from .utils import parse_name_series
from .utils import create_new_features, new_features, fill_outliers, log_col, astype_numeric, apply_split_torque

def preparation_cat(df_features):
    df_cat = parse_name_series(df_features['name'])
    df_cat['drive'] = df_cat['drive'].map({'4X2': '2WD', '4X4': '4WD'})
    df_cat = df_cat.replace({None: np.nan, '': np.nan})
    df_cat.drop(['fuel', 'transmission'], axis=1, inplace=True)
    df_cat = pd.concat([df_cat, 
                        df_features[['fuel', 'transmission', 'owner', 'seats']]], 
                        axis=1)

    return df_cat

def preparation_for_model_1(df_features, model_features_names_path, ohe_path, ohe_features_names_path):
    with open(ohe_features_names_path, 'rb') as f:
        ohe_features = pickle.load(f)
    with open(ohe_path, 'rb') as f:
        ohe = pickle.load(f)
    with open(model_features_names_path, 'rb') as f:
        model_features = pickle.load(f)

    df_cat = preparation_cat(df_features)

    df_cat = df_cat.astype('str')
    df_cat['engine_displacement'] = df_cat['engine_displacement'].astype('float')
    df_cat['is_sport'] = df_cat['is_sport'].astype('int')
    df_cat['seats'] = df_cat['seats'].astype('int')
    
    df_cat_ohe = ohe.transform(df_cat)
    df_cat_ohe = pd.DataFrame(df_cat_ohe.toarray(),
                                  index=df_cat.index,
                                  columns=ohe.get_feature_names_out())
    
    return df_cat_ohe[model_features]

def preparation_for_model_2(df_features, model_features_names_path, model_IQRbounds_path, model_meanNum_path):
    df_features = astype_numeric(df_features, ['mileage', 'engine', 'max_power'], float)
    df_features = apply_split_torque(df_features)
    with open(model_IQRbounds_path, 'rb') as f:
        train_bounds = pickle.load(f)
    with open(model_features_names_path, 'rb') as f:
        model_features = pickle.load(f)
    with open(model_meanNum_path, 'rb') as f:
        meanNum = pickle.load(f)

    num_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'torque', 'max_torque_rpm']
    df_num = df_features[num_columns]

    df_num = create_new_features(df_num)
    
    for col in df_num:
        df_num[col] = df_num[col].fillna(meanNum[col])   
              
    df_num = fill_outliers(df_num, train_bounds)
    df_num = log_col(df_num)
    
    df_bool = new_features(df_features[['owner']])

    df_cat = preparation_cat(df_features).astype('str')

    df_all = pd.concat((df_num, df_bool, df_cat), axis=1)

    return df_all[model_features]

