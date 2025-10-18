import os
import pandas as pd
import numpy as np


# 新增: 在顶层初始化全局缓存变量
_X_train, _y_train, _X_test, _test_ids = None, None, None, None

# 文件地址
DATA_DIR = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'data')

# 文件读取


def read_csv(file_name):
    file_path = os.path.join(DATA_DIR, file_name)
    return pd.read_csv(file_path)

# 数据预处理


def data_reshape(data):

    # 将含义为空的类别特征填充为'None'
    for col in ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
        data[col] = data[col].fillna('None')

    # 将含义为0的数值特征填充为0
    for col in ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
        data[col] = data[col].fillna(0)

    # 将MSSubClass转为str类型
    data['MSSubClass'] = data['MSSubClass'].astype(str)

    return data

# 主数据加载与处理


def _load_and_process_all_data():

    # 全局缓存
    global _X_train, _y_train, _X_test, _test_ids

    # 如果已缓存则直接返回
    if _X_train is not None:
        return

    # 原始数据加载
    train_df = read_csv('train.csv')
    test_df = read_csv('test.csv')

    # 分离数据内容和ID，并合并训练集与测试集
    n_train = len(train_df)
    y_train_processed = np.log1p(train_df['SalePrice'])
    test_ids_processed = test_df['Id']

    all_data = pd.concat((train_df.drop(['SalePrice', 'Id'], axis=1),
                          test_df.drop('Id', axis=1))).reset_index(drop=True)

    # 处理缺失值
    
    # 填充LotFrontage缺失值
    # 使用同一街区的中位数进行填充
    lotfrontage_median_map = train_df.groupby(
        'Neighborhood')['LotFrontage'].median()
    all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(lotfrontage_median_map.get(
            x.name, train_df['LotFrontage'].median()))
    )

    # 填充少量缺失的类别特征
    mode_cols = ['MSZoning', 'Electrical', 'KitchenQual',
                 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional']
    for col in mode_cols:
        mode_val = train_df[col].mode()[0]
        all_data[col] = all_data[col].fillna(mode_val)

    # 填充数值特征的缺失值为0
    all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
    
    # 数据预处理
    all_data = data_reshape(all_data)

    # 独热编码
    all_data = pd.get_dummies(all_data, drop_first=True)

    # 分离训练集和测试集
    X_train_processed = all_data[:n_train].copy()
    X_test_processed = all_data[n_train:].copy()

    # 确保训练集和测试集的列完全一致
    train_cols = X_train_processed.columns
    test_cols = X_test_processed.columns

    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test_processed[c] = 0

    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        X_train_processed[c] = 0

    # 标准化数据
    numeric_cols = X_train_processed.columns
    train_mean = X_train_processed[numeric_cols].mean()
    train_std = X_train_processed[numeric_cols].std()
    train_std[train_std == 0] = 1
    X_train_processed[numeric_cols] = (X_train_processed[numeric_cols] - train_mean) / train_std
    X_test_processed[numeric_cols] = (X_test_processed[numeric_cols] - train_mean) / train_std

    # 存入缓存
    _X_train, _y_train, _X_test, _test_ids = X_train_processed, y_train_processed, X_test_processed, test_ids_processed
    print("数据已加载并缓存")


def get_train_data():
    """
    X_train (pd.DataFrame): 训练特征
    y_train (pd.Series): 训练目标
    """
    _load_and_process_all_data()
    return _X_train, _y_train


def get_test_data():
    """
    X_test (pd.DataFrame): 测试特征
    test_ids (pd.Series): 测试集的ID
    """
    _load_and_process_all_data()
    return _X_test, _test_ids


if __name__ == '__main__':
    print("--- 数据 ---")
    X_train_data, y_train_data = get_train_data()

    print("\n训练数据 (X_train) :", X_train_data.shape)
    print("训练目标 (y_train) :", y_train_data.shape)
    print("训练数据 (X_train) :")
    print(X_train_data.head())

    X_test_data, test_ids_data = get_test_data()

    print("\n测试数据 (X_test) :", X_test_data.shape)
    print("测试ID (test_ids) :", test_ids_data.shape)
    print("测试数据 (X_test) :")
    print(X_test_data.head())
