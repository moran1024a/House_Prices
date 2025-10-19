import os
import pandas as pd
import numpy as np

# 初始化全局缓存变量
_X_train, _y_train, _X_test, _test_ids = None, None, None, None

# 文件目录
DATA_DIR = os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'data')


# csv文件读取模块
def _read_csv(file_name: str):
    """
    从DATA_DIR读取指定的csv文件。
    """
    file_path = os.path.join(DATA_DIR, file_name)
    return pd.read_csv(file_path)


# 数据预处理模块
def _handle_missing_values(data: pd.DataFrame, train_df: pd.DataFrame):
    """
    集中处理数据集中的各种缺失值。
    """
    # LotFrontage处理：缺失值充填为每个街区（Neighborhood）房价的中位数
    # 为什么中位数：中位数对极端值不敏感，更具有鲁棒性
    # 对训练集分组，计算每个街道的LotFrontage中位数
    lotfrontage_median_map = train_df.groupby(
        'Neighborhood')['LotFrontage'].median()
    # 填充缺失值（使用transform对每个分组应用填充，使用lambda构建匿名函数构成递归）
    data['LotFrontage'] = data.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(lotfrontage_median_map.get(
            x.name, train_df['LotFrontage'].median()))
    )

    # 分类变量特征处理：充填为数据集中对应项的众数
    # 为什么众数：众数代表最常见的类别，适合填充分类变量的缺失值
    mode_cols = ['MSZoning', 'Electrical', 'KitchenQual',
                 'Exterior1st', 'Exterior2nd', 'SaleType', 'Functional']
    for col in mode_cols:
        mode_val = train_df[col].mode()[0]
        data[col] = data[col].fillna(mode_val)

    # MasVnrArea处理：缺失值为0
    # 为什么是0：此特征代表石材面积，缺失即没有石材
    data['MasVnrArea'] = data['MasVnrArea'].fillna(0)

    # 含义为空的特征处理：填充为'None'
    for col in ['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu',
                'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']:
        data[col] = data[col].fillna('None')

    # 含义为0的数值特征：填充为0
    for col in ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1',
                'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']:
        data[col] = data[col].fillna(0)

    return data

# 数据转换与编码模块


def _transform_and_encode_features(data: pd.DataFrame):
    """
    进行数据类型转换和独热编码。
    """
    # 处理MSSubClass：转换为str，方便独热编码
    # 为什么转换为str：MSSubClass是类别特征，转换为字符串可以避免数值特征的误解
    data['MSSubClass'] = data['MSSubClass'].astype(str)

    # 对类别特征进行独热编码，drop_first=True 可以减少一列，避免多重共线性
    # 独热编码：将类别变量转换为二进制向量，适合机器学习模型处理；用人话就是把分类变量变成0和1的二元特征。此方法会增加特征数量，但能更好地表示类别信息。
    # 使用drop_first=True：去除转换后第一个特征，避免多重共线性问题
    data = pd.get_dummies(data, drop_first=True)

    return data

# 对齐训练集和测试集的列
def _align_columns(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    确保训练集和测试集的列完全一致，即特征顺序相同。
    """
    train_cols = X_train.columns
    test_cols = X_test.columns

    # 在测试集中添加训练集中存在但测试集中缺失的列
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        X_test[c] = 0

    # 在训练集中添加测试集中存在但训练集中缺失的列（罕见情况）
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        X_train[c] = 0

    # 保证测试集的列顺序与训练集一致
    X_test = X_test[train_cols]

    return X_train, X_test

# 数据标准化
def _scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    使用均值和标准差对数据进行标准化。
    """
    numeric_cols = X_train.columns
    train_mean = X_train[numeric_cols].mean()
    train_std = X_train[numeric_cols].std()

    # 防止除以零
    train_std[train_std == 0] = 1

    X_train[numeric_cols] = (X_train[numeric_cols] - train_mean) / train_std
    X_test[numeric_cols] = (X_test[numeric_cols] - train_mean) / train_std

    return X_train, X_test


# 主处理模块
def _load_and_process_all_data():
    """
    主处理模块，利用全局缓存避免重复加载。
    """
    global _X_train, _y_train, _X_test, _test_ids

    # 如果数据已经加载，直接返回
    if _X_train is not None:
        return

    # 加载原始数据
    train_df = _read_csv('train.csv')
    test_df = _read_csv('test.csv')

    # 分离ID，并合并特征集（将训练集和测试集合并，避免分别处理不一致）
    n_train = len(train_df)
    y_train_processed = np.log1p(train_df['SalePrice'])
    test_ids_processed = test_df['Id']

    all_data = pd.concat((train_df.drop(['SalePrice', 'Id'], axis=1),
                          test_df.drop('Id', axis=1))).reset_index(drop=True)

    # 数据预处理
    all_data = _handle_missing_values(all_data, train_df)
    all_data = _transform_and_encode_features(all_data)

    # 重新分离训练集和测试集
    X_train_processed = all_data[:n_train].copy()
    X_test_processed = all_data[n_train:].copy()

    # 对齐训练集和测试集的列
    X_train_processed, X_test_processed = _align_columns(
        X_train_processed, X_test_processed)

    # 标准化数据
    X_train_processed, X_test_processed = _scale_features(
        X_train_processed, X_test_processed)

    # 加载数据
    _X_train, _y_train, _X_test, _test_ids = X_train_processed, y_train_processed, X_test_processed, test_ids_processed
    print("数据已加载并缓存")


# 外部接口
def get_train_data():
    """
    获取处理好的训练数据。
    返回:
        X_train (pd.DataFrame): 训练特征
        y_train (pd.Series): 训练目标
    """
    _load_and_process_all_data()
    return _X_train, _y_train


def get_test_data():
    """
    获取处理好的测试数据。
    返回:
        X_test (pd.DataFrame): 测试特征
        test_ids (pd.Series): 测试集的ID
    """
    _load_and_process_all_data()
    return _X_test, _test_ids


# 测试接口
if __name__ == '__main__':
    print("--- 正在加载和处理数据 ---")
    X_train_data, y_train_data = get_train_data()

    print("\n训练数据 (X_train) shape:", X_train_data.shape)
    print("训练目标 (y_train) shape:", y_train_data.shape)
    print("训练数据 (X_train) head:")
    print(X_train_data.head())

    X_test_data, test_ids_data = get_test_data()

    print("\n测试数据 (X_test) shape:", X_test_data.shape)
    print("测试ID (test_ids) shape:", test_ids_data.shape)
    print("测试数据 (X_test) head:")
    print(X_test_data.head())
