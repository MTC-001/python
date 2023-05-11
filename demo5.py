"""
@author: Inki
@contact: inki.yinji@qq.com
@version: Created in 2020 1209, last modified in 2020 1209.
"""

import numpy as np
import pandas as pd


def load_data(para_train_path, para_test_path, is_save=False):
    """
    Load datasets.
    """
    temp_train_data = pd.read_csv(para_train_path)
    temp_test_data = pd.read_csv(para_test_path)
    # 连接训练集和测试集的所有样本
    # 第一列是序号，训练集的最后一列是标签
    temp_all_features = pd.concat((temp_train_data.iloc[:, 1:-1], temp_test_data.iloc[:, 1:]))
    # 获取数值型数据的索引
    temp_numeric_features_idx = temp_all_features.dtypes[temp_all_features.dtypes != 'object'].index
    # 标准化数据
    temp_all_features[temp_numeric_features_idx] = temp_all_features[temp_numeric_features_idx].apply(
        lambda x: (x - x.mean()) / x.std())
    # 标准化后，可以使用0来代替缺失值
    temp_all_features = temp_all_features.fillna(0)
    # 离散值处理：
    # 例如某特征有两个不同的离散值，则该属性将被处理为二维：0 1 或者 1 0
    # 三中不同的离散值时，则对于 0 0 1、 0 1 0 以及 1 0 0，以此类推
    temp_all_features = pd.get_dummies(temp_all_features, dummy_na=True)
    # 数据划分
    temp_num_train = len(temp_train_data)
    ret_train_data = np.array(temp_all_features[:temp_num_train].values, dtype=np.float)
    ret_test_data = np.array(temp_all_features[temp_num_train:].values, dtype=np.float)
    ret_train_label = temp_train_data.values[:, -1]

    # 文件保存
    if is_save:
        temp_save_train_data = np.zeros((temp_num_train, len(ret_train_data[0]) + 1), dtype=float)
        temp_save_train_data[:, :-1] = ret_train_data
        temp_save_train_data[:, -1] = np.mat(ret_train_label)
        pd.DataFrame.to_csv(pd.DataFrame(temp_save_train_data), 'house_price_train.csv',
                            index=False, header=False, float_format='%.6f')
        pd.DataFrame.to_csv(pd.DataFrame(ret_test_data),  'house_price_test.csv',
                            index=False, header=False, float_format='%.6f')

    return ret_train_data, ret_train_label, ret_test_data


if __name__ == '__main__':

    train_path =  'train.csv'
    test_path = 'test.csv'
    load_data(train_path, test_path, True)
