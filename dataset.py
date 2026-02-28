import pandas as pd
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


def read_excel_data(file_path, use_header=True):
    """
    读取 Excel 文件，并将第一列作为序号，最后一列作为输出，中间列作为特征取值。

    参数：
        file_path (str): Excel 文件路径。
        use_header (bool): 是否使用第一行作为表头。如果为 False，则所有行均视为数据。

    返回：
        tuple: (index, features, output)
            - index: 第一列的数据（序号）
            - features: 中间列的数据（特征取值）
            - output: 最后一列的数据（输出）
    """
    # 根据 use_header 参数决定是否读取表头
    if use_header:#第一行会被忽略掉
        df = pd.read_excel(file_path)
    else:#第一行会作为普通数据
        df = pd.read_excel(file_path, header=None)

    # 检查列数是否至少有三列：序号、至少一个特征、输出
    if df.shape[1] < 3:
        raise ValueError("Excel 文件中至少需要三列：第一列为序号，最后一列为输出，中间列为特征取值。")

    # 提取第一列（序号）、中间列（特征）和最后一列（输出）
    index = df.iloc[:, 0]
    features = df.iloc[:, 1:-1]
    output = df.iloc[:, -1]

    return index, features, output

def read_excel_data2(file_path1,file_path2, use_header=True):
    """
    读取 Excel 文件，并将第一列作为序号，最后一列作为输出，中间列作为特征取值。

    参数：
        file_path (str): Excel 文件路径。
        use_header (bool): 是否使用第一行作为表头。如果为 False，则所有行均视为数据。

    返回：
        tuple: (index, features, output)
            - index: 第一列的数据（序号）
            - features: 中间列的数据（特征取值）
            - output: 最后一列的数据（输出）
    """
    # 根据 use_header 参数决定是否读取表头

    df1 = pd.read_excel(file_path1).astype(float)

    df2 = pd.read_excel(file_path2, header=None).astype(float)


    # 提取第一列（序号）、中间列（特征）和最后一列（输出）
    index1 = df1.iloc[:, 0]
    features1 = df1.iloc[:, 1:-1]
    output1 = df1.iloc[:, -1]

    # 提取第一列（序号）、中间列（特征）和最后一列（输出）
    index2 = df2.iloc[:, 0]
    features2 = df2.iloc[:, 1:-1]
    output2 = df2.iloc[:, -1]

    import pdb;pdb.set_trace()

    merged_features = pd.concat([features1, features2], ignore_index=True, sort=False)
    merged_index = pd.concat([index1, index2], ignore_index=True, sort=False)
    merged_output = pd.concat([output1, output2], ignore_index=True, sort=False)

    return merged_index,merged_features, merged_output

# 自定义 Dataset，假设输入的 features 已经是 numpy 数组格式
class MyDataset(Dataset):
    def __init__(self, features, labels):
        """
        参数：
          features: numpy 数组，形状为 (n_samples, n_features)
          labels: numpy 数组，形状为 (n_samples,)
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # 将特征和标签转换为 torch.tensor
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return {'feature': feature, 'label': label}

def prepare_dataloaders(features, labels, test_featuers, test_labels, batch_size=32, test_size=0.2, random_state=42,
                         scaled=True, normalize_output=False):

    """
    根据特征和标签数据，进行归一化，并返回 DataLoader 对象。

    参数：
      features: pandas DataFrame，原始特征数据
      labels: pandas Series，原始标签数据
      batch_size: 每个批次的样本数
      test_size: 当 split_data 为 True 时，验证集比例（默认 0.2，即 20% 验证集）
      random_state: 随机数种子，保证划分可复现
      split_data: bool，是否对数据进行划分（训练/验证），如果为 False，则不进行划分（通常用于测试集）
      normalize_output: bool，是否对标签也进行归一化（只适用于回归任务）

    返回：
      当 split_data 为 True 时，返回 (train_loader, val_loader, feature_scaler, label_scaler)
      当 split_data 为 False 时，返回 (loader, feature_scaler, label_scaler)

      如果 normalize_output 为 False，则 label_scaler 返回 None
    """
    if scaled:
        # 对输入特征归一化使用 MinMaxScaler
        feature_scaler = MinMaxScaler()
        # 如果需要归一化标签，则创建 label_scaler
        label_scaler =MinMaxScaler() if normalize_output else None


    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    # import pdb;pdb.set_trace()
    if scaled:#如果归一化
        # 对训练集特征进行拟合并转换，再对验证集和测试集进行转换
        X_train_scaled = feature_scaler.fit_transform(X_train)
        X_val_scaled   = feature_scaler.transform(X_val)
        X_test_scaled= feature_scaler.transform(test_featuers)

        # 如果归一化输出，则对标签进行归一化
        if normalize_output:
            # 注意：MinMaxScaler 要求输入为二维数组，因此需要 reshape
            y_train_reshaped = y_train.values.reshape(-1, 1)
            y_val_reshaped   = y_val.values.reshape(-1, 1)
            y_test_reshaped   = test_labels.values.reshape(-1, 1)

            y_train_scaled = label_scaler.fit_transform(y_train_reshaped).flatten()
            y_val_scaled   = label_scaler.transform(y_val_reshaped).flatten()
            y_test_scaled   = label_scaler.transform(y_test_reshaped).flatten()
        else:
            y_train_scaled = y_train.values
            y_val_scaled   = y_val.values
    if scaled:
        # 创建 Dataset 对象
        train_dataset = MyDataset(X_train_scaled, y_train_scaled)
        val_dataset   = MyDataset(X_val_scaled, y_val_scaled)
        test_dataset   = MyDataset(X_test_scaled, y_test_scaled)
    else:
        train_dataset = MyDataset(X_train.to_numpy(), y_train.values)
        val_dataset   = MyDataset(X_val.to_numpy(), y_val.values)
        test_dataset   = MyDataset(test_featuers.to_numpy(), test_labels.values)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader,test_loader



if __name__ == "__main__":
    import os
    # 请修改为你的 Excel 文件路径
    current_file_path = os.path.abspath(__file__)  # 获取当前脚本的绝对路径
    current_dir = os.path.dirname(current_file_path) #当前脚本的父目录
    file_path = f"{current_dir}\datas\数据模式-光谱打分-训练和测试.xlsx"
    test_path= f"{current_dir}\datas\数据模式-光谱打分-预留的预测集.xlsx"
    # 修改 use_header 参数决定是否将第一行作为表头 True/False
    use_header = True


    index, features, output = read_excel_data(file_path, use_header=True)
    train_loader=prepare_dataloaders(features, output, batch_size=1, test_size=0.2, random_state=42,
                    split_data=False, normalize_output=True)

    print(len(train_loader))

