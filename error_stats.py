import pandas as pd
import numpy as np

def compute_error_stats(csv_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)

    # 提取 target 和 prediction 两列
    target = df['target']
    prediction = df['prediction']

    # 计算误差
    error = prediction - target

    # 计算平均值和方差
    avg_error = error.mean()
    var_error = error.var()

    print(f"平均误差（prediction - target）: {avg_error}")
    print(f"误差的方差: {var_error}")

if __name__ == "__main__":
    # 修改为你的 CSV 文件路径
    for i in range(0,3):
        csv_file = f"/home/xuzonghuan/quadratic-refiner/train_data/train_data_with_preds{i}.csv"
        compute_error_stats(csv_file)
