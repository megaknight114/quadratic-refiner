import torch
import matplotlib
matplotlib.use('Agg')  # 使用 Agg 后端以便保存图片而不是直接显示
import matplotlib.pyplot as plt
import os

# 1. 读取并打印文件内容
def load_and_print_data(filename='quadratic_data.pt'):
    # 加载数据
    X, y = torch.load(filename)
    
    # 打印数据基本信息
    print(f"Loaded data from {filename}")
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    print(f"First 5 entries of X:\n{X[:5]}")
    print(f"First 5 entries of y:\n{y[:5]}")

    return X, y

# 2. a, b, c, d 分布函数，画图（每个分布单独画）
def plot_data_distribution(X, y, save_dir='/home/xuzonghuan/quadratic-refiner/data_info'):
    # 确保保存文件夹存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存 a 的分布图
    plt.figure(figsize=(6, 4))
    plt.hist(X[:, 0].numpy(), bins=50, color='blue', alpha=0.7)
    plt.title('Distribution of a')
    plt.xlabel('a')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_dir, "a_distribution.png"))
    plt.close()

    # 保存 b 的分布图
    plt.figure(figsize=(6, 4))
    plt.hist(X[:, 1].numpy(), bins=50, color='green', alpha=0.7)
    plt.title('Distribution of b')
    plt.xlabel('b')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_dir, "b_distribution.png"))
    plt.close()

    # 保存 c 的分布图
    plt.figure(figsize=(6, 4))
    plt.hist(X[:, 2].numpy(), bins=50, color='red', alpha=0.7)
    plt.title('Distribution of c')
    plt.xlabel('c')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_dir, "c_distribution.png"))
    plt.close()

    # 保存解 y 的分布图
    plt.figure(figsize=(6, 4))
    plt.hist(y.numpy(), bins=50, color='purple', alpha=0.7)
    plt.title('Distribution of y (Max Root)')
    plt.xlabel('y')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(save_dir, "y_distribution.png"))
    plt.close()

# 运行数据加载和分布检测
if __name__ == "__main__":
    # 加载数据并打印基本信息
    X, y = load_and_print_data(filename='/home/xuzonghuan/quadratic-refiner/quadratic_data.pt')
    
    # 绘制数据分布并保存
    plot_data_distribution(X, y, save_dir='/home/xuzonghuan/quadratic-refiner/data_info')
