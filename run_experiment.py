import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from models import MLP, ResidualMLP
from losses import MSE_loss
from train_eval import train_one_epoch, evaluate
from torch.utils.data import DataLoader, TensorDataset, random_split

# 1. 读取数据
def load_data_from_file(filename='/home/xuzonghuan/quadratic-refiner/quadratic_data.pt'):
    X, y = torch.load(filename)
    print(f"Loaded data from {filename}")
    return X, y

# 2. 设置单次实验，并记录实验结果
def run_test_experiment(model_type=MLP, hidden_layers=3, hidden_units=128, loss_fn=MSE_loss, lr=1e-3, batch_size=5000,epoch=10000, experiment_name="test_experiment"):
    # 设备：使用 CPU
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 加载数据
    X, y = load_data_from_file()

    # 创建 TensorDataset
    dataset = TensorDataset(X, y)

    # 划分数据集：80% 训练集，20% 验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型和优化器
    model = model_type(hidden_dim=hidden_units, hidden_layers=hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 创建文件夹来保存实验结果
    save_dir = f'/home/xuzonghuan/quadratic-refiner/{experiment_name}'
    os.makedirs(save_dir, exist_ok=True)

    # 记录训练损失和验证损失
    epoch_losses = []
    val_losses = []

    # 训练过程
    for epoch in range(epoch):  
        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)

        epoch_losses.append(train_loss)
        val_losses.append(val_loss)

        # 打印进度
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Validation Loss = {val_loss:.6f}")

    # 保存损失列表到 CSV 文件
    loss_data = {'epoch': list(range(1,epoch+2,1)), 'train_loss': epoch_losses, 'val_loss': val_losses}
    loss_df = pd.DataFrame(loss_data)
    loss_df.to_csv(os.path.join(save_dir, "losses.csv"), index=False)
    print(f"Loss data saved to {os.path.join(save_dir, 'losses.csv')}")

    # 绘制损失曲线图并保存
    plt.figure(figsize=(8, 6))
    plt.plot(range(1,epoch+2,1), epoch_losses, label='Training Loss', color='blue')
    plt.plot(range(1,epoch+2,1), val_losses, label='Validation Loss', color='orange')
    plt.title(f'{experiment_name} Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()
    print(f"Loss curve saved to {os.path.join(save_dir, 'loss_curve.png')}")

# 3. 主程序入口
if __name__ == "__main__":
    # 设置实验超参数
    model_type = MLP  # 可以更改为 ResidualMLP
    hidden_layers = 3
    hidden_units = 128
    loss_fn = MSE_loss
    lr = 1e-3
    batch_size = 64
    epoch=1000
    experiment_name = "test_experiment"  # 可自定义实验名

    # 运行测试实验
    run_test_experiment(model_type, hidden_layers, hidden_units, loss_fn, lr, batch_size,epoch, experiment_name)
