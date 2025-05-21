import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from models import MLP, ResidualMLP
from losses import tolerance_loss
from train_eval import train_one_epoch, evaluate
from torch.utils.data import DataLoader, TensorDataset, random_split

# 1. 读取数据
def load_data_from_file(filename='/home/xuzonghuan/quadratic-refiner/quadratic_data.pt'):
    data = torch.load(filename)

    if isinstance(data, tuple) and len(data) == 3:
        X, y, z = data
    elif isinstance(data, tuple) and len(data) == 2:
        X, y = data
        z = torch.zeros_like(y)  # 与 y 同形状（通常是 (N, 1)）
        print("No prediction column found in data. Initialized z as zeros.")
    else:
        raise ValueError("Unsupported data format in .pt file")

    print(f"Loaded data from {filename}")
    return X, y, z

def load_data_from_pred(filename='/home/xuzonghuan/quadratic-refiner/train_data/train_data_with_preds.csv'):
    df = pd.read_csv(filename)
    x_columns = ['x1', 'x2', 'x3']
    y_column  = ['target']
    z_columns = ['prediction']

    X = torch.tensor(df[x_columns].values, dtype=torch.float32)
    z = torch.tensor(df[z_columns].values, dtype=torch.float32)
    y = torch.tensor(df[y_column].values,  dtype=torch.float32)

    print(f"Loaded data from {filename}")
    return X, y, z

# 2. 设置单次实验，并记录实验结果
def run_test_experiment(model_type=MLP, hidden_layers=3, hidden_units=128, loss_fn=tolerance_loss, lr=1e-3, batch_size=5000,epoch=10000, experiment_name="test_experiment",origin=0):
    # 设备：使用 CPU
    device = torch.device("cpu")
    print(f"Using device: {device}")
    # 加载数据
    if origin==0:
        X, y,z = load_data_from_file()
    if origin>0:
        X, y,z = load_data_from_pred(f'/home/xuzonghuan/quadratic-refiner/train_data/train_data_with_preds{origin-1}.csv')
    # 创建 TensorDataset
    dataset = TensorDataset(X, y,z)
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
    acc_epoch_losses = []
    val_acc_losses = []
    for epoch in range(epoch):
    # 训练和验证
        train_loss, train_acc = train_one_epoch(model, train_loader, loss_fn, optimizer, device,origin+1)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn, device,origin+1)

        # 记录损失和准确率
        epoch_losses.append(train_loss)
        val_losses.append(val_loss)
        acc_epoch_losses.append(train_acc)
        val_acc_losses.append(val_acc)

        # 打印进度
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Train Accuracy = {train_acc:.6f}, "
            f"Validation Loss = {val_loss:.6f}, Validation Accuracy = {val_acc:.6f}")

    # 保存损失和准确率列表到 CSV 文件
    loss_data = {'epoch': list(range(1, epoch+2)), 'train_loss': epoch_losses, 'val_loss': val_losses,
                'train_accuracy': acc_epoch_losses, 'val_accuracy': val_acc_losses}
    loss_df = pd.DataFrame(loss_data)
    loss_df.to_csv(os.path.join(save_dir, "loss_and_accuracy.csv"), index=False)
    print(f"Loss and accuracy data saved to {os.path.join(save_dir, 'loss_and_accuracy.csv')}")

    model.eval()                          # 关掉 dropout / BN 的训练模式
    with torch.no_grad():
        input_tensor = torch.cat([X.to(device), z.to(device)], dim=1)
        preds = model(input_tensor).cpu().squeeze()  # 保证顺序与 X 完全一致

    # 组合成 DataFrame（列名可按需要改）
    feature_cols = [f"x{i+1}" for i in range(X.shape[1])]     # e.g. x1, x2, x3
    df = pd.DataFrame(X.numpy(), columns=feature_cols)
    df["target"]      = y.numpy().squeeze()
    df["prediction"]  = preds.numpy()

    # 保存到指定文件夹
    train_data_dir = "/home/xuzonghuan/quadratic-refiner/train_data"
    os.makedirs(train_data_dir, exist_ok=True)
    out_path = os.path.join(train_data_dir, f"train_data_with_preds{origin}.csv")
    df.to_csv(out_path, index=False)
    print(f"Predictions for final epoch saved to {out_path}")

    # 绘制损失曲线图并保存
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epoch+2), epoch_losses, label='Training Loss', color='blue')
    plt.plot(range(1, epoch+2), val_losses, label='Validation Loss', color='orange')
    plt.title(f'{experiment_name} Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'loss_curve{origin}.png'))
    plt.close()
    print(f"Loss curve saved to {os.path.join(save_dir, f'loss_curve{origin+1}.png')}")

    # 绘制准确率曲线图并保存
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epoch+2), acc_epoch_losses, label='Training Accuracy', color='blue')
    plt.plot(range(1, epoch+2), val_acc_losses, label='Validation Accuracy', color='orange')
    plt.title(f'{experiment_name} Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"accuracy_curve{origin}.png"))
    plt.close()
    print(f"Accuracy curve saved to {os.path.join(save_dir, f'accuracy_curve{origin+1}.png')}")

# 3. 主程序入口
if __name__ == "__main__":
    # 设置实验超参数
    model_type = MLP  # 可以更改为 ResidualMLP
    hidden_layers = 5
    hidden_units = 128
    loss_fn = tolerance_loss
    lr = 1e-3
    batch_size = 500
    epoch=100
    experiment_name = "test_experiment"  # 可自定义实验名

    # 运行测试实验

    for i in range(0,10,1):
        print(i)
        run_test_experiment(model_type, hidden_layers, hidden_units, loss_fn, lr, batch_size,epoch, experiment_name,i)
