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

# 2. 将超参数缩写并生成文件夹
def generate_experiment_name(model_type, hidden_layers, hidden_units, lr, batch_size, epoch):
    # 使用缩写生成实验名称
    model_name = model_type.__name__[:3]  # 使用模型名缩写（MLP, Res）
    L = f"L{hidden_layers}"
    U = f"U{hidden_units}"
    LR = f"LR{str(lr).replace('.', '')}"  # 去掉小数点来简化学习率名称
    BS = f"BS{batch_size}"
    E = f"E{epoch}"

    # 合成实验名称
    return f"{model_name}_{L}_{U}_{LR}_{BS}_{E}"

# 3. 执行大规模实验
def run_large_scale_experiment():
    # 设备：使用 CPU
    device = torch.device("cpu")
    print(f"Using device: {device}")

    # 设置超参数组合
    model_types = [MLP, ResidualMLP]  # 两种模型
    hidden_layer_options = [2,4]  # 隐藏层数
    hidden_unit_options = [128, 256]  # 每层的隐藏单元数
    learning_rates = [1e-3]  # 学习率
    batch_sizes = [128]  # 批量大小
    epochs = [100]  # 训练周期

    # 创建结果保存文件夹
    results_dir = '/home/xuzonghuan/quadratic-refiner/large_scale_baseline_experiment'
    os.makedirs(results_dir, exist_ok=True)

    # 加载数据
    X, y = load_data_from_file()

    # 创建 TensorDataset
    dataset = TensorDataset(X, y)
    all_results = []
    # 划分数据集：80% 训练集，20% 验证集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建 DataLoader
    # 在实验中尝试不同的超参数组合
    for model_type in model_types:
        for hidden_layers in hidden_layer_options:
            for hidden_units in hidden_unit_options:
                for lr in learning_rates:
                    for batch_size in batch_sizes:
                        for epoch in epochs:
                            # 生成实验名称
                            experiment_name = generate_experiment_name(model_type, hidden_layers, hidden_units, lr, batch_size, epoch)
                            print(f"Running experiment: {experiment_name}")

                            # 训练与评估
                            model = model_type(hidden_dim=hidden_units, hidden_layers=hidden_layers).to(device)
                            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                            # 训练过程
                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

                            # 结果文件夹
                            experiment_dir = os.path.join(results_dir, experiment_name)
                            os.makedirs(experiment_dir, exist_ok=True)

                            # 记录训练损失和验证损失
                            epoch_losses = []
                            val_losses = []

                            # 训练模型并保存损失数据
                            for ep in range(epoch):  # 训练 epoch 次
                                train_loss = train_one_epoch(model, train_loader, MSE_loss, optimizer, device)
                                val_loss = evaluate(model, val_loader, MSE_loss, device)

                                epoch_losses.append(train_loss)
                                val_losses.append(val_loss)

                                print(f"Epoch {ep + 1}: Train Loss = {train_loss:.6f}, Validation Loss = {val_loss:.6f}")

                                if ep == epoch - 1:
                                    all_results.append({
                                        'experiment_name': experiment_name,
                                        'train_loss': train_loss,
                                        'val_loss': val_loss
                                    })
                            # 保存损失列表到 CSV 文件
                            loss_data = {'epoch': list(range(1, epoch + 1)), 'train_loss': epoch_losses, 'val_loss': val_losses}
                            loss_df = pd.DataFrame(loss_data)
                            loss_df.to_csv(os.path.join(experiment_dir, "losses.csv"), index=False)
                            print(f"Loss data saved to {os.path.join(experiment_dir, 'losses.csv')}")

                            # 绘制损失曲线图并保存
                            plt.figure(figsize=(8, 6))
                            plt.plot(range(1, epoch + 1), epoch_losses, label='Training Loss', color='blue')
                            plt.plot(range(1, epoch + 1), val_losses, label='Validation Loss', color='orange')
                            plt.title(f'{experiment_name} Loss Curve')
                            plt.xlabel('Epoch')
                            plt.ylabel('Loss')
                            plt.legend()
                            plt.grid(True)
                            plt.savefig(os.path.join(experiment_dir, "loss_curve.png"))
                            plt.close()
                            print(f"Loss curve saved to {os.path.join(experiment_dir, 'loss_curve.png')}")

    summary_data = []
    for result in all_results:
        summary_data.append(result)

    # 创建一个 DataFrame 并保存汇总数据
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(results_dir, 'experiment_summary.csv'), index=False)
    print(f"All experiments completed. Summary saved to {os.path.join(results_dir, 'experiment_summary.csv')}")

# 5. 主程序入口
if __name__ == "__main__":
    run_large_scale_experiment()