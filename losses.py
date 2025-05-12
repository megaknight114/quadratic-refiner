import torch
def MSE_loss(pred,target):
    return ((pred-target)**2).mean()

def relative_error_loss(pred,target):
    eps=1e-100
    relative_err=((pred-target).abs()/target.abs()+eps).mean()
    return relative_err

def tolerance_loss(pred, target, prev_pred,time):
    # 计算当前误差和前一个预测的误差
    scale=1
    err = pred - target
    prev_err = prev_pred - target
    ratio = err / (prev_err)

    loss = torch.zeros_like(err)

    # 1. 合理改进：方向一致，误差变小
    mask_small_error = (ratio >= 0.8) & (ratio < 1)
    loss[mask_small_error] = 0

    # 2. 改得太少或反弹：方向一致但误差变大
    mask_large_error = (ratio >= 1)
    scaled_error = err[mask_large_error] - prev_err[mask_large_error]
    loss[mask_large_error] = (torch.abs(scaled_error) ) * (scale ** time)

    # 3. 方向错误
    mask_reverse_error = (ratio < 0.8)
    scaled_error = err[mask_reverse_error]
    loss[mask_reverse_error] = (torch.abs(scaled_error) ) * (scale ** time)
    # 返回损失的均值
    return loss.mean(),((ratio >= 0) & (ratio < 1)).sum().item()