import torch
def MSE_loss(pred,target):
    return ((pred-target)**2).mean()

def relative_error_loss(pred,target):
    eps=1e-100
    relative_err=((pred-target).abs()/target.abs()+eps).mean()
    return relative_err



    

def tolerance_loss(pred, target, prev_pred, time, 
                   tolerance=0.5,   # 误差缩小阈值（0.5 表示需要至少减半）
                   scale=1.0,       # 时间衰减系数
                   eps=1e-100):       # 防 0 除
    """
    pred, target, prev_pred: Tensor 同形状
    time: 当前迭代步 (int 或 float)，用于衰减
    返回: (mean loss, 合格样本数)
    """

    err       = pred - target
    prev_err  = prev_pred - target
    abs_ratio = torch.abs(err) / (torch.abs(prev_err) + eps)

    # 判定是否改进到位
    good_mask = abs_ratio <= tolerance
    bad_mask  = ~good_mask          # 其余情况都要罚

    loss      = torch.zeros_like(err)

    # 对 bad 部分罚 “超额误差”：|err| - tolerance*|prev_err|
    excess = torch.abs(err[bad_mask]) - tolerance * torch.abs(prev_err[bad_mask])
    loss[bad_mask] = excess * (scale ** time)

    return loss.mean(), good_mask.sum().item()