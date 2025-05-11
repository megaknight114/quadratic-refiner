import torch
def MSE_loss(pred,target):
    return ((pred-target)**2).mean()

def relative_error_loss(pred,target):
    eps=1e-100
    relative_err=((pred-target).abs()/target.abs()+eps).mean()
    return relative_err

