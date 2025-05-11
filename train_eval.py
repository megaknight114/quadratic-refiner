import torch
def train_one_epoch(model,dataloader,loss_fn,optimizer,device):
    model.train()
    total_loss=0.0
    total_samples=0
    for x_batch,y_batch in dataloader:
        x_batch,y_batch=x_batch.to(device),y_batch.to(device)
        optimizer.zero_grad()
        predictions=model(x_batch)
        loss=loss_fn(predictions,y_batch)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()*x_batch.size(0)
        total_samples+=x_batch.size(0)
    avg_loss=total_loss/total_samples
    print(total_samples)
    return avg_loss

def evaluate(model,dataloader,loss_fn,device):
    model.eval()
    total_loss=0.0
    total_samples=0
    with torch.no_grad():  
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            predictions = model(x_batch)
            loss = loss_fn(predictions, y_batch)
            total_loss += loss.item() * x_batch.size(0)
            total_samples += x_batch.size(0)
    print(total_samples)
    avg_loss = total_loss / total_samples
    return avg_loss