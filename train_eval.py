import torch
def train_one_epoch(model,dataloader,loss_fn,optimizer,device,time):
    model.train()
    total_loss=0.0
    total_samples=0
    correct_samples = 0
    for x_batch,y_batch,z_batch in dataloader:
        x_batch,y_batch,z_batch=x_batch.to(device),y_batch.to(device),z_batch.to(device)
        optimizer.zero_grad()
        predictions=model(torch.cat([x_batch, z_batch], dim=1))
        loss,temp_correct_samples=loss_fn(predictions,y_batch,z_batch,time)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()*x_batch.size(0)
        total_samples+=x_batch.size(0)
        correct_samples += temp_correct_samples
    avg_loss=total_loss/total_samples
    accuracy = correct_samples / total_samples
    print(total_samples)
    return avg_loss,accuracy

def evaluate(model,dataloader,loss_fn,device,time):
    model.eval()
    total_loss=0.0
    total_samples=0
    correct_samples = 0 
    with torch.no_grad():  
        for x_batch,y_batch,z_batch in dataloader:
            x_batch,y_batch,z_batch=x_batch.to(device),y_batch.to(device),z_batch.to(device)
            predictions=model(torch.cat([x_batch, z_batch], dim=1))
            loss,temp_correct_samples=loss_fn(predictions,y_batch,z_batch,time)
            total_loss+=loss.item()*x_batch.size(0)
            total_samples+=x_batch.size(0)
            correct_samples += temp_correct_samples
    print(correct_samples)
    avg_loss = total_loss / total_samples
    accuracy = correct_samples / total_samples

    return avg_loss,accuracy