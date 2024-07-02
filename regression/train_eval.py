import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, writer, epoch):
    model = model.train()
    losses = []
    step = 0

    for d in tqdm(data_loader, desc="Training"):
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs, targets.unsqueeze(1))
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        step += 1
        if step % 500 == 0:
            avg_loss = np.mean(losses[-500:])
            writer.add_scalar('Loss/train', avg_loss, epoch * len(data_loader) + step)
    
    return np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, writer, epoch):
    model = model.eval()
    losses = []
    predictions = []
    real_values = []
    step = 0

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, targets.unsqueeze(1))
            losses.append(loss.item())
            
            predictions.extend(outputs.cpu().numpy())
            real_values.extend(targets.cpu().numpy())

            step += 1
            if step % 500 == 0:
                avg_loss = np.mean(losses[-500:])
                pearson_corr = pearsonr(real_values[-500:], predictions[-500:])[0]
                writer.add_scalar('Loss/val', avg_loss, epoch * len(data_loader) + step)
                writer.add_scalar('Pearson/val', pearson_corr, epoch * len(data_loader) + step)

    avg_loss = np.mean(losses)
    pearson_corr = pearsonr(real_values, predictions)[0]
    writer.add_scalar('Loss/val_epoch', avg_loss, epoch)
    writer.add_scalar('Pearson/val_epoch', pearson_corr, epoch)
    
    return avg_loss, predictions, real_values
