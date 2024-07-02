import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from tqdm import tqdm
import os

class ProteinDataset(Dataset):
    def __init__(self, sequences, targets, tokenizer, max_length):
        self.sequences = sequences
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            sequence,
            max_length=self.max_length,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.float)
        }

class ProteinRegressor(nn.Module):
    def __init__(self, transformer_model, dropout_rate=0.3):
        super(ProteinRegressor, self).__init__()
        self.transformer = BertModel.from_pretrained(transformer_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.out = nn.Linear(self.transformer.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs[1]
        dropped_out = self.dropout(pooled_output)
        return self.out(dropped_out)

def train_model(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()
    losses = []
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

    return np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    losses = []
    predictions = []
    real_values = []

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

    return np.mean(losses), predictions, real_values

def main():

    # os.environ["CUDA_VISIBLE_DEVICES"] = "9"
    df = pd.read_csv('data_with_mut_seq.csv')

    sequences = df['mut_seq'].values
    targets = df['Brightness'].values
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
    max_length = 512  
    train_sequences, val_sequences, train_targets, val_targets = train_test_split(sequences, targets, test_size=0.1, random_state=42)

    train_dataset = ProteinDataset(train_sequences, train_targets, tokenizer, max_length)
    val_dataset = ProteinDataset(val_sequences, val_targets, tokenizer, max_length)

    batch_size = 16 
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    
    model = ProteinRegressor('Rostlab/prot_bert_bfd')
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * 5  # 5 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps // 3, gamma=0.1)
    loss_fn = nn.MSELoss().to(device)

    best_mse = float("inf")

    for epoch in range(5):
        print(f'Epoch {epoch + 1}/{5}')
        print('-' * 10)
        train_loss = train_model(model, train_data_loader, loss_fn, optimizer, device, scheduler, len(train_dataset))
        val_loss, predictions, real_values = eval_model(model, val_data_loader, loss_fn, device, len(val_dataset))
        
        val_mse = mean_squared_error(real_values, predictions)
        print(f'Train loss: {train_loss} Val loss: {val_loss}')
        print(f'Val MSE: {val_mse}')

        if val_mse < best_mse:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_mse = val_mse

    print("Training complete!")

if __name__ == "__main__":
    main()
