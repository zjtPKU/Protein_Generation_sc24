import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

from dataset import ProteinDataset
from model import ProteinRegressor
from train_eval import train_model, eval_model

def main():
    # 超参数配置
    config = {
        "cuda_device": "7",
        "data_file": "data_with_mut_seq.csv",
        "transformer_model": "Rostlab/prot_bert_bfd",
        "do_lower_case": False,
        "max_length": 512,
        "test_size": 0.1,
        "random_state": 42,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "num_epochs": 5,
        "step_size": 1,  
        "gamma": 0.1,
        "dropout_rate": 0.3,
        "log_dir": "runs"
    }

    os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_device"]
    writer = SummaryWriter(log_dir=config["log_dir"])  # 初始化 TensorBoard

    df = pd.read_csv(config["data_file"])

    sequences = df['mut_seq'].values
    targets = df['Brightness'].values
    tokenizer = BertTokenizer.from_pretrained(config["transformer_model"], do_lower_case=config["do_lower_case"])

    train_sequences, val_sequences, train_targets, val_targets = train_test_split(
        sequences, targets, test_size=config["test_size"], random_state=config["random_state"])

    train_dataset = ProteinDataset(train_sequences, train_targets, tokenizer, config["max_length"])
    val_dataset = ProteinDataset(val_sequences, val_targets, tokenizer, config["max_length"])

    train_data_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ProteinRegressor(config["transformer_model"], dropout_rate=config["dropout_rate"])
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    total_steps = len(train_data_loader) * config["num_epochs"]
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config["step_size"], gamma=config["gamma"])
    loss_fn = torch.nn.MSELoss().to(device)

    best_mse = float("inf")

    for epoch in range(config["num_epochs"]):
        print(f'Epoch {epoch + 1}/{config["num_epochs"]}')
        print('-' * 10)
        train_loss = train_model(model, train_data_loader, loss_fn, optimizer, device, scheduler, writer, epoch)
        val_loss, predictions, real_values = eval_model(model, val_data_loader, loss_fn, device, writer, epoch)

        val_mse = mean_squared_error(real_values, predictions)
        print(f'Train loss: {train_loss} Val loss: {val_loss}')
        print(f'Val MSE: {val_mse}')

        if val_mse < best_mse:
            torch.save(model.state_dict(), 'best_model_state_1e-3.bin')
            best_mse = val_mse

    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    main()
