import torch.nn as nn
from transformers import BertModel

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
