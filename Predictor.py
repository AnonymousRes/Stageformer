import torch

from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn import LayerNorm
import torch.nn.functional as F


class DPPredictor(nn.Module):
    def __init__(self, core_model, label_dim):
        super(DPPredictor, self).__init__()
        self.core_model = core_model
        self.hidden_size = core_model.output_dim
        self.label_dim = label_dim
        self.linear = nn.Linear(self.hidden_size, self.label_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        final_output, _ = self.core_model(inputs)
        final_output = self.linear(final_output)
        final_output = self.dropout(final_output)
        final_output = self.sigmoid(final_output)
        return final_output
