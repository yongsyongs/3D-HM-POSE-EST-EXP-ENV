import torch
import torch.nn as nn

class BiDirectionalLSTM(nn.Module):
    # size : joints_num * dim
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.25):
        super(BiDirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.selu = nn.SELU()

    def to(self, device):
        self.device = device
        super().to(device)

    def forward(self, x):
        x = x.view(x.shape[:2] + (-1,))
        # Set initial states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)  # 2 for bidirection
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(self.device)


        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        idc = out.shape[-1] // 2
        out = self.selu(self.drop(out))
        out = (out[:, :, :idc] + out[: , :, idc:]) / 2
        out = out.view(out.shape[:2] + (-1, 2))

        # Decode the hidden state of the last time step
        return out