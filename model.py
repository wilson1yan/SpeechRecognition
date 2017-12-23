import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        batch_size, n_seq, seq_len = x.size() # batch_size x 50 x 320
        hidden_state = Variable(torch.zeros(batch_size, 1, self.hidden_size))
        hidden_state = self.gru(x, hidden_state)[1]
        return self.classifier(hidden_state.squeeze(0))          
