import torch
import torch.nn as nn

class MagNet(nn.Module):
    def __init__(self, input_channels=3, input_length=6000):
        super(MagNet, self).__init__()
        
        # CNN Layers
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Calculate LSTM input size
        # Length reduces by factor of 2^4 = 16
        # 6000 / 16 = 375
        self.lstm_input_size = 256
        
        # Bi-LSTM
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        
        # Fully Connected
        self.fc = nn.Sequential(
            nn.Linear(128 * 2, 64), # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1) # Output magnitude (scalar)
        )

    def forward(self, x):
        # x: (Batch, 3, 6000)
        
        # CNN
        x = self.cnn(x) # (Batch, 256, 375)
        
        # Prepare for LSTM: (Batch, Seq_Len, Features)
        x = x.permute(0, 2, 1) # (Batch, 375, 256)
        
        # LSTM
        # output: (Batch, Seq_Len, Num_Directions * Hidden_Size)
        # h_n: (Num_Layers * Num_Directions, Batch, Hidden_Size)
        _, (h_n, _) = self.lstm(x)
        
        # Concatenate forward and backward hidden states
        # h_n shape: (2, Batch, 128)
        x = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1) # (Batch, 256)
        
        # FC
        x = self.fc(x)
        return x
