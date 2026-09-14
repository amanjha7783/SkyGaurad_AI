import torch
import torch.nn as nn
import numpy as np
import os

class LSTMAutoencoderNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.0):
        super(LSTMAutoencoderNet, self).__init__()
        self.seq_len = None # Set dynamically during forward pass
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.output_layer = nn.Linear(hidden_dim, input_dim)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        self.seq_len = x.shape[1]
        
        # Encode
        _, (hidden, _) = self.encoder_lstm(x)
        
        # hidden shape: (num_layers, batch_size, hidden_dim)
        # Take the last layer's hidden state
        last_hidden = hidden[-1] # (batch_size, hidden_dim)
        
        # Repeat the hidden state for seq_len times to feed to decoder
        # (batch_size, seq_len, hidden_dim)
        decoder_input = last_hidden.unsqueeze(1).repeat(1, self.seq_len, 1)
        
        # Decode
        decoder_out, _ = self.decoder_lstm(decoder_input)
        
        # Reconstruct
        out = self.output_layer(decoder_out)
        return out

class LSTMAnomalyDetector:
    def __init__(self, input_dim, hidden_dim=32, num_layers=1, dropout=0.1, lr=1e-3, device='cpu'):
        self.device = device
        self.model = LSTMAutoencoderNet(input_dim, hidden_dim, num_layers, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='none') # Compute loss per element
        
    def calculate_reconstruction_error(self, X_tensor):
        """Calculates MSE across the features for each timestep, then averages over the sequence."""
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
            loss = self.criterion(reconstructed, X_tensor)
            # Average over features and sequence length
            # Shape is (batch_size, seq_len, input_dim)
            mse_per_sample = loss.mean(dim=[1, 2])
        return mse_per_sample.cpu().numpy(), reconstructed.cpu().numpy()
        
    def train_step(self, dataloader):
        self.model.train()
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(self.device)
            self.optimizer.zero_grad()
            reconstructed = self.model(x)
            loss = self.criterion(reconstructed, x).mean()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * x.size(0)
        return total_loss / len(dataloader.dataset)
        
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device)
                reconstructed = self.model(x)
                loss = self.criterion(reconstructed, x).mean()
                total_loss += loss.item() * x.size(0)
        return total_loss / len(dataloader.dataset)
        
    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        
    def load(self, filepath):
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
