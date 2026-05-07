# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Attention(nn.Module):
#     """Attention mechanism to focus on semantically important frames[cite: 16, 118]."""
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.attn = nn.Linear(hidden_dim, 1)

#     def forward(self, x):

#         weights = torch.tanh(self.attn(x))
#         weights = F.softmax(weights, dim=1)
       
#         context = torch.sum(weights * x, dim=1)
#         return context

# class SLRHybridModel(nn.Module):
#     def __init__(self, num_classes, input_dim=99, hidden_dim=64):
#         super().__init__()
        

#         self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
#         self.pool = nn.MaxPool1d(2)
        

#         self.lstm = nn.LSTM(
#             input_size=64, 
#             hidden_size=hidden_dim, 
#             num_layers=1, 
#             batch_first=True, 
#             bidirectional=True
#         )
        

#         self.attention = Attention(hidden_dim * 2)
        

#         self.classifier = nn.Sequential(
#             nn.Linear(hidden_dim * 2, 64),
#             nn.ReLU(),
#             nn.Dropout(0.4),
#             nn.Linear(64, num_classes)
#         )

#     def forward(self, x):

#         x = x.transpose(1, 2)
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
        
#         x = x.transpose(1, 2) 
#         x, _ = self.lstm(x)
        

#         x = self.attention(x)
        
#         return self.classifier(x)






import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.tanh(self.attn(x))
        weights = F.softmax(weights, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context

class SLRHybridModel(nn.Module):
    def __init__(self, num_classes, input_dim=99, hidden_dim=128):
        super().__init__()
        
        
        self.conv_block = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        self.lstm = nn.LSTM(
            input_size=128, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.3
        )

        self.attention = Attention(hidden_dim * 2)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
     
        x = x.transpose(1, 2) 
        
        
        x = self.conv_block(x)
        
        
        x = x.transpose(1, 2) 
  
        x, _ = self.lstm(x)
        x = self.attention(x)
        
        return self.classifier(x)