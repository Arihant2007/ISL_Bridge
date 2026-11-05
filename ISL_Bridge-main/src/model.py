import torch.nn as nn

#defining model
class ISLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(126,256) #(in,out)
        self.fc2 = nn.Linear(256,128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64,34)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self,x):
        x = self.fc1(x) #input went in thru input layer
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc4(x) #out thru the output layer
        return x