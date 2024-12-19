import torch
import torch.nn as nn

class GlobalModel(nn.Module):
    def __init__(self, blockchain):
        super().__init__()

        self.blockchain = blockchain
        model_data = self.blockchain.get_model_data()


        # Initialize the model with the weights and biases from the blockchain

        self.fc1 = nn.Linear(64, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 3)

        self.initialize_weights(model_data)

    def initialize_weights(self, model_data):
        """Initialize model weights and biases from blockchain"""
        weights = model_data['weights']
        biases = model_data['biases']
        with torch.no_grad():
            self.fc1.weight = nn.Parameter(torch.tensor(weights['fc1'], dtype=torch.float32))
            self.fc1.bias = nn.Parameter(torch.tensor(biases['fc1'], dtype=torch.float32))
            self.fc2.weight = nn.Parameter(torch.tensor(weights['fc2'], dtype=torch.float32))
            self.fc2.bias = nn.Parameter(torch.tensor(biases['fc2'], dtype=torch.float32))
            self.fc3.weight = nn.Parameter(torch.tensor(weights['fc3'], dtype=torch.float32))
            self.fc3.bias = nn.Parameter(torch.tensor(biases['fc3'], dtype=torch.float32))

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x
    

