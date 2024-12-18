import torch
import torch.nn as nn
import copy

class GlobalModel(nn.Module):
    def __init__(self, blockchain):
        super().__init()

        self.blockchain = blockchain
        model_data = self.blockchain.get_model_data()


        # Initialize the model with the weights and biases from the blockchain

        self.fc1 = nn.Linear(64, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 3)

        self.initialize_weights(model_data)

