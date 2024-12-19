from global_model import GlobalModel
import torch
import torch.optim as optim
import numpy as np

class Node:
    def __init__(self, node_id, blockchain, dataset):
        self.node_id = node_id
        self.blockchain = blockchain
        self.dataset = dataset

        # Initialize the model from the blockchain
        self.model = GlobalModel(self.blockchain)

        # Training parameters
        self.batch_size = self.blockchain.get_model_data()['batch_size']
        self.epochs = self.blockchain.get_model_data()['epochs']
        self.learning_rate = self.blockchain.get_model_data()['learning_rate']
        self.optimizer = optim.Adam(self.model.parameters, lr = self.learning_rate)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def load_data(self):
        """Load dataset into DataLoader"""
        X, y = self.dataset
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tensor, y_tensor), batch_size = self.batch_size)
    
    def train(self):
        """Train the model on it's dataset"""
        dataloader = self.load_data()

        for epoch in range(self.epochs):
            running_loss = 0.0
            for inputs, labels in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(f"Node {self.node_id}, Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}")

    def update_model(self):
        """Update the blockchain with the new model weights"""
        weights = {
            'fc1': self.model.fc1.weight.detach().numpy(),
            'fc2': self.model.fc2.weight.detach().numpy(),
            'fc3': self.model.fc3.weight.detach().numpy(),
        }

        biases = {
            'fc1': self.model.fc1.bias.detach().numpy(),
            'fc2': self.model.fc2.bias.detach().numpy(),
            'fc3': self.model.fc3.bias.detach().numpy(),
        }

        self.blockchain.update_model(weights, biases, self.batch_size, self.epochs, self.learning_rate)

def train_node(node):
    """Train a node in the blokchain"""
    node.train()
    node.update_model()


