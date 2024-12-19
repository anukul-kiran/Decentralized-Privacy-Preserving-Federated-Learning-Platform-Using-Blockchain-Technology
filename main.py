import threading
import numpy as np
from blockchain import Blockchain
from node import Node

def train_node(node):
    print(f"Node {node.node_id} is starting training")

    # Perform local training
    node.train()

    # Update the blockchain with the trained model weights
    print(f'Node {node.node_id} is updating the blockchain')
    node.update_blockchain()

    print(f"Node {node.node_id} has completed training")


def main():
    # Initialize the blockchain
    blockchain = Blockchain()

    # Example synthetic dataset(X, y)
    X = np.random.randn(1000, 64)
    y = np.random.randint(0, 3, size = 1000)

    # Create and start nodes using multithreading
    nodes = []
    threads = []
    for i in range(3):
        node = Node(node_id=1, blockchain=blockchain, dataset=(X[i * 300:(i + 1) * 300], y[i * 300:(i + 1) * 300]))
        nodes.append(node)
        thread = threading.Thread(target=train_node, args=(node,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

    print("Global model updated and stored in blockchain")
    
if __name__ == "__main__":
    main()