import hashlib
import json
from time import time

class Blockchain:
    def __init__(self):
        self.current_transactions = []
        self.chain = []

        # Create genesis block
        self.new_block(previous_hash='1', proof=100, model_update_data = {})

    def valid_chain(self, chain):
        """Determine if a given blockchain is valid"""
        last_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]

            # Check whether the hash of the block is correct
            last_block_hash = self.hash(last_block)
            if block['previous_hash'] != last_block_hash:
                return False
            
            last_block = block
            current_index += 1

            
        return True
    
    def new_block(self, proof, previous_hash, model_update_data):
        """Create a new block in the Blockchain"""
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'model_update_data': model_update_data,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),

        }

        # Reset the current list of transactions
        self.current_transactions = []

        self.chain.append(block)
        return block
    
    def new_transaction(self, sender, recipient, weights, biases, sender_hash):
        """Creates a neew transaction to send and receive the weights and biases"""
        self.current_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'weights': weights,
            'biases': biases,
            'sender_hash': sender_hash,
        })

        return self.last_block['index'] + 1
    

    def hash(self, block):
        """Creates a SHA-256 hash of a block"""

        # Must make sure that the dictionary is ordered 
        block_string = json.dump(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    @staticmethod
    def valid_proof(last_proof, proof, last_hash):
        """validates the Proof"""

        guess = f'{last_proof}{proof}{last_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"
    
    def proof_of_work(self, last_block):
        """Simple Proof of Work Algorithm"""
        last_proof = last_block['proof']
        last_hash = self.hash(last_block)

        proof = 0
        while self.valid_proof(last_proof, proof, last_hash) is False:
            proof += 1

        return proof


    def get_model_data(self):
        """Retrieve global model's weights and training parameters"""
        latest_block = self.chain[-1]
        model_update_data = latest_block['model_update_data']
        return model_update_data

    
    def update_model(self, weights, biases, batch_size, epochs, learning_rate):
        """Store the updated model parameters and training configuration"""
        model_update_data = {
            'weights': weights,
            'biases': biases,
            'batch_size': batch_size,
            'epochs': epochs,
            'learning_rate': learning_rate,
        }

        # Update the blockchain with the new model data
        last_block = self.chain[-1]
        proof = self.proof_of_work(last_block)
        previous_hash = self.hash(last_block)

        self.new_block(proof, previous_hash, model_update_data)


        