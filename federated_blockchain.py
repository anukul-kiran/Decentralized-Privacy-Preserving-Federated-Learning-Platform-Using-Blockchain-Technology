import hashlib
import json
import os
from time import time
import numpy as np
import pickle

class FederatedBlockchain:
    def __init__(self, node_id):
        self.node_id = node_id
        self.current_transactions = []
        self.chain = []
        self.nodes = set()

        # Load chain or create genesis block
        self.load_chain()
        if not self.chain:
            self.create_genesis_block()

    def create_genesis_block(self):
        """ Create the first block in the blockchain"""
        genesis_block = {
            'index': 1,
            'timstamp': time(),
            'transactions': [],
            'proof': 100,
            'previous_hash': '0'

        }

        self.chain.append(genesis_block)
        self.save_chain()

    def save_chain(self):
        """Save the blockchain to a file specific to this node"""
        os.makedirs('blockchain_nodes', exist_ok=True)
        filepath = f'blockchain_nodes/blockchain_{self.node_id}.json'
        with open(filepath, 'w') as file:
            serializable_chain = []
            for block in self.chain:
                serializable_block = block.copy()
                serializable_block['transactions'] = [
                    {**tx, 'model_weights': self._serialize_weights(tx['model_weights'])
                     if 'model_weights' in tx else tx}
                     for tx in serializable_block['transactions']
                ]

                serializable_chain.append(serializable_block)

            json.dump(serializable_chain, file)

    
    def load_chain(self):
        """Load the blockchain from a file specific to this node"""
        filepath = f'blockchain_nodes/blockchain_{self.node_id}.json'
        if os.path.exists(filepath):
            with open(filepath, "r") as file:
                loaded_chain = json.load(file)

                # Deserialize the model weights
                for block in loaded_chain:
                    for tx in block['transactions']:
                        if 'model_weights' in tx:
                            tx['model_weights'] = self._deserialize_weights(tx['model_weights'])
                self.chain = loaded_chain


    def _serialize_weights(self, weights):
        """ Serialize weights to base64 for json storage"""
        if weights is None:
            return None
        
        return {k: w.tolist() if isinstance(w, np.ndarray) else w for k, w in weights.items()}
    
    def _deserialize_weights(self, serialized_weights):
        """Deserialize weights back to numpy arrays"""
        if serialized_weights is None:
            return None
        
        return {k: np.array(w) if isinstance(w, list) else w for k, w in serialized_weights.items()}
    
    def new_model_transaction(self, sender, model_weights, update_type):
        """ Create a new model transaction for model weights"""

        transaction = {
            'sender': sender,
            'model_weights': model_weights,
            'update_type': update_type,
            'timestamp': time()
        }

        self.current_transactions.append(transaction)
        return self.last_block['index'] + 1
    

    def new_block(self, proof, previous_hash=None):
        """ Create a new block in the blockchain"""
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1])

        }

        # Reset current_trasactions
        self.current_transactions = []
        self.chain.append(block)
        self.save_chain()
        return block
    
    @property
    def last_block(self):
        """Return the last block in the chain"""
        return self.chain[-1]
    
    @staticmethod
    def hash(block):
        """ Create a SHA-256 hash of a block"""

        # Create a copy to avoid modifying the original block
        block_copy = block.copy()

        # Convert model weights to a hashable format
        if 'transactions' in block_copy:
            block_copy['transactions'] = [
                {**tx, 'model_weights':
                 {k: w.tobytes() if isinstance(w, np.ndarray) else w
                  for k, w in tx.get('model_weights', {}).items()}}
                for tx in block_copy['transactions']          
            ]

        # sort keys to ensure consistent hashing
        block_string = json.dumps(block_copy, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    def proof_of_work(self, last_block):
        """Simple proof of work algorithm"""
        last_proof = last_block['proof']
        last_hash = self.hash(last_block)

        proof = 0
        while not self.valid_proof(last_proof, proof, last_hash):
            proof += 1
        
        return proof
    
    @staticmethod
    def valid_proof(last_proof, proof, last_hash):
        """validate the proof"""
        guess = f'{last_proof}{proof}{last_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"
    

    def get_latest_model_weights(self):
        """Retrieve the most recent model weights from the blockchain"""

        for block in reversed(self.chain):
            for transaction in reversed(block['transaction']):
                if 'model_weights' in transaction:
                    return transaction['model_weights']
                
        return None
    
    
                
        