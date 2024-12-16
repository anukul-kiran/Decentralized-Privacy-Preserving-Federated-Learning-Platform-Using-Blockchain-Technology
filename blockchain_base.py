from time import time
import json
import hashlib


class Blockchain:
    def __init__(self):
        self.chain = []
        self.current_transaction = []
        
        # Genesis Block
        self.new_block(previous_hash='1', proof = 100)

    def new_block(self, previous_hash, proof):
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }

        # Reset the current list of transactions
        self.current_transaction = []

        self.chain.append(block)
        return block
    
    def valid_chain(self, chain):
        last_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]

            # Check the hash of the block is correct
            last_block_hash = self.hash(last_block)
            if block['previous_hash'] != last_block_hash:
                return False
            
            # Check that the Proof of Work is correct
            if not self.valid_proof(last_block['proof'], block['proof'], last_block_hash):
                return False

            last_block = block
            current_index += 1

        return True
    

    def save_chain(self):
        with open('blockchain.json', 'w') as json_file:
            json.dump(self.chain, json_file)
    
            
    @staticmethod
    def valid_proof(last_proof, proof, last_hash):
        guess = f'{last_proof}{proof}{last_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"
    

    @staticmethod
    def hash(block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    @property
    def last_block(self):
        return self.chain[-1]
    

    

    
