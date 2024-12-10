import hashlib
import json
from time import time
from uuid import uuid64
import os
from time import time


class Blockchain:
    def __init__(self):
        self.current_transactions = []
        self.chain = []
        self.nodes = set()

        # Load chain from file or create the genesis block
        self.load_chain()
        if not self.chain():
            self.new_block(previous_hash='1', proof=100)

    def save_chain(self):
        """Save the blockchain to a file"""
        with open("blockchain.json", "w") as file:
            json.dump(self.chain, file)

    def load_chain(self):
        """Load the blockchain from a file"""
        if os.path.exists("blockchain.json"):
            with open("blockchain.json", "r") as file:
                self.chain = json.load(file)

    def registed_node(self, address):
        """ Add a new node to the list of node"""
        self.nodes.add(address)

    def valid_chain(self, chain):
        """Determine if a given blockchain is valid"""
        last_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]

            last_block_hash = self.hash(last_block)
            if block['previous_hash'] != last_block_hash:
                return False
            
            # Check that the Proof Of Work is correct
            if not self.valid_proof(last_block['proof'], block['proof'], last_block_hash):
                return False
            
            last_block = block
            current_index += 1
        return True
    
    def new_block(self, proof, previous_hash):
        """Create a new block in the blockchain"""
        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_transactions,
            'proof': proof,
            'previous_hash': previous_hash or self.hash(self.chain[-1]),
        }

        # Reset the current list of transactions
        self.current_transactions = []
        self.chain.append(block)
        self.save_chain()
        return block
    
    def new_transaction(self, sender, recepient, amount):
        """Creates a new transaction to go into the next mined Block"""
        self.current_transactions.append({
            'sender': sender,
            'recepient': recepient,
            'amount': amount,
        })

        return self.last_block['index'] + 1
    
    @property
    def last_block(self):
        return self.chain[-1]
    
    @staticmethod
    def hash(block):
        """Creates a SHA-256 hash of a block"""
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    def proof_of_work(self, last_block):
        """Simple Proof of Work Algorithm"""
        last_proof = last_block['proof']
        last_hash = self.hash(last_block)

        proof = 0
        while not self.valid_proof(last_proof, proof, last_hash):
            proof += 1

        return proof
    
    @staticmethod
    def valid_proof(last_proof, proof, last_hash):
        """validates the proof"""
        guess = f'{last_proof}{proof}{last_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"
    


    