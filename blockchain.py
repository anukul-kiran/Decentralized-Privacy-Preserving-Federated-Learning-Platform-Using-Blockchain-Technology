from time import time
import hashlib
import json

class Blockchain:
    def __init__(self):
        self.current_transactions = []
        self.chain = []
        
        # Create genesis block
        self.new_block(previous_hash='1', proof=100)


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
            
        return True
    
    def new_block(self, proof, previous_hash, model_update_data):
        """Create a new Block in the Blockchain"""
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
    
    def new_transaction(self, sender, recipient, weights, biases):
        """Creates a new transaction to send and receive the weights and biases"""
        self.current_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'weights': weights,
            'biases': biases,
        })

        return self.last_block['index'] + 1
    
    @property
    def hash(block):
        """ Creates a SHA-256 hash of a block"""
        
        # Must make sure that the dictionary is ordered or we'll have inconsistent hashes
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    @staticmethod
    def valid_proof(last_proof, proof, last_hash):
        """Validates the Proof"""

        guess = f'{last_proof}{proof}{last_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"
    

    def proof_of_work(self, last_block):
        """Simple proof of work algorithm
            - Find a number p' such that hash(pp') contains leading 4 zeros
            - Where p is the previous proof, and p' is the new proof

            :param last_block: <dict> last Block
            :return: <int>
            """
        last_proof = last_block['proof']
        last_hash = self.hash(last_block)

        proof = 0
        while self.valid_proof(last_proof, proof, last_hash) is False:
            proof += 1

        return proof
    

    


    
    
