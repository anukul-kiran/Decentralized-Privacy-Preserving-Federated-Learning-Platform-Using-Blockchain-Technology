import hashlib
import json
from time import time
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("blockchain.log", mode="a")]
)

class Blockchain:
    def __init__(self):
        self.current_transactions = []
        self.chain = []

        # Create genesis block
        self.new_block(previous_hash='1', proof=100, model_update_data = {})
        logging.info("Genesis Block created")

    def valid_chain(self, chain):
        """Determine if a given blockchain is valid"""
        logging.info("validating the blockchain")
        last_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]

            # Check whether the hash of the block is correct
            last_block_hash = self.hash(last_block)
            if block['previous_hash'] != last_block_hash:
                logging.error("Invalid block detected at index {current_index}.")
                return False
            
            last_block = block
            current_index += 1

        logging.info("Blockchain is valid")   
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
        logging.info(f"New Block created with index {block['index']}.")
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

        logging.info(f"Transaction added: sender: {sender}, recipient: {recipient}.")
        return self.last_block['index'] + 1
    

    def hash(self, block):
        """Creates a SHA-256 hash of a block"""

        # Must make sure that the dictionary is ordered 
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()
    
    @staticmethod
    def valid_proof(last_proof, proof, last_hash):
        """validates the Proof"""

        guess = f'{last_proof}{proof}{last_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        is_valid = guess_hash[:4] == "0000"
        if is_valid:
            logging.info("Proof is valid: {proof}.")
        return is_valid
    
    def proof_of_work(self, last_block):
        """Simple Proof of Work Algorithm"""
        last_proof = last_block['proof']
        last_hash = self.hash(last_block)

        proof = 0
        while self.valid_proof(last_proof, proof, last_hash) is False:
            proof += 1

        logging.info(f"Proof of work completed: {proof}")
        return proof


    def get_model_data(self):
        """Retrieve global model's weights and training parameters"""
        latest_block = self.chain[-1]
        model_update_data = latest_block['model_update_data']
        logging.info("Retrieved model data from the latest block")
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
        logging.info("Model updated on the blockchain")
    
    @property
    def last_block(self):
        """Displays the last block in the chain"""
        last_block = self.chain[-1]
        logging.info("Retrieved the last block")
        return last_block
    
    def save_chain(self):
        """Saves the blockchain"""
        try:
            with open("blockchain.json", "w", exist_ok=True) as blockchain:
                json.dump(self.chain, blockchain, indent=4)
            logging.info("Blockchain saved to 'blockchain.json")
        except:
            raise Exception 

    def load_chain(self, filename="blockchain.json"):
        """Loads the blockchain from a file"""
        try:
            with open(filename, "r") as blockchain:
                self.chain = json.load(blockchain)
        except FileNotFoundError:
            print("Blockchain file not found. Starting a new blockchain")
    
    def display_chain(self, index):
        """Function used to display a specific block in the chain"""
        if index < 0 or index >= len(self.chain):
            return f"Invalid block index. Blockchain lenght: {len(self.chain)}"
        return self.chain[index]
    


        