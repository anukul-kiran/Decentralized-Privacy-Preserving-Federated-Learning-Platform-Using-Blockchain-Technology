import hashlib
import json
from time import time
import logging
from Crypto.Hash import SHA256
from Crypto.Signature import PKCS1_v1_5
from Crypto.PublicKey import RSA
import binascii
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("blockchain.log", mode="a")]
)

# Define pydantic model for request validation
class TransactionModel(BaseModel):
    sender: str
    recipient: str
    weights: List[float]
    biases: List[float]

class ModelUpdateModel(BaseModel):
    weights: List[float]
    biases: List[float]
    batch_size: int
    epochs: int
    learning_rate: float


def generate_keys():
    """Generate a pair of RSA keys (private and public)"""
    key = RSA.generate(2048)
    private_key = key.export_key().decode('utf8')
    public_key = key.publickey().export_key().decode('utf8')
    return private_key, public_key

class Blockchain:
    def __init__(self):
        self.current_transactions = []
        self.chain = []
        self.participants = {}
        self.app = FastAPI(title="Blockchain API")


        # Create genesis block
        self.new_block(previous_hash='1', proof=100, model_update_data = {})
        logging.info("Genesis Block created")

        # Setup API routes
        self.setup_routes()

    def setup_routes(self):
        """Setup FastAPI routes"""

        @self.app.get("/")
        def read_root():
            return {"status": "active", "blocks": len(self.chain)}
        
        @self.app.get("/chain")
        def get_chain():
            return {"chain": self.chain, "length": len(self.chain)}
        
        @self.app.get("/chain/{index}")
        def get_block(index: int):
            try:
                return {"block": self.display_chain(index)}
            except:
                raise HTTPException(status_code=404, detail="Block not Found")
        
        @self.app.post("/transaction/new")
        def new_transaction(transaction: TransactionModel):
            index = self.new_transaction(
                transaction.sender,
                transaction.recipient,
                transaction.weights,
                transaction.biases
            )

            if index is None:
                raise HTTPException(status_code=400, detail="Invalid Transaction")
            return {"message": f"Transaction will be added to the Block {index}"}
        
        @self.app.post("/mine")
        def mine():
            last_block = self.last_block
            proof = self.proof_of_work(last_block)
            previous_hash = self.hash(last_block)

            block = self.new_block(proof, previous_hash, {})

            return {
                'message': 'New Block Forged',
                'index': block['index'],
                'transactions': block['transaction'],
                'proof': block['proof'],
                'previous_hash': block['previous_hash'],

            }
        
        
        @self.app.post('/model/update')
        def update_model(model_data: ModelUpdateModel):
            self.update_model(

                model_data.weights,
                model_data.biases,
                model_data.batch_size,
                model_data.batch_size,
                model_data.epochs,
                model_data.learning_rate

            )

            return {'message': 'Model updated successfully'}
        
        @self.app.get('/model/current')
        def get_model():
            return {'model_data': self.get_model_data()}
        
        @self.app.post('/save')
        def save_blockchain():
            self.save_chain()
            return {'message': 'Blockchain saved successfully'}
        
        @self.app.post('/load')
        def load_blockchain():
            self.load_chain()
            return {"message": "Blockchain loaded successfully"}
    

    def valid_chain(self, chain):
        """Determine if a given blockchain is valid by checking
           1. Hash links between blocks
           2. Proof of work for each block
           3. Block structure and contents
        """

        # First verify the genesis block
        if len(chain) == 0:
            logging.error("Empty chain")
            return False
        
        genesis = chain[0]
        if genesis['previous_hash'] != '1' or genesis['proof'] != 100:
            logging.error("Invalid genesis block")
            return False
        
        # Then verify the rest of the chain
        for i in range(1, len(chain)):
            current_block = chain[i]
            previous_block = chain[i - 1]

            # Check block structure
            required_keys = {'index', 'timestamp', 'transactions', 'proof', 'model_update_data', 'previous_hash'}
            if not all(key in current_block for key in required_keys):
                logging.error(f"Invalid Block structure at index {i}")
                return False
            
            # Check Block index continuity
            if current_block['index'] != previous_block['index'] + 1:
                logging.error(f"Invalid block index at position {i}")
                return False
            
            # Check hash link
            if current_block['previous_hash'] != self.hash(previous_block):
                logging.error(f"Invalid hash link at index {i}")
                return False
            
        logging.info("Blockchain is valid")
        return True
    
    def verify_block_proof(self, previous_block, current_block):
        """Verify the proof of work for a block"""
        guess = f"{previous_block['proof']}{current_block['proof']}{previous_block['previous_hash']}".encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"
    
    @staticmethod
    def valid_proof(last_proof, proof, last_hash):
        """Validates the proof"""
        guess = f'{last_proof}{proof}{last_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"


    def verify_transaction_structure(self, transaction):
        """Verify that a transaction has all required fields and valid signatures"""
        required_fields = {'sender', 'recipient', 'weights', 'biases', 'signature', 'sender_public_key'}
        
        # Check if all required fields are present
        if not all(field in transaction for field in required_fields):
            logging.error("Transaction missing required fields")
            return False
        
        # Verify data types
        if not isinstance(transaction['weights'], list) or not isinstance(transaction['biases'], list):
            logging.error("Invalid data types in transaction")
            return False
        
        # Verify signature
        try:
            return self.verify_signature(transaction)
        except Exception as e:
            logging.error(f"Signature verification failed: {e}")
            return False
    
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
    
    def new_transaction(self, sender, recipient, weights, biases):
        """Creates a neew transaction to send and receive the weights and biases"""

        sender_keys = self.get_or_create_keys(sender)
        sender_private_key = sender_keys['private_key']
        sender_public_key = sender_keys['public_key']

        transaction_data = {
            'sender': sender,
            'recipient': recipient,
            'weights': weights,
            'biases': biases,
        }

        signature = self.sign_transaction(sender_private_key, transaction_data)
        

        transaction = {
            'sender': sender,
            'recipient': recipient,
            'weights': weights,
            'biases': biases,
            'signature': signature,
            'sender_public_key': sender_public_key,
        }

        if self.verify_signature(transaction):
            self.current_transactions.append(transaction)
            logging.info(f"Transaction added: {transaction}") 
            return self.last_block['index'] + 1
        else:
            logging.warning("Invalid transaction signature. Transaction discarded")
            return None

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
            logging.info(f"Proof is valid: {proof}.")
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
            with open("blockchain.json", "w") as blockchain:
                json.dump(self.chain, blockchain, indent=4)
            logging.info("Blockchain saved to 'blockchain.json")
        except Exception as e:
            logging.error(f"Failed to save blockchain: {e}")
            raise HTTPException(status_code=500, detail="Failed to save the blockchain")

    def load_chain(self, filename="blockchain.json"):
        """Loads the blockchain from a file"""
        try:
            with open(filename, "r") as blockchain:
                self.chain = json.load(blockchain)
            logging.info("Blockchain loaded from file")
        except FileNotFoundError:
            logging.warning("Blockchain file not found. Starting a new Blockchain")
            raise HTTPException(status_code=404, detail="Blockchain file not found")
    
    def display_chain(self, index):
        """Function used to display a specific block in the chain"""
        if index < 0 or index >= len(self.chain):
            return f"Invalid block index. Blockchain lenght: {len(self.chain)}"
        return self.chain[index]
    
    def verify_signature(self, transaction):
        """verify the transaction"""
        try:
            public_key = RSA.import_key(transaction['sender_public_key'])
            verifier = PKCS1_v1_5.new(public_key)
            transaction_data = {
                'sender': transaction['sender'],
                'recipient': transaction['recipient'],
                'weights': transaction['weights'],
                'biases': transaction['biases'],
            }
            h = SHA256.new(json.dumps(transaction_data, sort_keys=True).encode('utf-8'))
            return verifier.verify(h, binascii.unhexlify(transaction['signature']))
        except Exception as e:
            logging.error(f"Signature verification failed: {e}")
            raise HTTPException(status_code=500, detail="Signature Verification failed")
            return False
    

    def sign_transaction(self, private_key, transaction_data):
        """Sign a transaction witht the private key"""
        try:
            key = RSA.import_key(private_key)
            signer = PKCS1_v1_5.new(key)
            h = SHA256.new(json.dumps(transaction_data, sort_keys=True).encode('utf-8'))
            return binascii.hexlify(signer.sign(h)).decode('utf-8')
        except Exception as e:
            logging.error(f"Transaction signing failed: {e}")
            raise HTTPException(status_code=500, detail="Failed to sign transaction")
        

    def get_or_create_keys(self, participant):
        """Retrieve or generate keys for a participant"""
        if participant not in self.participants:
            private_key, public_key = generate_keys()
            self.participants[participant] = {
                'private_key': private_key,
                'public_key': public_key
            }

            logging.info(f"Generated new keys for participant: {participant}")
        return self.participants[participant]
    
    def run_server(self, host="0.0.0.0", port=8000):
        """Run the blockchain server"""
        uvicorn.run(self.app, host=host, port=port)

    




    

    


        