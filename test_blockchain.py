import unittest
import requests
import json
from blockchain import Blockchain, generate_keys

class TestBlockchain(unittest.TestCase):
    def setUp(self):
        self.blockchain = Blockchain()
        self.base_url = "http://localhost:8000"

        # sample test data
        self.test_weights = [0.1, 0.2, 0.3]
        self.test_biases = [0.01, 0.02]
        self.test_sender = "test_sender"
        self.test_recipient = "test_receipient"

    def test_genesis_block(self):
        """Test if genesis block is created correctly"""
        self.assertEqual(len(self.blockchain.chain), 1)
        genesis = self.blockchain.chain[0]
        self.assertEqual(genesis['previous_hash'], '1')
        self.assertEqual(genesis['proof'], 100)

    def test_key_generation(self):
        """Test key pair generation"""
        private_key, public_key = generate_keys()
        self.assertIsNotNone(private_key)
        self.assertIsNotNone(public_key)
        self.assertIn("BEGIN RSA PRIVATE KEY", private_key)
        self.assertIn("BEGIN PUBLIC KEY", public_key)

    def test_new_transaction(self):
        """Test creating a new transaction"""
        index = self.blockchain.new_transaction(
            self.test_sender,
            self.test_recipient,
            self.test_weights,
            self.test_biases
        )

        self.assertIsNotNone(index)
        self.assertEqual(len(self.blockchain.current_transactions), 1)

    def test_proof_of_work(self):
        """Test Proof of work calculation"""
        last_block = self.blockchain.last_block
        proof = self.blockchain.proof_of_work(last_block)
        self.assertTrue(self.blockchain.valid_proof(
            last_block['proof'],
            proof,
            self.blockchain.hash(last_block)
        ))

    def test_mining(self):
        """Test block mining"""

        # Add a transaction first
        self.blockchain.new_transaction(
            self.test_sender,
            self.test_recipient,
            self.test_weights,
            self.test_biases
        )

        # Mine a Block
        last_block = self.blockchain.last_block
        proof = self.blockchain.proof_of_work(last_block)
        previous_hash = self.blockchain.hash(last_block)
        block = self.blockchain.new_block(proof, previous_hash, {})

        self.assertEqual(len(self.blockchain.chain), 2)
        self.assertEqual(block['previous_hash'], previous_hash)

    def test_model_update(self):
        """Test model update functionality"""
        model_data = {
            'weights': self.test_weights,
            'biases': self.test_biases,
            'batch_size': 32,
            'epochs': 10,
            'learning_rate': 0.001
        }

        self.blockchain.update_model(
            model_data['weights'],
            model_data['biases'],
            model_data['batch_size'],
            model_data['epochs'],
            model_data['learning_rate']
        )

        current_model = self.blockchain.get_model_data()
        self.assertEqual(current_model['weights'], model_data['weights'])
        self.assertEqual(current_model['biases'], model_data['biases'])

    def test_chain_validation(self):
        """Test blockchain validation"""
        
        # Create a valid chain
        self.assertEqual(self.blockchain.valid_chain(self.blockchain.chain))

        # Tamper with the chain
        invalid_chain = self.blockchain.chain.copy()
        invalid_chain[0]['proof'] = 999999
        self.assertFalse(self.blockchain.valid_chain(invalid_chain))


    def test_save_load_chain(self):
        """Test saving and loading the blockchain"""
        # Add some data
        self.blockchain.new_transaction(
            self.test_sender,
            self.test_recipient,
            self.test_weights,
            self.test_biases
        )

        original_chain = self.blockchain.copy()

        # Save and load
        self.blockchain.save_chain()
        self.blockchain.chain = []
        self.blockchain.load_chain()

        self.assertEqual(len(self.blockchain.chain), len(original_chain))
        self.assertEqual(
            self.blockchain.hash(self.blockchain.chain[0]),
            self.blockchain.hash(original_chain[0])
        )


def run_api_tests():
    """Run tests against the running API server"""

    base_url = "http://localhost:8000"

    # Test Root Endpoint
    response = requests.get(f"{base_url}/")
    assert response.status_code == 200

    # Test chain endpoint
    response = requests.get(f"{base_url}/chain")
    assert response.status_code == 200

    # Test new transaction
    transaction_data = {
        'sender': 'test_sender',
        'recepient': 'test_recipient',
        'weights': [0.1, 0.2, 0.3],
        'biases': [0.01, 0.02]
    }

    response = requests.post(f"{base_url}/transaction/new", json=transaction_data)
    assert response.status_code == 200

    # Test mining endpoint
    response = requests.post(f"{base_url}/mine")
    assert response.status_code == 200

    # Test model update
    model_data = {
        'weights': [0.1, 0.2, 0.3],
        'biases': [0.01, 0.02],
        'batch_size': 32,
        'epochs': 10,
        'learning_rate': 0.001
    }

    response = requests.post(f"{base_url}/model/update", json=model_data)
    assert response.status_code == 200

    print("API tests completed successfully")

if __name__ == "__main__":
    blockchain = Blockchain()
    blockchain.run_server()

    unittest.main(verbosity=2)

    # Run api tests
    try:
        run_api_tests()
    except requests.exceptions.ConnectionError:
        print("Please start the blockchain server first to run the API tests")
        
