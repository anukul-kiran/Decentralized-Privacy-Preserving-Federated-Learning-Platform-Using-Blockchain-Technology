from blockchain import Blockchain
import json


def main():
    blockchain = Blockchain()

    while True:
        print("\nMenu")
        print("1.View Blockchain")
        print("2.Add Transaction")
        print("3.Mine Block")
        print("4.Exit")

        choice = input("Enter you choice")

        if choice == "1":
            print("\n Blockchain")
            for block in blockchain.chain:
                print(json.dumps(block, indent=4))
        elif choice == "2":
            sender = input("Enter sender")
            recepient = input("Enter recepients")
            amount = input("Enter amount")
            index = blockchain.new_transaction(sender, recepient, amount)
            print(f"Transaction will be added to Block {index}")

        elif choice == "3":
            last_block = blockchain.last_block
            proof = blockchain.proof_of_work(last_block)
            previous_hash = blockchain.hash(last_block)
            block = blockchain.new_block(proof, previous_hash)
            print(f"New Block Forged:\n{json.dumps(block, indent=4)}")

        elif choice == "4":
            print("Exiting...")
            break
        else:
            print("Ivalid choice! Please try again")


if __name__ == "__main__":
    main()
