import json
from pathlib import Path
from web3 import Web3

BASE_DIR = Path(__file__).resolve().parent
ABI_PATH = BASE_DIR / "contracts" / "MFCCOracle.abi.json"


def load_oracle_contract(w3: Web3, address: str):
    """Load the MFCCOracle contract using the local ABI file."""
    with ABI_PATH.open() as f:
        abi = json.load(f)
    return w3.eth.contract(address=Web3.to_checksum_address(address), abi=abi)


if __name__ == "__main__":
    # Example usage: connect to a local node and load the contract.
    web3 = Web3(Web3.HTTPProvider("http://localhost:8545"))
    # Replace with the deployed contract address.
    oracle_address = "0x0000000000000000000000000000000000000000"
    contract = load_oracle_contract(web3, oracle_address)
    print("Loaded contract at", contract.address)
