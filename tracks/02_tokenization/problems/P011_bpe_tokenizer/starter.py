class BPETokenizer:
    def __init__(self, vocab_size: int):
        raise NotImplementedError("Initialize this class in the starter.")

    def train(self, corpus):
        raise NotImplementedError("Implement P011.train().")

    def encode(self, text: str):
        raise NotImplementedError("Implement P011.encode().")

    def decode(self, tokens):
        raise NotImplementedError("Implement P011.decode().")
