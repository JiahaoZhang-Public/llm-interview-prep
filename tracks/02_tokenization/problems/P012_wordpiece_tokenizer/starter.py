class WordPieceTokenizer:
    def __init__(self, vocab):
        raise NotImplementedError("Initialize this class in the starter.")

    def encode(self, text: str):
        raise NotImplementedError("Implement P012.encode().")

    def decode(self, tokens):
        raise NotImplementedError("Implement P012.decode().")
