class RoundTripTokenizer:
    def __init__(self):
        raise NotImplementedError("Initialize this class in the starter.")

    def fit(self, corpus):
        raise NotImplementedError("Implement P014.fit().")

    def encode(self, text: str):
        raise NotImplementedError("Implement P014.encode().")

    def decode(self, token_ids):
        raise NotImplementedError("Implement P014.decode().")
