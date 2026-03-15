import hashlib


class PromptCache:
    def __init__(self):
        raise NotImplementedError("Initialize this class in the starter.")

    def make_key(self, prompt: str):
        raise NotImplementedError("Implement P048.make_key().")

    def get(self, prompt: str):
        raise NotImplementedError("Implement P048.get().")

    def put(self, prompt: str, value):
        raise NotImplementedError("Implement P048.put().")
