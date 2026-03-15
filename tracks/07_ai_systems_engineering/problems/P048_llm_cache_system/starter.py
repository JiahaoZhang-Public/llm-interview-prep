import hashlib


class PromptCache:
    def __init__(self):
        self.cache = {}

    def make_key(self, prompt: str):
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get(self, prompt: str):
        key = self.make_key(prompt)
        return self.cache.get(key)

    def put(self, prompt: str, value):
        key = self.make_key(prompt)
        self.cache[key] = value
