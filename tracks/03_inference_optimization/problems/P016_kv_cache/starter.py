class KVCache:
    def __init__(self):
        self.keys = []
        self.values = []

    def append(self, key, value):
        self.keys.append(key)
        self.values.append(value)

    def get(self):
        return self.keys, self.values

    def reset(self):
        self.keys = []
        self.values = []
