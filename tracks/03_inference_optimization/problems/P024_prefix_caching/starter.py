class PrefixCache:
    def __init__(self):
        self.cache = {}

    def add_prefix(self, prefix_tokens, value):
        self.cache[tuple(prefix_tokens)] = value

    def lookup(self, prefix_tokens):
        prefix_tokens = tuple(prefix_tokens)
        best_value = None
        best_len = 0
        for key, value in self.cache.items():
            klen = len(key)
            if klen <= len(prefix_tokens) and prefix_tokens[:klen] == key:
                if klen > best_len:
                    best_len = klen
                    best_value = value
        return best_value
